"""Replay recorded audio through the SIRE Foundry Voice Agent and capture everything the evaluators need.

Each case gets its own realtime session (one spoken request, like a badge interaction):

1. connect to the agent endpoint with the case's dialog flow as structured inputs;
2. optionally override session settings: text-only output (skips TTS cost) and/or a different
   transcription (STT) model for STT experiments;
3. stream the WAV plus trailing silence; the agent's own server-side VAD closes the turn and starts
   the response, exactly as in production (manual commit is only a fallback);
4. execute the agent's function calls client-side (multi-strategy RRF search over Azure AI Search)
   and return results until the agent produces its final reply;
5. record the input transcription, tool calls, search candidates, reply and latencies.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_spec import SEARCH_TOOLS, intent_from_tool_call, structured_inputs, tool_definitions_for_evaluators
from .audio import duration_seconds, load_pcm16
from .search_client import SireSearchClient, tool_payload

CHUNK_BYTES = 9600  # 200 ms of 24 kHz mono PCM16
GOLD_KEYS = (
    "case_id", "case_key", "usage", "lane", "flow", "facility", "facility_ids", "asr_label", "gold_transcript",
    "expected_intent", "expected_entities", "expected_tools", "expected_choice", "choice_options",
    "entity_mentions", "caller_urn", "caller_department_ids", "audio_condition", "tags",
)


@dataclass
class RunConfig:
    agent_name: str
    run_label: str
    stt_model: str | None = None
    text_only: bool = True
    concurrency: int = 4
    latency_budget_ms: int = 1500
    max_tool_rounds: int = 4
    case_timeout_s: float = 60.0
    transcript_grace_s: float = 8.0
    trailing_silence_s: float = 1.6
    vad_fallback_s: float = 8.0
    retries: int = 2
    extra_session: dict[str, Any] = field(default_factory=dict)


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "as_dict"):
        return event.as_dict()
    return dict(event)


class CaseSession:
    """State machine for one case's realtime session."""

    def __init__(self, case: dict[str, Any], cfg: RunConfig, search: SireSearchClient) -> None:
        self.case = case
        self.cfg = cfg
        self.search = search
        self.t_commit = 0.0
        self.transcript: str | None = None
        self.transcript_ms: float | None = None
        self.transcript_error: str | None = None
        self.calls: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.first_tool_ms: float | None = None
        self.response_ms: float | None = None
        self.rounds = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "audio_input_tokens": 0}
        self.session_id = ""
        self.conversation_id = ""
        self.errors: list[str] = []
        self.session_ack: dict[str, Any] = {}
        self.eos_seen = False

    def _ms(self) -> float:
        return round((time.perf_counter() - self.t_commit) * 1000, 1)

    async def _execute(self, call: dict[str, Any]) -> None:
        name, args = call["name"], call["arguments"]
        started = time.perf_counter()
        try:
            if name == "search_user":
                result = await self.search.search_user(
                    name=str(args.get("name") or ""), department=str(args.get("department") or ""),
                    doctor=bool(args.get("doctor")), facility_ids=self.case.get("facility_ids"),
                    caller_department_ids=self.case.get("caller_department_ids"))
                call["search"] = result
                call["output"] = tool_payload(result)
            elif name == "search_group":
                result = await self.search.search_group(group=str(args.get("group") or ""),
                                                        facility_ids=self.case.get("facility_ids"))
                call["search"] = result
                call["output"] = tool_payload(result)
            elif name == "route_command":
                call["output"] = {"status": "routed", "intent": args.get("intent")}
            elif name == "pick_choice":
                call["output"] = {"status": "selected", "kind": args.get("kind"), "value": args.get("value")}
            else:
                call["output"] = {"status": "error", "message": f"unknown tool {name}"}
        except Exception as exc:  # noqa: BLE001 - surface tool failures to the model and the report
            call["output"] = {"status": "error", "message": str(exc)[:300]}
            self.errors.append(f"tool {name}: {exc}")
        call["tool_ms"] = round((time.perf_counter() - started) * 1000, 1)

    async def run(self, project) -> None:
        from azure.ai.projects.models import RealtimeConversationItemFunctionCallOutput

        pcm = load_pcm16(self.case["_audio_abs"])
        inputs = structured_inputs(self.case["flow"], self.case.get("choice_options"))
        async with project.beta.voice_agents.realtime.connect(agent_name=self.cfg.agent_name, structured_inputs=inputs) as conn:

            async def recv() -> dict[str, Any]:
                return _event_dict(await asyncio.wait_for(conn.recv(), timeout=self.cfg.case_timeout_s))

            while True:
                ev = await recv()
                kind = ev.get("type")
                if ev.get("conversation_id"):
                    self.conversation_id = ev["conversation_id"]
                if kind == "error":
                    raise RuntimeError(f"session error: {json.dumps(ev.get('error'))[:300]}")
                if kind == "session.updated":
                    self.session_id = (ev.get("session") or {}).get("id", "")
                    self.session_ack = ev.get("session") or {}
                    break

            audio_input: dict[str, Any] = {}
            if self.cfg.stt_model:
                audio_input["transcription"] = {"model": self.cfg.stt_model, "language": "en-US"}
            session: dict[str, Any] = {"type": "realtime"}
            if audio_input:
                session["audio"] = {"input": audio_input}
            if self.cfg.text_only:
                session["output_modalities"] = ["text"]
            session.update(self.cfg.extra_session)
            if len(session) > 1:
                await conn.session.update(session=session)
                while True:
                    ev = await recv()
                    if ev.get("type") == "error":
                        raise RuntimeError(f"session.update rejected: {json.dumps(ev.get('error'))[:300]}")
                    if ev.get("type") == "session.updated":
                        self.session_ack = ev.get("session") or {}
                        break

            # The agent's server-side (semantic) VAD detects end of speech exactly as in production;
            # trailing silence lets it close the turn. Manual commit is only a fallback.
            for offset in range(0, len(pcm), CHUNK_BYTES):
                await conn.input_audio_buffer.append(audio=pcm[offset:offset + CHUNK_BYTES])
            silence = bytes(CHUNK_BYTES)
            for _ in range(int(self.cfg.trailing_silence_s * 5)):
                await conn.input_audio_buffer.append(audio=silence)
            self.t_commit = time.perf_counter()
            sent_at = self.t_commit
            manual = False

            pending: list[asyncio.Task] = []
            round_calls: list[dict[str, Any]] = []
            finished = False
            deadline = time.perf_counter() + self.cfg.case_timeout_s
            while time.perf_counter() < deadline:
                if finished and self.transcript is not None:
                    break
                if finished:
                    timeout = self.cfg.transcript_grace_s
                elif not self.eos_seen and not manual:
                    timeout = max(0.5, self.cfg.vad_fallback_s - (time.perf_counter() - sent_at))
                else:
                    timeout = max(1.0, deadline - time.perf_counter())
                try:
                    ev = _event_dict(await asyncio.wait_for(conn.recv(), timeout=timeout))
                except asyncio.TimeoutError:
                    if finished:
                        self.transcript_error = self.transcript_error or "transcription not received"
                        break
                    if not self.eos_seen and not manual:
                        manual = True
                        self.errors.append("VAD did not close the turn; committed manually")
                        await conn.input_audio_buffer.commit()
                        self.t_commit = time.perf_counter()
                        await conn.response.create()
                        continue
                    raise
                kind = ev.get("type")
                if kind in ("input_audio_buffer.speech_stopped", "input_audio_buffer.committed") and not self.eos_seen:
                    self.eos_seen = True
                    self.t_commit = time.perf_counter()
                elif kind == "conversation.item.input_audio_transcription.completed":
                    self.transcript = (ev.get("transcript") or "").strip()
                    self.transcript_ms = self._ms()
                elif kind == "conversation.item.input_audio_transcription.failed":
                    self.transcript = ""
                    self.transcript_error = json.dumps(ev.get("error"))[:300]
                elif kind == "response.function_call_arguments.done":
                    if self.first_tool_ms is None:
                        self.first_tool_ms = self._ms()
                    try:
                        args = json.loads(ev.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": ev.get("arguments")}
                        self.errors.append("tool arguments were not valid JSON")
                    call = {"call_id": ev.get("call_id"), "name": ev.get("name"), "arguments": args,
                            "t_ms": self._ms(), "round": self.rounds}
                    self.calls.append(call)
                    round_calls.append(call)
                    pending.append(asyncio.create_task(self._execute(call)))
                elif kind in ("response.output_text.done", "response.output_audio_transcript.done", "response.text.done"):
                    text = ev.get("text") or ev.get("transcript") or ""
                    if text.strip():
                        self.texts.append(text.strip())
                elif kind == "response.done":
                    response = ev.get("response") or {}
                    usage = response.get("usage") or {}
                    self.usage["input_tokens"] += usage.get("input_tokens") or 0
                    self.usage["output_tokens"] += usage.get("output_tokens") or 0
                    details = usage.get("input_token_details") or {}
                    self.usage["cached_tokens"] += details.get("cached_tokens") or 0
                    self.usage["audio_input_tokens"] += details.get("audio_tokens") or 0
                    if response.get("status") not in ("completed", None):
                        self.errors.append(f"response status {response.get('status')}: {json.dumps(response.get('status_details'))[:200]}")
                    self.rounds += 1
                    if round_calls and self.rounds <= self.cfg.max_tool_rounds:
                        await asyncio.gather(*pending)
                        for call in round_calls:
                            await conn.conversation.item.create(item=RealtimeConversationItemFunctionCallOutput(
                                call_id=call["call_id"], output=json.dumps(call["output"])))
                        pending, round_calls = [], []
                        await conn.response.create()
                    else:
                        self.response_ms = self._ms()
                        finished = True
                elif kind == "error":
                    err = ev.get("error") or {}
                    self.errors.append(f"{err.get('code')}: {err.get('message')}"[:300])
                    if err.get("type") in ("server_error",) or err.get("code") in ("rate_limit_exceeded",):
                        raise RuntimeError(f"service error {err.get('code')}")
            if pending:
                await asyncio.gather(*pending)
            if not finished:
                self.errors.append("timed out before the final response")

    def result_row(self, audio_seconds: float) -> dict[str, Any]:
        case = self.case
        tool_calls, predicted_entities, candidate_ids, retrieved = [], [], [], {}
        predicted_intent, predicted_choice = "", {}
        for idx, call in enumerate(self.calls):
            tool_calls.append({"type": "tool_call", "tool_call_id": call["call_id"] or f"call_{idx}",
                               "name": call["name"], "arguments": call["arguments"]})
            if not predicted_intent:
                predicted_intent = intent_from_tool_call(call["name"], call["arguments"]) or ""
            if call["name"] == "pick_choice" and not predicted_choice:
                predicted_choice = {"kind": str(call["arguments"].get("kind") or ""), "value": str(call["arguments"].get("value") or "")}
            results = (call.get("search") or {}).get("results") or []
            if call["name"] in SEARCH_TOOLS:
                if results:
                    top = results[0]
                    predicted_entities.append({"id": top["id"], "name": top["name"], "call_index": idx,
                                               "type": "user" if call["name"] == "search_user" else "group",
                                               "score": top["_match_score"], "confident": bool(top["_confident"])})
                for r in results:
                    candidate_ids.append(r["id"])
                    retrieved[r["id"]] = max(retrieved.get(r["id"], 0.0), r["_match_score"] / 100.0)
        gold_urns = sorted({u for e in case.get("expected_entities") or [] for u in e["urns"]})
        transcript = self.transcript or ""
        search_ms = sum(c.get("search", {}).get("latency_ms", 0) for c in self.calls if c.get("search"))
        return {
            **{k: case.get(k) for k in GOLD_KEYS},
            "run_label": self.cfg.run_label,
            "status": "ok" if not self.errors else "warning",
            "transcript": transcript,
            "query": transcript,
            "response": " ".join(self.texts),
            "predicted_intent": predicted_intent,
            "called_tools": sorted({c["name"] for c in self.calls}),
            "tool_calls": tool_calls,
            "tool_definitions": tool_definitions_for_evaluators(),
            "tool_results": [{"call_id": c["call_id"], "name": c["name"], "arguments": c["arguments"],
                              "t_ms": c["t_ms"], "tool_ms": c.get("tool_ms"), "output": c.get("output"),
                              "search": {k: v for k, v in (c.get("search") or {}).items() if k != "strategies"}}
                             for c in self.calls],
            "predicted_entities": predicted_entities,
            "candidate_ids": candidate_ids,
            "predicted_choice": predicted_choice,
            "retrieval_ground_truth": [{"document_id": u, "query_relevance_label": 1} for u in gold_urns],
            "retrieved_documents": [{"document_id": k, "relevance_score": round(v, 4)} for k, v in
                                    sorted(retrieved.items(), key=lambda kv: -kv[1])],
            "latency_ms": self.first_tool_ms if self.first_tool_ms is not None else -1,
            "latency_budget_ms": self.cfg.latency_budget_ms,
            "transcript_ms": self.transcript_ms,
            "response_ms": self.response_ms,
            "search_ms": round(search_ms, 1),
            "tool_rounds": self.rounds,
            "audio_seconds": round(audio_seconds, 2),
            "token_usage": self.usage,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "transcription_model": ((self.session_ack.get("audio") or {}).get("input") or {}).get("transcription", {}).get("model")
            if isinstance(((self.session_ack.get("audio") or {}).get("input") or {}).get("transcription"), dict) else None,
            "model": self.session_ack.get("model"),
            "errors": self.errors + ([self.transcript_error] if self.transcript_error else []),
        }


async def preflight(project, cfg: RunConfig, case: dict[str, Any]) -> dict[str, Any]:
    """Open one session with the run's overrides and verify the service acknowledged them.

    Some session fields are fixed by the agent version (for example the transcription model):
    the service accepts the update but keeps its own value, which would silently mislabel a run.
    """
    inputs = structured_inputs(case["flow"], case.get("choice_options"))
    async with project.beta.voice_agents.realtime.connect(agent_name=cfg.agent_name, structured_inputs=inputs) as conn:
        ack: dict[str, Any] = {}
        while True:
            ev = _event_dict(await asyncio.wait_for(conn.recv(), timeout=cfg.case_timeout_s))
            if ev.get("type") == "error":
                raise RuntimeError(f"session error: {json.dumps(ev.get('error'))[:300]}")
            if ev.get("type") == "session.updated":
                ack = ev.get("session") or {}
                break
        session: dict[str, Any] = {"type": "realtime"}
        if cfg.stt_model:
            session["audio"] = {"input": {"transcription": {"model": cfg.stt_model, "language": "en-US"}}}
        if cfg.text_only:
            session["output_modalities"] = ["text"]
        if len(session) > 1:
            await conn.session.update(session=session)
            while True:
                ev = _event_dict(await asyncio.wait_for(conn.recv(), timeout=cfg.case_timeout_s))
                if ev.get("type") == "error":
                    raise RuntimeError(f"session.update rejected: {json.dumps(ev.get('error'))[:300]}")
                if ev.get("type") == "session.updated":
                    ack = ev.get("session") or {}
                    break
    transcription = ((ack.get("audio") or {}).get("input") or {}).get("transcription") or {}
    problems = []
    if cfg.stt_model and transcription.get("model") != cfg.stt_model:
        problems.append(f"transcription model override to '{cfg.stt_model}' not applied (session uses "
                        f"'{transcription.get('model')}'); deploy a candidate agent with "
                        f"render_agent_yaml.py --candidate-transcription-model {cfg.stt_model} instead")
    if cfg.text_only and ack.get("output_modalities") != ["text"]:
        problems.append(f"text-only output not applied (session uses {ack.get('output_modalities')})")
    return {"model": ack.get("model"), "transcription_model": transcription.get("model"),
            "output_modalities": ack.get("output_modalities"), "problems": problems}


async def run_cases(project, cases: list[dict[str, Any]], cfg: RunConfig, search: SireSearchClient,
                    on_result=None) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.concurrency)
    results: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]

    async def one(i: int, case: dict[str, Any]) -> None:
        async with sem:
            audio_seconds = duration_seconds(load_pcm16(case["_audio_abs"]))
            last_error = ""
            for attempt in range(cfg.retries + 1):
                session = CaseSession(case, cfg, search)
                try:
                    await session.run(project)
                    row = session.result_row(audio_seconds)
                    break
                except Exception as exc:  # noqa: BLE001 - retry transient session failures
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < cfg.retries:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    row = session.result_row(audio_seconds)
                    row["status"] = "error"
                    row["errors"] = row["errors"] + [last_error, traceback.format_exc(limit=2)[-500:]]
            results[i] = row
            if on_result:
                on_result(i, row)

    await asyncio.gather(*(one(i, c) for i, c in enumerate(cases)))
    return results


def resolve_audio(cases: list[dict[str, Any]], manifest: Path) -> None:
    for case in cases:
        path = Path(case["audio_path"])
        case["_audio_abs"] = str(path if path.is_absolute() else (manifest.parent / path).resolve())
