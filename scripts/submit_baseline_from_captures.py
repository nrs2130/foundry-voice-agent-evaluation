"""Turn the production (SCR) session captures into a scored baseline run - no audio replay needed.

    python scripts/submit_baseline_from_captures.py --suite smoke
    python scripts/submit_baseline_from_captures.py --suite regression --no-submit

Each corpus phrase has a capture of its most recent production-pipeline run (gpt-realtime-mini +
MAI-Transcribe + the production slot search). This script maps those captures onto the same result
schema the voice-agent runner produces, so the same Foundry scorecards grade production and every
candidate configuration side by side.

Production uses a single ``command`` tool with a cross-type entity search; for tool selection and
the tool-call LLM judge its calls are expressed in the demo vocabulary based on what its search
resolved for each slot (search_user / search_group / route_command / pick_choice).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.agent_spec import CHOICE_KIND_TO_INTENT, tool_definitions_for_evaluators  # noqa: E402
from sire_eval.dataset import read_jsonl  # noqa: E402
from sire_eval.runs import new_run_dir, print_local_summary, save_run, submit_run_dir  # noqa: E402
from sire_eval.settings import DATASETS_DIR, load_settings, long_path  # noqa: E402
from sire_eval.voice_runner import GOLD_KEYS  # noqa: E402

SELF_REFERENCES = {"me", "myself", "everyone", "self", "i"}


def capture_path(captures_root: Path, case: dict[str, Any]) -> Path:
    return captures_root / (case["audio_src"][: -len(".wav")] + ".capture.json")


def _event_ms(timings: dict[str, Any], name: str) -> float | None:
    for ev in timings.get("events") or []:
        if ev.get("name") == name and ev.get("t_ms") is not None:
            return float(ev["t_ms"])
    return None


def translate_tool_calls(captured: dict[str, Any], intent: str) -> list[dict[str, Any]]:
    """Express production's single ``command`` call in the demo tool vocabulary.

    Production fans each entity mention out across users and groups, so the call type is taken from
    what its slot search resolved (u-* -> search_user, g-* -> search_group), not from the argument list.
    """
    tool_call = captured.get("tool_call") or {}
    if not tool_call.get("name"):
        return []
    args = tool_call.get("arguments") or {}
    if tool_call["name"] == "pick_choice":
        return [{"name": "pick_choice", "arguments": {"kind": args.get("kind", ""), "value": args.get("value", "")}}]
    mentions = {s.get("slot_name"): s.get("slot_value") for s in (captured.get("slot_mapping") or {}).get("slots") or []}
    calls = []
    for slot in (captured.get("slot_search") or {}).get("slots") or []:
        if slot.get("slot_type") not in ("user", "group"):
            continue
        cands = slot.get("candidates") or []
        resolved = ("user" if cands[0]["entity_urn"].startswith("u-") else "group") if cands else slot["slot_type"]
        mention = mentions.get(slot.get("slot_name")) or ""
        mention = mention[0] if isinstance(mention, list) and mention else str(mention)
        if resolved == "user":
            calls.append({"name": "search_user", "arguments": {"intent": intent, "name": mention}})
        else:
            calls.append({"name": "search_group", "arguments": {"intent": intent, "group": mention}})
    if not calls:
        for name in args.get("user") or []:
            if str(name).strip().lower() not in SELF_REFERENCES:
                calls.append({"name": "search_user", "arguments": {"intent": intent, "name": name}})
        for group in args.get("group") or []:
            if str(group).strip().lower() not in SELF_REFERENCES:
                calls.append({"name": "search_group", "arguments": {"intent": intent, "group": group}})
    return calls or [{"name": "route_command", "arguments": {"intent": intent}}]


def capture_row(case: dict[str, Any], cap: dict[str, Any], label: str, budget_ms: int) -> dict[str, Any]:
    c = cap.get("captured") or {}
    intent = (c.get("intent") or "").strip().lower()
    calls = translate_tool_calls(c, intent)
    if not intent and calls and calls[0]["name"] == "pick_choice":
        intent = CHOICE_KIND_TO_INTENT.get(calls[0]["arguments"]["kind"], "")

    predicted_entities, candidate_ids, retrieved = [], [], {}
    final = (c.get("voice_server") or {}).get("slots") or {}
    for idx, slot in enumerate((c.get("slot_search") or {}).get("slots") or []):
        stype = slot.get("slot_type")
        if stype not in ("user", "group"):
            continue
        fanout = slot.get("fanout") or []
        confident = any(f.get("confident") for f in fanout if f.get("primary"))
        cands = slot.get("candidates") or []
        chosen = ((final.get(slot.get("slot_name")) or {}).get("slot_values") or [{}])[0].get("slot_value_id")
        top = chosen or (cands[0]["entity_urn"] if cands else None)
        if top:
            name = next((x.get("display_name", "") for x in cands if x.get("entity_urn") == top), "")
            score = next((x.get("score", 0) for x in cands if x.get("entity_urn") == top), 0)
            predicted_entities.append({"id": top, "name": name, "type": stype, "call_index": idx,
                                       "score": score, "confident": bool(confident)})
        for x in cands[:5]:
            candidate_ids.append(x["entity_urn"])
            retrieved[x["entity_urn"]] = max(retrieved.get(x["entity_urn"], 0.0), float(x.get("score") or 0) / 100.0)

    predicted_choice = {}
    if calls and calls[0]["name"] == "pick_choice":
        predicted_choice = {"kind": str(calls[0]["arguments"]["kind"] or ""), "value": str(calls[0]["arguments"]["value"] or "")}
    timings = c.get("server_timings") or {}
    eos, tool_t = _event_ms(timings, "end_of_speech"), _event_ms(timings, "tool_call")
    latency = round(tool_t - eos, 1) if eos is not None and tool_t is not None else -1
    names = ", ".join(p["name"] for p in predicted_entities if p["name"])
    gold_urns = sorted({u for e in case.get("expected_entities") or [] for u in e["urns"]})
    transcript = c.get("input_transcription") or c.get("asr_final_text") or ""
    return {
        **{k: case.get(k) for k in GOLD_KEYS},
        "run_label": label,
        "status": "ok" if not cap.get("errors") else "warning",
        "transcript": transcript,
        "query": transcript,
        "response": f"[production router] intent={intent or 'none'}" + (f"; resolved: {names}" if names else ""),
        "predicted_intent": intent,
        "called_tools": sorted({call["name"] for call in calls}),
        "tool_calls": [{"type": "tool_call", "tool_call_id": f"call_{i}", **call} for i, call in enumerate(calls)],
        "tool_definitions": tool_definitions_for_evaluators(),
        "tool_results": [],
        "production_tool_call": c.get("tool_call"),
        "predicted_entities": predicted_entities,
        "candidate_ids": candidate_ids,
        "predicted_choice": predicted_choice,
        "retrieval_ground_truth": [{"document_id": u, "query_relevance_label": 1} for u in gold_urns],
        "retrieved_documents": [{"document_id": k, "relevance_score": round(v, 4)} for k, v in sorted(retrieved.items(), key=lambda kv: -kv[1])],
        "latency_ms": latency,
        "latency_budget_ms": budget_ms,
        "search_ms": round(float(timings.get("search_ms") or 0), 1),
        "session_id": cap.get("session_id", ""),
        "errors": [str(e)[:200] for e in cap.get("errors") or []],
    }


def effective_config(cap: dict[str, Any]) -> dict[str, str]:
    for ev in cap.get("events") or []:
        nlu = ev.get("nlu_json") or {}
        cfg = nlu.get("effective_config") or (nlu.get("session_start") or {}).get("effective_config")
        if cfg:
            return cfg
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--label", default="production SCR capture")
    parser.add_argument("--latency-budget-ms", type=int, default=1500)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--no-llm-judges", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    cases = read_jsonl(DATASETS_DIR / f"sire-{args.suite}" / "cases.jsonl")
    rows, missing, config = [], 0, {}
    for case in cases:
        path = capture_path(settings.captures_root, case)
        try:
            cap = json.load(open(long_path(path), encoding="utf-8"))
        except FileNotFoundError:
            missing += 1
            continue
        config = config or effective_config(cap)
        rows.append(capture_row(case, cap, args.label, args.latency_budget_ms))
    print(f"{len(rows)} captures mapped ({missing} cases have no capture)")

    info = {"label": args.label, "suite": args.suite, "source": "scr-capture", "agent_name": "SIRE production (SCR)",
            "agent_version": "", "recognition_profile": config.get("recognition.profile", ""),
            "model": config.get("voice_live.model", "gpt-realtime-mini"),
            "stt_model": config.get("voice_live.transcription_model", "mai-transcribe-2"),
            "search_profile": f"production {config.get('slot_search.profile', '')}".strip(),
            "latency_budget_ms": args.latency_budget_ms, "missing_captures": missing}
    run_dir = new_run_dir(args.suite, args.label)
    save_run(run_dir, rows, info)
    print_local_summary(rows, f"\nLocal scores ({run_dir.name}):")
    if not args.no_submit:
        submit_run_dir(run_dir, llm_judges=not args.no_llm_judges)
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
