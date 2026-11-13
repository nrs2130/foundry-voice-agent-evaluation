"""Gold case model built from the regression corpus (phrases + entity lists + per-phrase metadata)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .agent_spec import render_instructions, structured_inputs, tool_definitions
from .settings import long_path

FLOW_BY_PREFIX = {
    "main_menu": "MAIN_MENU",
    "login": "LOGIN",
    "partialcall": "PARTIAL_CALL",
    "yesno": "YES_NO",
    "choose": "CHOOSE",
}
USAGE_CODES = {
    "01-production": "prod",
    "02-user-lanes": "user",
    "03-group-lanes": "group",
    "04-intent-acceptance": "accept",
    "05-constrained-choice": "choice",
}
ENTITY_PREFIXES = {"u-": "user", "g-": "group"}
CLEAN_LABELS = {"clean", "clean-low-evidence", "not-measured"}


def flow_for(case_id: str) -> str:
    prefix = case_id.split("/", 1)[0]
    for key, flow in FLOW_BY_PREFIX.items():
        if prefix == key or prefix.startswith(key + "_"):
            return flow
    return "MAIN_MENU"


def case_key(usage: str, lane: str, case_id: str) -> str:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{USAGE_CODES.get(usage, usage)}-{lane}-{digest}"


def stable_order(case_id: str) -> str:
    return hashlib.sha1(("sire-eval/" + case_id).encode("utf-8")).hexdigest()


def expected_entities(slots: dict[str, list[str]], names: dict[str, str]) -> list[dict[str, Any]]:
    """One entry per slot that references directory entities; any listed URN is an acceptable answer."""
    out = []
    for slot, values in slots.items():
        urns = [v for v in values if v[:2] in ENTITY_PREFIXES]
        if not urns:
            continue
        out.append({
            "slot": slot,
            "type": ENTITY_PREFIXES[urns[0][:2]],
            "urns": urns,
            "names": [names.get(u, "") for u in urns],
        })
    return out


def expected_answer(case: dict[str, Any]) -> str:
    """Ground-truth answer text for LLM-judge evaluators and the Voice Live evaluation harness."""
    if case["flow"] == "CHOOSE":
        choice = case.get("expected_choice") or {}
        kind = choice.get("kind")
        if kind == "choice":
            return f"Selects the option '{choice.get('value')}' (pick_choice kind=choice)."
        if kind == "exit":
            return "The user wants to leave the prompt (pick_choice kind=exit)."
        return "The reply matches none of the options (pick_choice kind=none)."
    parts = [f"Intent: {case['expected_intent']}."]
    ents = case.get("expected_entities") or []
    if ents:
        rendered = []
        for e in ents:
            label = " or ".join(f"{n} ({u})" for n, u in zip(e["names"], e["urns"]))
            rendered.append(f"{e['type']} {label}")
        parts.append("Resolved entities: " + "; ".join(rendered) + ".")
        parts.append("The agent searches the directory for each entity and confirms the resolved names.")
    else:
        parts.append("No person or group is referenced; the agent routes the command without a directory search.")
    return " ".join(parts)


def build_case(row: dict[str, Any], meta: dict[str, Any], names: dict[str, str],
               caller_departments: dict[str, list[str]] | None = None) -> dict[str, Any]:
    usage, lane = row["usage"], row["lane"]
    flow = "CHOOSE" if usage == "05-constrained-choice" else flow_for(row["case_id"])
    ents = expected_entities(row.get("slots") or {}, names)
    ner = meta.get("nerEntities") or {}
    mentions = [v for k, v in ner.items() if k.startswith(("user", "group")) and v]
    caller = meta.get("caller") or {}
    caller_urn = caller.get("urn") or ""

    choice_options: list[str] = []
    expected_choice: dict[str, str] = {}
    if flow == "CHOOSE":
        choice_list = row.get("choice_list") or meta.get("choice_list") or {}
        choice_options = list(choice_list.get("values") or [])
        if row.get("expected_tool_call"):
            args = row["expected_tool_call"].get("arguments") or {}
            expected_choice = {"kind": args.get("kind", "none"), "value": args.get("value", "")}
        elif row["intent"] == "choose":
            values = (row.get("slots") or {}).get("CHOICE") or [""]
            expected_choice = {"kind": "choice", "value": values[0]}
        elif row["intent"] == "goodbye":
            expected_choice = {"kind": "exit", "value": ""}
        else:
            expected_choice = {"kind": "none", "value": ""}

    if flow == "CHOOSE":
        tools = ["pick_choice"]
    else:
        tools = sorted({"search_user" if e["type"] == "user" else "search_group" for e in ents})
        if not tools:
            tools = ["route_command"]

    review = row.get("asr_review") or {}
    heard = review.get("heard_as") or []
    return {
        "case_id": row["case_id"],
        "case_key": case_key(usage, lane, row["case_id"]),
        "usage": usage,
        "lane": lane,
        "flow": flow,
        "facility": row.get("facility") or "",
        "facility_ids": [f for f in [row.get("facility"), "s-global"] if f],
        "caller_urn": caller_urn,
        "caller_relation": caller.get("relation") or "",
        "caller_department_ids": (caller_departments or {}).get(caller_urn, []) if caller_urn else [],
        "audio_src": f"{usage}/{row['audio']}" if row.get("audio") and not row.get("text_only") else "",
        "gold_transcript": row["phrase"],
        "spoken_text": row.get("spoken") or row["phrase"],
        "expected_intent": row["intent"],
        "expected_entities": ents,
        "expected_tools": tools,
        "expected_choice": expected_choice,
        "choice_options": choice_options,
        "entity_mentions": mentions,
        "slots": row.get("slots") or {},
        "asr_label": row.get("asr_label") or "not-measured",
        "asr_mismatch_rate": review.get("mismatch_rate"),
        "asr_heard_as": heard[0]["text"] if heard else "",
        "asr_mismatch_span": review.get("mismatch_span") or "",
        "difficulty": row.get("difficulty") or "",
        "voice": row.get("voice") or (meta.get("phrase_generator") or {}).get("voice") or "",
        "accent": row.get("accent") or "",
        "form": row.get("form") or "",
        "audio_condition": row.get("audio_condition") or "",
        "tags": row.get("tags") or [],
    }


def harness_row(case: dict[str, Any], wav_path: str, instructions: str | None = None) -> dict[str, Any]:
    """Row in the Voice Live evaluation harness format (WavPath/Question/Answer/system_prompt/tool_definitions)."""
    instructions = instructions or render_instructions()
    inputs = structured_inputs(case["flow"], case.get("choice_options"))
    system_prompt = instructions.replace("{{flow}}", inputs["flow"]).replace("{{flow_guidance}}", inputs["flow_guidance"])
    return {
        "WavPath": wav_path,
        "Question": case["gold_transcript"],
        "Answer": expected_answer(case),
        "conversationID": case["case_key"],
        "system_prompt": system_prompt,
        "tool_definitions": tool_definitions(),
        "case_id": case["case_id"],
        "usage": case["usage"],
        "lane": case["lane"],
        "flow": case["flow"],
        "expected_intent": case["expected_intent"],
        "expected_entities": case["expected_entities"],
        "expected_tools": case["expected_tools"],
        "expected_choice": case["expected_choice"],
        "asr_label": case["asr_label"],
    }


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(long_path(path), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
