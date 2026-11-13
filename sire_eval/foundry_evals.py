"""Submit SIRE run results to Microsoft Foundry Evaluations (one eval group per scorecard)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .agent_spec import tool_definitions_for_evaluators
from .evaluators import EVAL_GROUPS, EvalGroup, graders_fingerprint, testing_criteria
from .settings import FOUNDRY_DIR

# Every row sent to Foundry carries every field with a non-null value: rows with missing keys
# are loaded as NaN by the service and fail the whole run.
ITEM_FIELDS: dict[str, str] = {
    "case_id": "string", "case_key": "string", "usage": "string", "lane": "string", "flow": "string",
    "facility": "string", "asr_label": "string", "run_label": "string",
    "gold_transcript": "string", "transcript": "string", "query": "string", "response": "string",
    "expected_intent": "string", "predicted_intent": "string",
    "expected_tools": "array", "called_tools": "array",
    "expected_entities": "array", "predicted_entities": "array", "candidate_ids": "array",
    "entity_mentions": "array",
    "expected_choice": "object", "predicted_choice": "object",
    "tool_calls": "array", "tool_definitions": "array",
    "retrieval_ground_truth": "array", "retrieved_documents": "array",
    "latency_ms": "number", "latency_budget_ms": "number",
}
_DEFAULTS = {"string": "", "array": [], "number": -1, "object": {"kind": "", "value": ""}}

# Code graders always return a score, so an errored result is a transient service-side failure
# (ProviderGradingError, typically a block of items at the start of a run under load).
# Runs above this rate are resubmitted once and the worse run is deleted.
MAX_CODE_GRADER_ERROR_RATE = 0.02


def item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": v} for k, v in ITEM_FIELDS.items()},
        "required": list(ITEM_FIELDS),
    }


def to_eval_item(row: dict[str, Any], tool_defs: list[dict] | None = None) -> dict[str, Any]:
    """Project a run-result row onto the eval item schema (consistent keys, no nulls)."""
    item: dict[str, Any] = {}
    for key, kind in ITEM_FIELDS.items():
        value = row.get(key)
        if value is None or (kind == "object" and not value):
            value = _DEFAULTS[kind] if kind != "array" else []
        if kind == "string":
            value = str(value)
        elif kind == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = -1.0
        item[key] = value
    item["query"] = item["query"] or item["transcript"] or "(no transcription)"
    item["response"] = item["response"] or "(no spoken response)"
    if not item["tool_definitions"]:
        item["tool_definitions"] = tool_defs or tool_definitions_for_evaluators()
    if not item["retrieval_ground_truth"]:
        item["retrieval_ground_truth"] = [{"document_id": "__none__", "query_relevance_label": 0}]
    if not item["retrieved_documents"]:
        item["retrieved_documents"] = [{"document_id": "__no_results__", "relevance_score": 0.0}]
    return item


def _metadata_path() -> Path:
    return FOUNDRY_DIR / "agent-metadata.yaml"


def load_metadata() -> dict[str, Any]:
    path = _metadata_path()
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"defaultEnvironment": "dev", "environments": {"dev": {}}}


def save_metadata(meta: dict[str, Any]) -> None:
    path = _metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _env(meta: dict[str, Any]) -> dict[str, Any]:
    env = meta.setdefault("defaultEnvironment", "dev")
    return meta.setdefault("environments", {}).setdefault(env, {})


def find_eval_by_name(client, name: str) -> str | None:
    after = None
    while True:
        page = client.evals.list(limit=100, after=after) if after else client.evals.list(limit=100)
        items = list(page.data) if hasattr(page, "data") else list(page)
        for ev in items:
            if getattr(ev, "name", None) == name:
                return ev.id
        if not getattr(page, "has_more", False) or not items:
            return None
        after = items[-1].id


def ensure_eval_group(client, group: EvalGroup, judge: str | None, llm_judges: bool, suffix: str = "",
                      versions: dict[str, str] | None = None) -> str:
    """Return the eval id for a scorecard, creating it on first use (cached in agent-metadata.yaml).

    Runs are only comparable within one scorecard, so scorecards without LLM judges get their own name,
    and a scorecard pins the evaluator versions it was created with.
    """
    versions = versions or {}
    fingerprint = graders_fingerprint(versions)
    meta = load_metadata()
    env = _env(meta)
    labels = ([] if llm_judges else ["no LLM judges"]) + ([suffix] if suffix else [])
    title = group.title + (f" ({', '.join(labels)})" if labels else "")
    cache_key = "-".join([group.key] + [slug for slug in ("" if llm_judges else "nollm", suffix) if slug])
    eval_id = (env.get("evaluationGroups") or {}).get(cache_key)
    existing = None
    if eval_id:
        try:
            existing = client.evals.retrieve(eval_id=eval_id)
        except Exception:  # noqa: BLE001 - deleted in the portal; look it up or recreate
            existing = None
    if existing is None:
        found = find_eval_by_name(client, title)
        existing = client.evals.retrieve(eval_id=found) if found else None
    if existing is not None:
        pinned = (getattr(existing, "metadata", None) or {}).get("graders")
        if pinned and pinned != fingerprint:
            raise SystemExit(
                f"Scorecard '{title}' was created with evaluator versions {pinned}; the catalog now resolves to "
                f"{fingerprint}. Runs scored by different grader code are not comparable: start a new scorecard "
                f"series with --scorecard-suffix <name> and resubmit the baseline into it.")
        eval_id = existing.id
    else:
        created = client.evals.create(
            name=title,
            data_source_config={"type": "custom", "item_schema": item_schema(), "include_sample_schema": False},
            testing_criteria=testing_criteria(group, judge, llm_judges, versions),
            metadata={"scorecard": group.key, "llm_judges": str(llm_judges).lower(), "graders": fingerprint},
        )
        eval_id = created.id
    env.setdefault("evaluationGroups", {})[cache_key] = eval_id
    env.setdefault("evaluatorVersions", {}).update(versions)
    save_metadata(meta)
    return eval_id


INLINE_LIMIT_BYTES = 2_000_000


def submit_run(client, eval_id: str, run_name: str, items: list[dict[str, Any]], metadata: dict[str, str]):
    """Create an eval run; large payloads are uploaded as a JSONL file (inline content is size-limited).

    Items are sent in a fixed pseudo-random order (hash of case_key): transient grading errors hit the
    first items of a run, and the prepared suites are sorted by lane, so in suite order a burst of errors
    would remove whole lanes from the pass rates.
    """
    ordered = sorted(items, key=lambda it: hashlib.sha1(str(it.get("case_key", "")).encode("utf-8")).hexdigest())
    payload = [{"item": it} for it in ordered]
    body = "\n".join(json.dumps(p, ensure_ascii=False) for p in payload)
    if len(body.encode("utf-8")) > INLINE_LIMIT_BYTES:
        uploaded = client.files.create(file=("eval_items.jsonl", body.encode("utf-8"), "application/jsonl"), purpose="evals")
        source: dict[str, Any] = {"type": "file_id", "id": uploaded.id}
    else:
        source = {"type": "file_content", "content": payload}
    return client.evals.runs.create(
        eval_id=eval_id,
        name=run_name[:256],
        metadata={k: str(v)[:512] for k, v in list(metadata.items())[:16]},
        data_source={"type": "jsonl", "source": source},
    )


def wait_for_runs(client, runs: list[tuple[str, str, str]], timeout_s: int = 3600, poll_s: int = 10) -> dict[str, Any]:
    """Poll (group_key, eval_id, run_id) triples until all reach a terminal state."""
    pending = {r[2]: r for r in runs}
    final: dict[str, Any] = {}
    started = time.time()
    while pending and time.time() - started < timeout_s:
        for run_id, (key, eval_id, _) in list(pending.items()):
            try:
                run = client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)
            except Exception:  # noqa: BLE001 - transient network/service error: poll again next round
                continue
            if run.status in ("completed", "failed", "canceled", "cancelled"):
                final[key] = run
                pending.pop(run_id)
        if pending:
            time.sleep(poll_s)
    return final


def run_summary(run) -> dict[str, Any]:
    counts = run.result_counts.model_dump() if getattr(run, "result_counts", None) else {}
    per = []
    for c in getattr(run, "per_testing_criteria_results", None) or []:
        d = c.model_dump()
        total = (d.get("passed") or 0) + (d.get("failed") or 0)
        per.append({"criterion": d.get("testing_criteria"), "passed": d.get("passed"), "failed": d.get("failed"),
                    "errored": d.get("errored"), "pass_rate": round(d["passed"] / total, 4) if total else None})
    return {"run_id": run.id, "eval_id": run.eval_id, "status": run.status, "counts": counts,
            "per_criterion": per, "report_url": getattr(run, "report_url", None),
            "error": run.error.model_dump() if getattr(run, "error", None) else None}


def code_grader_error_rate(summary: dict[str, Any], group: EvalGroup) -> float:
    """Worst errored fraction across the scorecard's custom code-grader criteria."""
    names = {name.removeprefix("sire-") for name in group.evaluators}
    worst = 0.0
    for c in summary.get("per_criterion") or []:
        if c["criterion"] not in names:
            continue
        errored = c.get("errored") or 0
        total = (c.get("passed") or 0) + (c.get("failed") or 0) + errored
        if total:
            worst = max(worst, errored / total)
    return worst


def wait_with_retries(client, submitted: list[tuple[str, str, str]], rows: list[dict[str, Any]], run_name: str,
                      metadata: dict[str, str], timeout_s: int = 3600,
                      max_error_rate: float = MAX_CODE_GRADER_ERROR_RATE
                      ) -> tuple[list[tuple[str, str, str]], dict[str, Any], list[dict[str, Any]]]:
    """Wait for runs; resubmit (once) any run whose code graders hit transient errors and keep the better run.

    Returns the final (group, eval_id, run_id) triples, the terminal runs by group, and a log of retries.
    """
    final = wait_for_runs(client, submitted, timeout_s=timeout_s)
    triples = list(submitted)
    retries: list[dict[str, Any]] = []
    for i, (key, eval_id, run_id) in enumerate(triples):
        run = final.get(key)
        if run is None or run.status != "completed":
            continue
        rate = code_grader_error_rate(run_summary(run), EVAL_GROUPS[key])
        if rate <= max_error_rate:
            continue
        items = [to_eval_item(r) for r in rows if EVAL_GROUPS[key].applies(r)]
        new = submit_run(client, eval_id, run_name, items, {**metadata, "scorecard": key, "retry_of": run_id})
        print(f"  [{key}] {rate:.0%} of code-grader results errored (transient service error); resubmitted as {new.id}")
        retries.append({"group": key, "eval_id": eval_id, "original_run_id": run_id, "retry_run_id": new.id,
                        "original_error_rate": round(rate, 4), "index": i})
    if retries:
        retried = wait_for_runs(client, [(r["group"], r["eval_id"], r["retry_run_id"]) for r in retries],
                                timeout_s=timeout_s)
        for r in retries:
            idx = r.pop("index")
            new_run = retried.get(r["group"])
            new_rate = (code_grader_error_rate(run_summary(new_run), EVAL_GROUPS[r["group"]])
                        if new_run is not None and new_run.status == "completed" else None)
            r["retry_error_rate"] = None if new_rate is None else round(new_rate, 4)
            if new_rate is not None and new_rate < r["original_error_rate"]:
                keep, drop = r["retry_run_id"], r["original_run_id"]
                final[r["group"]] = new_run
                triples[idx] = (r["group"], r["eval_id"], keep)
            elif new_run is not None:
                keep, drop = r["original_run_id"], r["retry_run_id"]
            else:
                r["kept"] = "both (retry did not finish)"
                continue
            client.evals.runs.delete(eval_id=r["eval_id"], run_id=drop)
            r["kept"], r["deleted"] = keep, drop
            after = f"{r['retry_error_rate']:.0%}" if r["retry_error_rate"] is not None else "n/a"
            print(f"  [{r['group']}] kept {keep}, deleted {drop} "
                  f"(code-grader errors {r['original_error_rate']:.0%} -> {after})")
    return triples, final, retries


def delete_runs(client, entries: list[dict[str, Any]], timeout_s: int = 120) -> None:
    """Delete eval runs ({group, eval_id, run_id}); active runs are cancelled first."""
    for s in entries:
        try:
            run = client.evals.runs.retrieve(eval_id=s["eval_id"], run_id=s["run_id"])
            if run.status in ("queued", "in_progress"):
                client.evals.runs.cancel(eval_id=s["eval_id"], run_id=s["run_id"])
                started = time.time()
                while run.status in ("queued", "in_progress") and time.time() - started < timeout_s:
                    time.sleep(5)
                    run = client.evals.runs.retrieve(eval_id=s["eval_id"], run_id=s["run_id"])
            client.evals.runs.delete(eval_id=s["eval_id"], run_id=s["run_id"])
            print(f"  deleted previous [{s['group']}] run {s['run_id']}")
        except Exception as exc:  # noqa: BLE001 - report and continue with the remaining runs
            print(f"  could not delete previous [{s['group']}] run {s['run_id']}: {exc}")


def download_output_items(client, eval_id: str, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for item in client.evals.runs.output_items.list(eval_id=eval_id, run_id=run_id):
        rows.append({
            "case_id": (item.datasource_item or {}).get("case_id"),
            "status": item.status,
            "results": [
                {k: v for k, v in r.model_dump().items() if k in ("name", "metric", "score", "passed", "label", "reason", "threshold", "properties")}
                for r in (item.results or [])
            ],
        })
    return rows


def submit_results(project, rows: list[dict[str, Any]], run_name: str, run_metadata: dict[str, str],
                   judge: str | None, llm_judges: bool, groups: list[str] | None = None,
                   scorecard_suffix: str = "", versions: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Submit result rows to each applicable scorecard; returns the new {group, eval_id, run_id} entries."""
    client = project.get_openai_client()
    submitted = []
    for key in groups or list(EVAL_GROUPS):
        group = EVAL_GROUPS[key]
        items = [to_eval_item(r) for r in rows if group.applies(r)]
        if not items:
            continue
        eval_id = ensure_eval_group(client, group, judge, llm_judges, suffix=scorecard_suffix, versions=versions)
        run = submit_run(client, eval_id, run_name, items, {**run_metadata, "scorecard": key})
        print(f"  submitted {len(items):4d} rows to '{group.title}' eval={eval_id} run={run.id}")
        submitted.append({"group": key, "eval_id": eval_id, "run_id": run.id})
    return submitted
