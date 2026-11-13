"""Run folders under agent/.foundry/results/: save, load, summarize and submit to Foundry."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .dataset import read_jsonl, write_jsonl
from .scoring import aggregate, breakdown, format_summary
from .settings import RESULTS_DIR, load_settings, project_client


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "run"


def new_run_dir(suite: str, label: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}-{suite}-{slug(label)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(run_dir: Path, rows: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    write_jsonl(run_dir / "results.jsonl", rows)
    info = {**info, "summary": aggregate(rows), "by_lane": breakdown(rows, "lane")}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def resolve_run_dir(ref: str) -> Path:
    """Accept a path, a run folder name, 'latest', or 'latest:<suite>'."""
    path = Path(ref)
    if path.exists():
        return path
    if (RESULTS_DIR / ref).exists():
        return RESULTS_DIR / ref
    if ref.startswith("latest"):
        suite = ref.split(":", 1)[1] if ":" in ref else None
        runs = sorted(p for p in RESULTS_DIR.glob("*") if (p / "results.jsonl").exists()
                      and (suite is None or f"-{suite}-" in p.name))
        if runs:
            return runs[-1]
    raise SystemExit(f"Run not found: {ref}")


def load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(run_dir / "results.jsonl")
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    return rows, info


def run_name(info: dict[str, Any]) -> str:
    parts = [info.get("label", "run"), info.get("suite", "")]
    if info.get("agent_name"):
        version = info.get("agent_version")
        parts.append(f"{info['agent_name']}:v{version}" if version else info["agent_name"])
    if info.get("model"):
        parts.append(f"llm={info['model']}")
    if info.get("stt_model"):
        parts.append(f"stt={info['stt_model']}")
    if info.get("search_profile"):
        parts.append(f"search={info['search_profile']}")
    return " | ".join(p for p in parts if p)


def run_metadata(run_dir: Path, info: dict[str, Any]) -> dict[str, str]:
    metadata = {k: info.get(k, "") for k in ("label", "suite", "agent_name", "agent_version", "model", "stt_model",
                                            "search_profile", "source")}
    metadata["local_run"] = run_dir.name
    return metadata


def submit_run_dir(run_dir: Path, llm_judges: bool = True, groups: list[str] | None = None,
                   wait: bool = True, scorecard_suffix: str = "",
                   max_error_rate: float | None = None, replace: bool = False) -> dict[str, Any]:
    """Submit a run folder to the scorecards.

    The new Foundry run ids are recorded in foundry_runs.json before waiting, so an interrupted wait can be resumed
    with collect_runs.py. Groups submitted before and not resubmitted now stay recorded. With ``replace`` the
    previous Foundry runs of the resubmitted groups are deleted once their replacements complete, so each
    scorecard keeps one run per folder.
    """
    from .evaluators import register_evaluators
    from .foundry_evals import load_metadata, save_metadata, submit_results

    settings = load_settings()
    rows, info = load_run(run_dir)
    project = project_client(settings)
    versions = register_evaluators(project)
    print(f"Evaluator catalog: {', '.join(f'{k}@{v}' for k, v in versions.items())}")
    name = run_name(info)
    runs_file = run_dir / "foundry_runs.json"
    previous = json.loads(runs_file.read_text(encoding="utf-8")) if runs_file.exists() else {}
    print(f"Submitting '{name}' ({len(rows)} rows) to Foundry evaluations ...")
    new = submit_results(project, rows, name, run_metadata(run_dir, info), settings.judge_deployment, llm_judges,
                         groups=groups, scorecard_suffix=scorecard_suffix, versions=versions)
    new_groups = {s["group"] for s in new}
    earlier = previous.get("submitted") or []
    replaced = [s for s in earlier if s["group"] in new_groups]
    if replaced and not replace:
        print(f"  note: earlier runs for {sorted({s['group'] for s in replaced})} remain in the scorecards "
              "(use --replace to delete them)")
    summary: dict[str, Any] = {
        "submitted": [s for s in earlier if s["group"] not in new_groups] + new,
        "runs": {k: v for k, v in (previous.get("runs") or {}).items() if k not in new_groups},
    }
    kept_retries = [r for r in previous.get("retries") or [] if r["group"] not in new_groups]
    if kept_retries:
        summary["retries"] = kept_retries
    replacing = (previous.get("replacing") or []) + (replaced if replace else [])
    if replacing:
        summary["replacing"] = replacing
    runs_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if wait and new:
        print(f"  waiting for grading; if interrupted, resume with: python scripts/collect_runs.py {run_dir.name}")
        summary = collect_run_dir(run_dir, max_error_rate=max_error_rate, groups=sorted(new_groups))
    print_foundry_summary(summary)
    meta = load_metadata()
    env = meta.setdefault("environments", {}).setdefault(meta.get("defaultEnvironment", "dev"), {})
    env["lastEval"] = {"run": run_dir.name, "name": name, "submitted": summary.get("submitted"),
                       "resultsFile": str(run_dir / "foundry_runs.json")}
    save_metadata(meta)
    return summary


def collect_run_dir(run_dir: Path, timeout_s: int = 3600, max_error_rate: float | None = None,
                    groups: list[str] | None = None) -> dict[str, Any]:
    """Wait for a run's submitted Foundry runs (see foundry_runs.json) and store their results locally.

    Runs whose code graders hit transient service errors are resubmitted once (see wait_with_retries). Earlier
    runs queued for replacement (submit_run.py --replace) are deleted once their replacement has completed.
    """
    from .foundry_evals import (MAX_CODE_GRADER_ERROR_RATE, delete_runs, download_output_items, run_summary,
                                wait_with_retries)

    runs_file = run_dir / "foundry_runs.json"
    if not runs_file.exists():
        raise SystemExit(f"{run_dir.name} has not been submitted (no foundry_runs.json).")
    summary = json.loads(runs_file.read_text(encoding="utf-8"))
    entries = summary.get("submitted") or []
    waiting = [(s["group"], s["eval_id"], s["run_id"]) for s in entries if groups is None or s["group"] in groups]
    rows, info = load_run(run_dir)
    client = project_client().get_openai_client()
    triples, final, retries = wait_with_retries(
        client, waiting, rows, run_name(info), run_metadata(run_dir, info), timeout_s=timeout_s,
        max_error_rate=MAX_CODE_GRADER_ERROR_RATE if max_error_rate is None else max_error_rate)
    current = {k: {"group": k, "eval_id": e, "run_id": r} for k, e, r in triples}
    summary["submitted"] = [current.get(s["group"], s) for s in entries]
    if retries:
        summary["retries"] = (summary.get("retries") or []) + retries
    summary["runs"] = {**(summary.get("runs") or {}), **{k: run_summary(run) for k, run in final.items()}}
    for key, eval_id, run_id in triples:
        if key in final and final[key].status == "completed":
            items = download_output_items(client, eval_id, run_id)
            (run_dir / f"foundry_output_{key}.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    replacing = summary.pop("replacing", None) or []
    completed = {k for k, run in final.items() if run.status == "completed"}
    delete_runs(client, [s for s in replacing if s["group"] in completed])
    still_pending = [s for s in replacing if s["group"] not in completed]
    if still_pending:
        summary["replacing"] = still_pending
    runs_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def print_foundry_summary(summary: dict[str, Any]) -> None:
    for r in summary.get("retries") or []:
        print(f"[{r['group']}] retried after transient grader errors: kept {r.get('kept')}")
    for key, run in (summary.get("runs") or {}).items():
        print(f"\n[{key}] status={run['status']} counts={run['counts']}")
        for c in run["per_criterion"]:
            rate = f"{c['pass_rate']:.1%}" if c["pass_rate"] is not None else "n/a"
            print(f"    {c['criterion']:28s} pass={rate:>7s}  (passed {c['passed']}, failed {c['failed']}, errored {c['errored']})")
        if run.get("report_url"):
            print(f"    report: {run['report_url']}")
        if run.get("error"):
            print(f"    error: {run['error']}")


def print_local_summary(rows: list[dict[str, Any]], title: str) -> None:
    print(format_summary(aggregate(rows), title))
