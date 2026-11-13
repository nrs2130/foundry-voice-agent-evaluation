"""Print a Markdown leaderboard of local runs (one row per run) with key metrics and Foundry report links.

    python scripts/summarize_runs.py --suite smoke
    python scripts/summarize_runs.py --suite regression --markdown leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.runs import load_run  # noqa: E402
from sire_eval.scoring import aggregate  # noqa: E402
from sire_eval.settings import RESULTS_DIR  # noqa: E402

COLUMNS = [
    ("task_success", "Task success"), ("intent_accuracy", "Intent"), ("tool_selection", "Tool selection"),
    ("entity_top1", "Entity top-1"), ("entity_recall_at_5", "Entity @5"), ("no_confident_wrong", "No confident-wrong"),
    ("choice_accuracy", "Choice"), ("stt_word_accuracy", "STT word acc."), ("stt_entity_recall", "STT entity words"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    runs = sorted(p for p in RESULTS_DIR.glob("*") if (p / "results.jsonl").exists()
                  and (args.suite is None or f"-{args.suite}-" in p.name))
    header = ["Run", "LLM", "STT", "Search", "n"] + [title for _, title in COLUMNS] + ["Latency p50/p95 ms", "Foundry"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run_dir in runs:
        rows, info = load_run(run_dir)
        summary = aggregate(rows)
        cells = [info.get("label", run_dir.name), info.get("model", ""), info.get("stt_model", ""),
                 info.get("search_profile", ""), str(summary["cases"])]
        for metric, _ in COLUMNS:
            m = summary["metrics"].get(metric)
            cells.append(f"{m['pass_rate']:.1%}" if m else "-")
        lat = summary["latency_ms"]
        cells.append(f"{lat['p50']:.0f} / {lat['p95']:.0f}" if lat["p50"] is not None else "-")
        links = ""
        runs_file = run_dir / "foundry_runs.json"
        if runs_file.exists():
            foundry = json.loads(runs_file.read_text(encoding="utf-8")).get("runs") or {}
            links = " ".join(f"[{k}]({v['report_url']})" for k, v in foundry.items() if v.get("report_url"))
        cells.append(links or "-")
        lines.append("| " + " | ".join(cells) + " |")
    report = "\n".join(lines)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
