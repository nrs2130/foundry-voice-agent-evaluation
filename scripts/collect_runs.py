"""Wait for submitted Foundry evaluation runs and store their results in the local run folders.

    python scripts/collect_runs.py latest:smoke
    python scripts/collect_runs.py <run> <run> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.runs import collect_run_dir, print_foundry_summary, resolve_run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folders, names, 'latest' or 'latest:<suite>'.")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-error-rate", type=float, default=None,
                        help="Resubmit a scorecard run once if more than this share of its code-grader results "
                             "errored (transient service errors; default 0.02, 1 disables).")
    args = parser.parse_args()
    for ref in args.runs:
        run_dir = resolve_run_dir(ref)
        print(f"== {run_dir.name}")
        print_foundry_summary(collect_run_dir(run_dir, timeout_s=args.timeout, max_error_rate=args.max_error_rate))


if __name__ == "__main__":
    main()
