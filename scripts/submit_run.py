"""Submit an existing local run folder to the Foundry evaluation scorecards.

    python scripts/submit_run.py latest:smoke
    python scripts/submit_run.py agent/.foundry/results/<run> --groups entities
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.evaluators import EVAL_GROUPS  # noqa: E402
from sire_eval.runs import resolve_run_dir, submit_run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Run folder, its name, 'latest' or 'latest:<suite>'.")
    parser.add_argument("--groups", nargs="*", choices=list(EVAL_GROUPS), default=None)
    parser.add_argument("--no-llm-judges", action="store_true",
                        help="Skip the LLM-judge evaluators (uses separate 'no LLM judges' scorecards).")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--scorecard-suffix", default="", help="Submit to separately named scorecards (e.g. 'test').")
    parser.add_argument("--max-error-rate", type=float, default=None,
                        help="When waiting, resubmit a scorecard run once if more than this share of its code-grader "
                             "results errored (transient service errors; default 0.02, 1 disables).")
    parser.add_argument("--replace", action="store_true",
                        help="Delete this run folder's earlier Foundry runs for the resubmitted scorecards.")
    args = parser.parse_args()
    submit_run_dir(resolve_run_dir(args.run), llm_judges=not args.no_llm_judges, groups=args.groups,
                   wait=not args.no_wait, scorecard_suffix=args.scorecard_suffix, max_error_rate=args.max_error_rate,
                   replace=args.replace)


if __name__ == "__main__":
    main()
