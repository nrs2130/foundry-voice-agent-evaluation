"""Register (or update) the SIRE custom evaluators in the Foundry evaluator catalog.

    python scripts/register_evaluators.py          # new version only when grader code changed
    python scripts/register_evaluators.py --force  # always create a new version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.evaluators import BY_NAME, register_evaluators  # noqa: E402
from sire_eval.settings import project_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    versions = register_evaluators(project_client(), force=args.force)
    for name, version in versions.items():
        ev = BY_NAME[name]
        print(f"{name:28s} v{version:>3s}  threshold={ev.threshold:<4}  {ev.display_name}")


if __name__ == "__main__":
    main()
