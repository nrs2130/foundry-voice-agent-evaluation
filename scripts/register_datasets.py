"""Register prepared suites as versioned Foundry datasets and record them in agent-metadata.yaml.

    python scripts/register_datasets.py --suites smoke regression

Each suite folder (voicelive_harness.jsonl + cases.jsonl + audio/*.wav + summary.json) becomes a
folder dataset "sire-voice-<suite>" whose version is a hash of the gold manifest and the harness file
(which embeds the agent prompt and tool definitions), so re-running is a no-op until the data or the
agent definition changes. The full suite references the extracted corpus in place and is not uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.evaluators import EVAL_GROUPS, EVALUATORS  # noqa: E402
from sire_eval.foundry_evals import load_metadata, save_metadata  # noqa: E402
from sire_eval.settings import DATASETS_DIR, REPO_ROOT, project_client  # noqa: E402

TIERS = {"smoke": "smoke", "regression": "regression", "full": "coverage"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="+", default=["smoke", "regression"], choices=["smoke", "regression"])
    args = parser.parse_args()

    project = project_client()
    meta = load_metadata()
    env = meta.setdefault("environments", {}).setdefault(meta.get("defaultEnvironment", "dev"), {})
    env.setdefault("azd", {"environmentName": "sire-voice-agent-dev", "service": "sire-voice"})
    suites = {s["id"]: s for s in env.get("evaluationSuites") or []}
    for suite in args.suites:
        folder = DATASETS_DIR / f"sire-{suite}"
        manifest = folder / "cases.jsonl"
        if not manifest.exists():
            raise SystemExit(f"{manifest} not found; run scripts/prepare_datasets.py first.")
        name = f"sire-voice-{suite}"
        digest = hashlib.sha1(manifest.read_bytes())
        harness = folder / "voicelive_harness.jsonl"
        if harness.exists():
            digest.update(harness.read_bytes())
        version = digest.hexdigest()[:10]
        try:
            dataset = project.datasets.get(name=name, version=version)
            print(f"{name}:{version} already registered")
        except Exception:  # noqa: BLE001 - not registered yet
            print(f"Uploading {folder} as {name}:{version} ...")
            dataset = project.datasets.upload_folder(name=name, version=version, folder=str(folder))
            print(f"  registered {dataset.id}")
        suites[f"sire-{suite}"] = {
            "id": f"sire-{suite}",
            "tags": {"tier": TIERS[suite], "purpose": "regression", "stage": "curated"},
            "dataset": name,
            "datasetVersion": version,
            "datasetUri": getattr(dataset, "data_uri", None) or dataset.id,
            "datasetFile": str(manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
            "harnessFile": str((folder / "voicelive_harness.jsonl").relative_to(REPO_ROOT)).replace("\\", "/"),
            "scorecards": {key: group.title for key, group in EVAL_GROUPS.items()},
            "evaluators": [{"name": e.name, "threshold": e.threshold} for e in EVALUATORS]
            + [{"name": "builtin.document_retrieval"}, {"name": "builtin.intent_resolution"}, {"name": "builtin.tool_call_accuracy"}],
        }
    env["evaluationSuites"] = list(suites.values())
    # Uploads take minutes; merge into a fresh copy so concurrent runs' cached scorecard IDs are kept.
    fresh = load_metadata()
    fresh_env = fresh.setdefault("environments", {}).setdefault(fresh.get("defaultEnvironment", "dev"), {})
    fresh_env.setdefault("azd", env["azd"])
    fresh_env["evaluationSuites"] = env["evaluationSuites"]
    save_metadata(fresh)
    print("Updated agent/.foundry/agent-metadata.yaml")


if __name__ == "__main__":
    main()
