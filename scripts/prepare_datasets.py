"""Prepare SIRE evaluation datasets from the tenant regression corpus.

    python scripts/prepare_datasets.py                     # smoke + regression (+ full manifest)
    python scripts/prepare_datasets.py --suites smoke

For every suite this writes to agent/.foundry/datasets/sire-<suite>/:
    cases.jsonl              gold manifest consumed by scripts/run_voice_eval.py
    voicelive_harness.jsonl  the same cases in the Voice Live evaluation-harness format
                             (WavPath, Question, Answer, system_prompt, tool_definitions, conversationID)
    audio/<case_key>.wav     short-path WAV copies (smoke and regression only)
    summary.json             counts per usage / lane / flow / ASR label

Selection is deterministic (hash-ordered) and stratified by usage and lane, with roughly a
third of each stratum drawn from phrases the ASR watchlist flags as misheard.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.dataset import CLEAN_LABELS, build_case, harness_row, read_jsonl, stable_order, write_jsonl  # noqa: E402
from sire_eval.search_schema import load_memberships, read_jsonl_gz  # noqa: E402
from sire_eval.settings import DATASETS_DIR, load_settings, long_path  # noqa: E402

# Departments larger than this ("Everyone", "All Staff") carry no disambiguating signal.
MAX_CONTEXT_GROUP_SIZE = 300

SUITES = {
    "smoke": {
        "01-production": 30,
        "02-user-lanes": {"name": 5, "dept": 5, "doctor": 5, "single": 5, "spelled": 5, "name_shape": 5},
        "03-group-lanes": {"name": 5, "spoken": 5, "spelled": 5, "member_singular": 5, "member_plural": 5, "joiner": 5},
        "04-intent-acceptance": {"main_menu": 6, "login": 3, "partialcall": 3, "choose": 2, "yesno": 1},
        "05-constrained-choice": {"clean": 3, "snr5": 2, "snr10": 1, "snr20": 1, "tel": 2, "speed090": 1,
                                  "speed110": 1, "clip150": 1, "voice1": 1, "drift": 2},
    },
    "regression": {
        "01-production": 300,
        "02-user-lanes": {"name": 50, "dept": 50, "doctor": 50, "single": 50, "spelled": 50, "name_shape": 50},
        "03-group-lanes": {"name": 50, "spoken": 50, "spelled": 50, "member_singular": 50, "member_plural": 50, "joiner": 50},
        "04-intent-acceptance": {"main_menu": 90, "login": 20, "partialcall": 20, "choose": 15, "yesno": 5},
        "05-constrained-choice": {"clean": 20, "snr5": 12, "snr10": 10, "snr20": 10, "tel": 12, "speed090": 10,
                                  "speed095": 8, "speed105": 8, "speed110": 10, "clip150": 10, "voice1": 10,
                                  "voice2": 10, "voice3": 10, "drift": 10},
    },
    "full": None,
}


def pick(cases: list[dict], n: int, hard_share: float = 1 / 3) -> list[dict]:
    """Deterministically pick n cases, ~hard_share of them from ASR-flagged phrases."""
    ordered = sorted(cases, key=lambda c: stable_order(c["case_id"]))
    hard = [c for c in ordered if c["asr_label"] not in CLEAN_LABELS]
    clean = [c for c in ordered if c["asr_label"] in CLEAN_LABELS]
    n_hard = min(len(hard), round(n * hard_share))
    chosen = hard[:n_hard] + clean[: n - n_hard]
    if len(chosen) < n:
        rest = [c for c in ordered if c not in chosen]
        chosen += rest[: n - len(chosen)]
    return chosen


def pick_production(cases: list[dict], total: int) -> list[dict]:
    """Traffic-weighted intent mix: one slot for each of the top intents, the rest proportional to volume."""
    by_intent = collections.defaultdict(list)
    for c in cases:
        by_intent[c["expected_intent"]].append(c)
    ranked = sorted(by_intent, key=lambda i: (-len(by_intent[i]), i))
    quota = collections.Counter({i: 1 for i in ranked[: max(1, total // 2)]})
    remaining = total - sum(quota.values())
    volume = sum(len(v) for v in by_intent.values())
    shares = {i: remaining * len(by_intent[i]) / volume for i in ranked}
    for i in ranked:
        quota[i] += math.floor(shares[i])
    for i in sorted(ranked, key=lambda i: shares[i] - math.floor(shares[i]), reverse=True)[: total - sum(quota.values())]:
        quota[i] += 1
    chosen = []
    for intent, n in quota.items():
        chosen += pick(by_intent[intent], min(n, len(by_intent[intent])))
    return chosen


def select(cases: list[dict], spec: dict | None) -> list[dict]:
    if spec is None:
        return [c for c in cases if c["audio_src"]]
    chosen = []
    for usage, quota in spec.items():
        pool = [c for c in cases if c["usage"] == usage and c["audio_src"]]
        if isinstance(quota, int):
            chosen += pick_production(pool, quota)
            continue
        for lane, n in quota.items():
            lane_pool = [c for c in pool if c["lane"] == lane]
            chosen += pick(lane_pool, min(n, len(lane_pool)))
    return chosen


def summarize(cases: list[dict]) -> dict:
    def count(key):
        return dict(sorted(collections.Counter(key(c) for c in cases).items()))

    return {
        "cases": len(cases),
        "by_usage": count(lambda c: c["usage"]),
        "by_usage_lane": count(lambda c: f"{c['usage']}/{c['lane']}"),
        "by_flow": count(lambda c: c["flow"]),
        "by_expected_tools": count(lambda c: "+".join(c["expected_tools"])),
        "by_asr_label": count(lambda c: c["asr_label"]),
        "with_entities": sum(1 for c in cases if c["expected_entities"]),
        "entity_slots": sum(len(c["expected_entities"]) for c in cases),
        "intents": len({c["expected_intent"] for c in cases}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    args = parser.parse_args()

    settings = load_settings()
    corpus = settings.corpus_root
    if not (corpus / "phrases-all.jsonl").exists():
        raise SystemExit(f"Corpus not found at {corpus}. Extract the test_data zips first (see README).")

    print("Loading entity names and memberships ...")
    names: dict[str, str] = {}
    for fname in ("users.jsonl.gz", "groups.jsonl.gz"):
        for rec in read_jsonl_gz(corpus / "entity-lists" / fname):
            names[rec["entity_urn"]] = rec.get("display_name") or ""
    _, memberships = load_memberships(corpus / "entity-lists")
    sizes = collections.Counter(g for groups in memberships.values() for g in groups)
    caller_departments = {u: sorted(g for g in groups if sizes[g] <= MAX_CONTEXT_GROUP_SIZE) for u, groups in memberships.items()}

    print("Building gold cases ...")
    rows = read_jsonl(corpus / "phrases-all.jsonl")
    cases = []
    for row in rows:
        meta_path = corpus / row["usage"] / row["metadata"]
        meta = json.load(open(long_path(meta_path), encoding="utf-8")) if row.get("metadata") else {}
        cases.append(build_case(row, meta, names, caller_departments))
    print(f"  {len(cases):,} cases ({sum(1 for c in cases if c['audio_src']):,} with audio)")

    for suite in args.suites:
        out_dir = DATASETS_DIR / f"sire-{suite}"
        chosen = sorted(select(cases, SUITES[suite]), key=lambda c: (c["usage"], c["lane"], c["case_key"]))
        copy_audio = suite != "full"
        if copy_audio and (out_dir / "audio").exists():
            shutil.rmtree(out_dir / "audio")
        manifest, harness = [], []
        for case in chosen:
            source = corpus / case["audio_src"]
            if copy_audio:
                rel = f"audio/{case['case_key']}.wav"
                (out_dir / "audio").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(long_path(source), out_dir / rel)
            else:
                rel = os.path.relpath(source, out_dir).replace("\\", "/")
            manifest.append({**case, "audio_path": rel})
            harness.append(harness_row(case, rel))
        write_jsonl(out_dir / "cases.jsonl", manifest)
        write_jsonl(out_dir / "voicelive_harness.jsonl", harness)
        summary = summarize(manifest)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"sire-{suite}: {summary['cases']:,} cases, {summary['intents']} intents, "
              f"{summary['entity_slots']:,} entity slots -> {out_dir}")


if __name__ == "__main__":
    main()
