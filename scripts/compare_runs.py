"""Compare a candidate run against a baseline run and apply regression gates.

    python scripts/compare_runs.py <baseline-run> <candidate-run>
    python scripts/compare_runs.py latest:smoke-baseline latest:smoke --markdown report.md

Runs are local folders under agent/.foundry/results/ (or 'latest:<suite>'). Metrics are recomputed
with the same grader functions as the Foundry evaluators, on the cases both runs share. A metric fails
its gate (agent/regression-gates.yaml) when it drops by more than `max_drop` AND a paired exact McNemar
test on the flipped cases says the drop is unlikely to be run-to-run noise (p <= alpha). Exit code is
1 when a gate fails, so this can gate a CI pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.runs import load_run, resolve_run_dir  # noqa: E402
from sire_eval.scoring import METRICS, THRESHOLDS, aggregate, paired_flips, regression_p_value, score_row  # noqa: E402
from sire_eval.settings import AGENT_DIR  # noqa: E402


def lane_key(row: dict) -> str:
    return f"{row.get('usage', '')[:5]}/{row.get('lane')}"


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--gates", default=str(AGENT_DIR / "regression-gates.yaml"))
    parser.add_argument("--markdown", default=None, help="Also write the report to this file.")
    parser.add_argument("--max-flips", type=int, default=15)
    args = parser.parse_args()

    base_dir, cand_dir = resolve_run_dir(args.baseline), resolve_run_dir(args.candidate)
    base_rows, base_info = load_run(base_dir)
    cand_rows, cand_info = load_run(cand_dir)
    base_by = {r["case_key"]: r for r in base_rows}
    cand_by = {r["case_key"]: r for r in cand_rows}
    shared = sorted(set(base_by) & set(cand_by))
    b = [base_by[c] for c in shared]
    c = [cand_by[c] for c in shared]
    bs, cs = aggregate(b), aggregate(c)
    gates = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8"))

    lines = [f"# Regression report: `{cand_info.get('label', cand_dir.name)}` vs `{base_info.get('label', base_dir.name)}`", ""]
    for name, info, d in (("Baseline", base_info, base_dir), ("Candidate", cand_info, cand_dir)):
        lines.append(f"- **{name}**: `{d.name}` - llm={info.get('model')} stt={info.get('stt_model')} "
                     f"search={info.get('search_profile')} agent={info.get('agent_name')}:v{info.get('agent_version')}")
    lines += [f"- Shared cases: **{len(shared)}**", "", "## Metrics (pass rate at the evaluator threshold)", "",
              "Flips = cases that went pass -> fail / fail -> pass. p = one-sided exact McNemar test on the flips "
              "(chance of a drop at least this large from run-to-run noise alone).", "",
              "| metric | threshold | baseline | candidate | delta | flips -/+ | p | gate |",
              "|---|---|---|---|---|---|---|---|"]
    default_alpha = float(gates.get("alpha", 0.05))
    base_scores, cand_scores = [score_row(r) for r in b], [score_row(r) for r in c]
    failures, warnings, watch = [], [], []
    for metric, _, _ in METRICS:
        bm, cm = bs["metrics"].get(metric), cs["metrics"].get(metric)
        if not bm or not cm:
            continue
        delta = cm["pass_rate"] - bm["pass_rate"]
        lost, gained = paired_flips(base_scores, cand_scores, metric)
        p = regression_p_value(lost, gained)
        gate = (gates.get("metrics") or {}).get(metric)
        status = ""
        if gate is not None:
            max_drop, alpha = float(gate.get("max_drop", 0)), float(gate.get("alpha", default_alpha))
            beyond = delta < -max_drop - 1e-9
            if beyond and p <= alpha:
                status = "**FAIL**"
                failures.append(f"{metric} dropped {delta:+.1%} (allowed -{max_drop:.1%}); {lost} cases lost vs "
                                f"{gained} gained, p={p:.3f} <= {alpha}")
            elif beyond:
                status = "noise"
                warnings.append(f"{metric} dropped {delta:+.1%} (allowed -{max_drop:.1%}) but {lost} lost vs "
                                f"{gained} gained is within run-to-run noise (p={p:.3f} > {alpha}); "
                                "re-run or confirm on a larger suite")
            elif delta < 0 and p <= alpha:
                status = "pass (watch)"
                watch.append(f"{metric} dropped {delta:+.1%}: within the allowed -{max_drop:.1%}, but consistent "
                             f"({lost} lost vs {gained} gained, p={p:.3f}) - likely a real, small effect")
            else:
                status = "pass"
        lines.append(f"| {metric} | {THRESHOLDS[metric]} | {fmt(bm['pass_rate'])} | {fmt(cm['pass_rate'])} | "
                     f"{delta:+.1%} | {lost}/{gained} | {p:.2f} | {status} |")

    lat_gate = (gates.get("latency") or {}).get("p95_ms")
    bp, cp = bs["latency_ms"]["p95"], cs["latency_ms"]["p95"]
    lat_status = ""
    if lat_gate and bp and cp:
        growth = (cp - bp) / bp * 100
        ok = growth <= float(lat_gate["max_increase_pct"])
        lat_status = "pass" if ok else "**FAIL**"
        if not ok:
            failures.append(f"latency p95 grew {growth:+.0f}% (allowed +{lat_gate['max_increase_pct']}%)")
    lines += ["", "## Latency (end of speech to first tool call)", "", "| | baseline | candidate | gate |", "|---|---|---|---|",
              f"| p50 ms | {fmt(bs['latency_ms']['p50'], False)} | {fmt(cs['latency_ms']['p50'], False)} | |",
              f"| p95 ms | {fmt(bp, False)} | {fmt(cp, False)} | {lat_status} |",
              f"| search p50 ms | {fmt(bs['search_ms']['p50'], False)} | {fmt(cs['search_ms']['p50'], False)} | |"]

    lines += ["", "## By lane (task success / entity top-1)", "", "| lane | n | task base | task cand | entity base | entity cand |",
              "|---|---|---|---|---|---|"]
    lanes = sorted({lane_key(r) for r in b})
    for lane in lanes:
        lb = aggregate([r for r in b if lane_key(r) == lane])
        lc = aggregate([r for r in c if lane_key(r) == lane])
        tb, tc = lb["metrics"].get("task_success", {}), lc["metrics"].get("task_success", {})
        eb, ec = lb["metrics"].get("entity_top1", {}), lc["metrics"].get("entity_top1", {})
        lines.append(f"| {lane} | {lb['cases']} | {fmt(tb.get('pass_rate'))} | {fmt(tc.get('pass_rate'))} | "
                     f"{fmt(eb.get('pass_rate'))} | {fmt(ec.get('pass_rate'))} |")

    regressions, fixes = [], []
    for cid, sb, sc in zip(shared, base_scores, cand_scores):
        if sb.get("task_success", 0) >= 0.5 > sc.get("task_success", 0):
            regressions.append(cand_by[cid])
        elif sc.get("task_success", 0) >= 0.5 > sb.get("task_success", 0):
            fixes.append(cand_by[cid])
    for title, rows in (("Regressions (task success: pass -> fail)", regressions), ("Fixes (task success: fail -> pass)", fixes)):
        lines += ["", f"## {title}: {len(rows)}", ""]
        for r in rows[: args.max_flips]:
            base = base_by[r["case_key"]]
            lines.append(f"- `{r['case_key']}` \"{r['gold_transcript']}\" | heard: \"{r.get('transcript', '')}\" | "
                         f"intent {base.get('predicted_intent')}->{r.get('predicted_intent')} (gold {r['expected_intent']}) | "
                         f"tools {base.get('called_tools')}->{r.get('called_tools')}")

    for label, d in (("Baseline", base_dir), ("Candidate", cand_dir)):
        runs_file = d / "foundry_runs.json"
        if runs_file.exists():
            runs = json.loads(runs_file.read_text(encoding="utf-8")).get("runs") or {}
            links = ", ".join(f"[{k}]({v.get('report_url')})" for k, v in runs.items() if v.get("report_url"))
            if links:
                lines.append(f"\n{label} Foundry reports: {links}")

    verdict = "PASS" if not failures else "FAIL"
    lines += ["", f"## Gate verdict: **{verdict}**"] + [f"- {f}" for f in failures]
    if warnings:
        lines += ["", "Within noise (not failing):"] + [f"- {w}" for w in warnings]
    if watch:
        lines += ["", "Watch (passing, but a consistent small drop):"] + [f"- {w}" for w in watch]
    report = "\n".join(lines)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report + "\n", encoding="utf-8")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
