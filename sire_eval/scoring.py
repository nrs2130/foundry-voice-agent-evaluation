"""Local scoring with the same grader functions the Foundry evaluators run.

Used for instant run summaries, per-lane breakdowns and CI regression gates without waiting for
the cloud evaluation (which remains the system of record in the Foundry portal).
"""

from __future__ import annotations

import collections
import math
import statistics
from typing import Any, Iterable

from . import grader_functions as g
from .evaluators import BY_NAME, EVAL_GROUPS

METRICS = [
    ("stt_word_accuracy", "e2e", g.grade_stt_word_accuracy),
    ("stt_exact_match", "e2e", g.grade_stt_exact_match),
    ("intent_accuracy", "e2e", g.grade_intent_accuracy),
    ("tool_selection", "e2e", g.grade_tool_selection),
    ("task_success", "e2e", g.grade_task_success),
    ("latency_sla", "e2e", g.grade_latency_sla),
    ("stt_entity_recall", "entities", g.grade_stt_entity_recall),
    ("entity_top1", "entities", g.grade_entity_top1),
    ("entity_recall_at_5", "entities", g.grade_entity_recall_at_5),
    ("no_confident_wrong", "entities", g.grade_no_confident_wrong),
    ("choice_accuracy", "choice", g.grade_choice_accuracy),
]
# Pass thresholds mirror the Foundry evaluator thresholds.
THRESHOLDS = {m: BY_NAME[f"sire-{m.replace('_', '-')}"].threshold for m, _, _ in METRICS}


def score_row(row: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for metric, group, fn in METRICS:
        if EVAL_GROUPS[group].applies(row):
            scores[metric] = float(fn({}, row))
    return scores


def paired_flips(base_scores: list[dict[str, float]], cand_scores: list[dict[str, float]], metric: str) -> tuple[int, int]:
    """(lost, gained) over paired cases: passing in the baseline but failing in the candidate, and vice versa."""
    threshold = THRESHOLDS[metric]
    lost = gained = 0
    for sb, sc in zip(base_scores, cand_scores):
        if metric not in sb or metric not in sc:
            continue
        passed_before, passed_after = sb[metric] >= threshold, sc[metric] >= threshold
        lost += passed_before and not passed_after
        gained += passed_after and not passed_before
    return lost, gained


def regression_p_value(lost: int, gained: int) -> float:
    """One-sided exact McNemar (sign) test on the flipped cases.

    The chance of at least ``lost`` pass->fail flips among ``lost + gained`` flips if the change had no
    effect; a small value means the drop is unlikely to be run-to-run noise of the realtime model.
    """
    n = lost + gained
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(lost, n + 1)) / 2 ** n


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    per_metric: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        for metric, value in score_row(row).items():
            per_metric[metric].append(value)
    metrics = {}
    for metric, values in per_metric.items():
        threshold = THRESHOLDS[metric]
        metrics[metric] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "pass_rate": round(sum(1 for v in values if v >= threshold) / len(values), 4),
        }
    latencies = [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float)) and r["latency_ms"] >= 0]
    searches = [float(r["search_ms"]) for r in rows if isinstance(r.get("search_ms"), (int, float)) and r["search_ms"] > 0]
    return {
        "cases": len(rows),
        "errors": sum(1 for r in rows if r.get("status") == "error"),
        "metrics": metrics,
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p90": _percentile(latencies, 0.9), "p95": _percentile(latencies, 0.95)},
        "search_ms": {"p50": _percentile(searches, 0.5), "p95": _percentile(searches, 0.95)},
    }


def breakdown(rows: list[dict[str, Any]], key: str = "lane") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        label = f"{row.get('usage', '')[:5]}/{row.get(key)}" if key == "lane" else str(row.get(key))
        groups[label].append(row)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def format_summary(summary: dict[str, Any], title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"cases={summary['cases']} errors={summary['errors']} "
                 f"latency p50={summary['latency_ms']['p50']}ms p95={summary['latency_ms']['p95']}ms "
                 f"search p50={summary['search_ms']['p50']}ms")
    for metric, _, _ in METRICS:
        m = summary["metrics"].get(metric)
        if m:
            lines.append(f"  {metric:22s} mean={m['mean']:.3f} pass={m['pass_rate']:.1%} (n={m['n']})")
    return "\n".join(lines)
