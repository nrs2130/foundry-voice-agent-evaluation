"""Grader functions for the SIRE evaluators.

Each ``grade_*`` function takes ``(sample, item)`` like a Foundry code-based evaluator and
returns a float. ``sire_eval/evaluators.py`` ships these functions (plus ``textnorm.py``) as
the evaluator ``code_text``; local reports import them directly, so cloud and local scores
always come from the same code.

Standard library only: this source is executed inside the Foundry evaluation sandbox.
"""

try:
    from .textnorm import compact, entity_word_recall, transcript_exact_match, word_error_rate
except ImportError:  # In the Foundry sandbox the textnorm source is prepended to this module.
    pass


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return list(value)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        import json

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _norm_choice(value):
    import re

    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _choice_equal(expected, predicted):
    """Options are compared ignoring case, spacing and punctuation ("I'm yelling" == "im yelling")."""
    return _norm_choice(expected) == _norm_choice(predicted) or compact(expected or "") == compact(predicted or "")  # noqa: F821


def grade_stt_word_accuracy(sample, item):
    """1 - WER between the gold transcript and the service's input transcription (normalized)."""
    wer = word_error_rate(item.get("gold_transcript") or "", item.get("transcript") or "")  # noqa: F821
    return max(0.0, 1.0 - wer)


def grade_stt_exact_match(sample, item):
    """1 when the normalized transcription equals the gold transcript (spacing-insensitive)."""
    return 1.0 if transcript_exact_match(item.get("gold_transcript") or "", item.get("transcript") or "") else 0.0  # noqa: F821


def grade_stt_entity_recall(sample, item):
    """Share of entity-mention words heard correctly; same-sounding spellings count except in spelled lanes."""
    mentions = _as_list(item.get("entity_mentions"))
    spelled = (item.get("lane") or "") == "spelled"
    return entity_word_recall(mentions, item.get("transcript") or "", soundalike=not spelled)  # noqa: F821


def grade_intent_accuracy(sample, item):
    expected = (item.get("expected_intent") or "").strip().lower()
    predicted = (item.get("predicted_intent") or "").strip().lower()
    return 1.0 if expected and expected == predicted else 0.0


def grade_tool_selection(sample, item):
    """1 when the set of tools called equals the expected set (e.g. {search_user, search_group})."""
    expected = set(_as_list(item.get("expected_tools")))
    called = set(_as_list(item.get("called_tools")))
    return 1.0 if expected and expected == called else 0.0


def _slot_hits(expected_entities, predicted_entities):
    tops = {str(p.get("id")) for p in _as_list(predicted_entities) if isinstance(p, dict) and p.get("id")}
    hits = 0
    slots = [e for e in _as_list(expected_entities) if isinstance(e, dict) and e.get("urns")]
    for slot in slots:
        if tops.intersection(slot["urns"]):
            hits += 1
    return hits, len(slots)


def grade_entity_top1(sample, item):
    """Share of gold entity slots resolved by the top-1 result of some search call."""
    hits, total = _slot_hits(item.get("expected_entities"), item.get("predicted_entities"))
    return 1.0 if total == 0 else hits / float(total)


def grade_entity_recall_at_5(sample, item):
    """Share of gold entity slots whose URN appears anywhere in the returned candidates (top-5 per call)."""
    candidates = {str(c) for c in _as_list(item.get("candidate_ids"))}
    slots = [e for e in _as_list(item.get("expected_entities")) if isinstance(e, dict) and e.get("urns")]
    if not slots:
        return 1.0
    return sum(1 for s in slots if candidates.intersection(s["urns"])) / float(len(slots))


def grade_no_confident_wrong(sample, item):
    """Safety: 0 if any search auto-confirmed (confident) an entity that is not in the gold answer."""
    gold = set()
    for slot in _as_list(item.get("expected_entities")):
        if isinstance(slot, dict):
            gold.update(slot.get("urns") or [])
    for pred in _as_list(item.get("predicted_entities")):
        if isinstance(pred, dict) and pred.get("confident") and str(pred.get("id")) not in gold:
            return 0.0
    return 1.0


def grade_choice_accuracy(sample, item):
    expected = _as_dict(item.get("expected_choice"))
    predicted = _as_dict(item.get("predicted_choice"))
    if not expected or not predicted:
        return 0.0
    if (expected.get("kind") or "") != (predicted.get("kind") or ""):
        return 0.0
    if expected.get("kind") == "choice":
        return 1.0 if _choice_equal(expected.get("value"), predicted.get("value")) else 0.0
    return 1.0


def grade_task_success(sample, item):
    """End to end: CHOOSE needs the right pick; otherwise the intent and every entity slot must be right."""
    if (item.get("flow") or "") == "CHOOSE":
        return grade_choice_accuracy(sample, item)
    if grade_intent_accuracy(sample, item) < 1.0:
        return 0.0
    hits, total = _slot_hits(item.get("expected_entities"), item.get("predicted_entities"))
    return 1.0 if hits == total else 0.0


def grade_latency_sla(sample, item):
    """1 when end-of-speech to first tool call is within the run's latency budget."""
    try:
        latency = float(item.get("latency_ms"))
        budget = float(item.get("latency_budget_ms") or 1500)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if 0 <= latency <= budget else 0.0
