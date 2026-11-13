"""Unit tests for the normalizer, graders, evaluator code shipped to Foundry, regression-gate statistics and
the Foundry submission retry logic (with a fake client)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval import grader_functions as g  # noqa: E402
from sire_eval.evaluators import EVALUATORS, code_text  # noqa: E402
from sire_eval.textnorm import entity_word_recall, normalize_text, transcript_exact_match, word_error_rate  # noqa: E402


@pytest.mark.parametrize("gold, heard", [
    ("Transfer this to Room 7781 Larkspur Therapist Units", "transfer this to room seventy seven hundred and eighty one larkspur therapist units"),
    ("Call Room 862 Bed 3 PCNA urgently", "call room eight six two bed three P.C.N.A. urgently"),
    ("Log me in as ottolinebrackwater", "log me in as o t t o l i n e b r a c k w a t e r"),
    ("Set a reminder for Doctor Ferncastle at 11", "set a reminder for Dr. Ferncastle at 11:00"),
    ("Who is in 4northicu", "who is in four n o r t h i c u"),
])
def test_normalizer_equates_spoken_forms(gold, heard):
    assert transcript_exact_match(gold, heard), (normalize_text(gold), normalize_text(heard))


def test_wer_counts_real_mishearings():
    assert word_error_rate("Call Ottoline Brackwater", "Call Ottoline Brackwood") == pytest.approx(1 / 3)
    assert word_error_rate("enable do not disturb for sixty minutes", "Enable Do Not Disturb for 60 minutes.") == 0.0


def test_entity_recall_soundalike_and_spelled():
    assert entity_word_recall(["brackwater"], "call brakwater now") == 1.0
    assert entity_word_recall(["brackwater"], "call brakwater now", soundalike=False) == 0.0
    assert entity_word_recall(["zephyrinebramwell"], "this is z e p h y r i n e b r a m w e l l") == 1.0


def _row(**kw):
    base = {"flow": "MAIN_MENU", "expected_intent": "call", "predicted_intent": "call",
            "expected_entities": [{"slot": "CALLEE", "type": "user", "urns": ["u-a", "u-b"], "names": ["A", "B"]}],
            "predicted_entities": [{"id": "u-b", "confident": True}], "candidate_ids": ["u-b", "u-c"],
            "expected_tools": ["search_user"], "called_tools": ["search_user"],
            "expected_choice": {"kind": "", "value": ""}, "predicted_choice": {"kind": "", "value": ""}}
    base.update(kw)
    return base


def test_entity_and_task_graders():
    assert g.grade_entity_top1({}, _row()) == 1.0
    assert g.grade_task_success({}, _row()) == 1.0
    wrong = _row(predicted_entities=[{"id": "u-x", "confident": True}], candidate_ids=["u-x"])
    assert g.grade_entity_top1({}, wrong) == 0.0
    assert g.grade_entity_recall_at_5({}, wrong) == 0.0
    assert g.grade_no_confident_wrong({}, wrong) == 0.0
    assert g.grade_task_success({}, _row(predicted_intent="page")) == 0.0
    assert g.grade_tool_selection({}, _row(called_tools=["search_user", "route_command"])) == 0.0


def test_choice_grader():
    row = _row(flow="CHOOSE", expected_choice={"kind": "choice", "value": "im yelling"},
               predicted_choice={"kind": "choice", "value": "I'm yelling"})
    assert g.grade_choice_accuracy({}, row) == 1.0
    assert g.grade_task_success({}, row) == 1.0
    assert g.grade_choice_accuracy({}, _row(flow="CHOOSE", expected_choice={"kind": "exit", "value": ""},
                                            predicted_choice={"kind": "none", "value": ""})) == 0.0


@pytest.mark.parametrize("evaluator", EVALUATORS, ids=lambda e: e.name)
def test_evaluator_code_is_self_contained(evaluator):
    namespace: dict = {"__name__": "foundry_sandbox"}
    exec(compile(code_text(evaluator), evaluator.name, "exec"), namespace)  # noqa: S102 - mirrors the cloud sandbox
    item = {**_row(), "gold_transcript": "Call Zephyrine", "transcript": "call zephyrine", "entity_mentions": ["zephyrine"],
            "lane": "name", "latency_ms": 500, "latency_budget_ms": 1500}
    score = namespace["grade"]({}, {k: item[k] for k in evaluator.fields})
    assert 0.0 <= score <= 1.0


def test_code_grader_error_rate_ignores_llm_judges():
    from sire_eval.evaluators import EVAL_GROUPS
    from sire_eval.foundry_evals import code_grader_error_rate

    summary = {"per_criterion": [
        {"criterion": "choice-accuracy", "passed": 108, "failed": 7, "errored": 49},
        {"criterion": "tool-selection", "passed": 121, "failed": 2, "errored": 41},
        {"criterion": "llm-tool-call-accuracy", "passed": 10, "failed": 0, "errored": 154},
    ]}
    assert code_grader_error_rate(summary, EVAL_GROUPS["choice"]) == pytest.approx(49 / 164)
    clean = {"per_criterion": [{"criterion": "choice-accuracy", "passed": 15, "failed": 2, "errored": 0},
                               {"criterion": "llm-tool-call-accuracy", "passed": 1, "failed": 0, "errored": 16}]}
    assert code_grader_error_rate(clean, EVAL_GROUPS["choice"]) == 0.0


def test_logic_signature_ignores_docs_only_edits():
    from sire_eval.evaluators import logic_signature

    base = 'def f(x):\n    """Doc A."""\n    return x + 1  # note\n'
    assert logic_signature(base) == logic_signature('def f(x):\n    """Doc B, reworded."""\n    return x + 1\n')
    assert logic_signature(base) != logic_signature('def f(x):\n    """Doc A."""\n    return x + 2\n')


def test_regression_p_value_sign_test():
    from sire_eval.scoring import regression_p_value

    assert regression_p_value(0, 0) == 1.0
    assert regression_p_value(6, 1) == pytest.approx(8 / 128)
    assert regression_p_value(1, 1) == pytest.approx(0.75)
    assert regression_p_value(10, 0) < 0.001


def test_paired_flips_counts_only_shared_applicable_cases():
    from sire_eval.scoring import paired_flips

    base = [{"task_success": 1.0}, {"task_success": 0.0}, {"task_success": 1.0}, {}]
    cand = [{"task_success": 0.0}, {"task_success": 1.0}, {"task_success": 1.0}, {"task_success": 0.0}]
    assert paired_flips(base, cand, "task_success") == (1, 1)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        return dict(self.__dict__)


class _FakeRuns:
    def __init__(self, runs, retry_errored):
        self.runs, self.retry_errored, self.deleted, self.created = runs, retry_errored, [], 0
        self.output_items = _Obj(list=lambda **_: [])

    def _run(self, run_id, errored):
        crit = [_Obj(testing_criteria=n, passed=10 - errored, failed=0, errored=errored)
                for n in ("choice-accuracy", "stt-word-accuracy", "tool-selection")]
        return _Obj(id=run_id, eval_id="e1", status="completed", result_counts=None,
                    per_testing_criteria_results=crit, report_url=None, error=None)

    def retrieve(self, eval_id, run_id):
        return self.runs[run_id]

    def create(self, eval_id, name, metadata, data_source):
        self.created += 1
        run = self._run(f"retry-{self.created}", self.retry_errored)
        self.runs[run.id] = run
        return run

    def delete(self, eval_id, run_id):
        self.deleted.append(run_id)


def _fake_client(retry_errored):
    runs = _FakeRuns({}, retry_errored)
    runs.runs["orig"] = runs._run("orig", 3)
    return _Obj(evals=_Obj(runs=runs)), runs


@pytest.mark.parametrize("retry_errored, kept, deleted", [(0, "retry-1", "orig"), (5, "orig", "retry-1")])
def test_wait_with_retries_keeps_the_cleaner_run(retry_errored, kept, deleted):
    from sire_eval.foundry_evals import wait_with_retries

    client, runs = _fake_client(retry_errored)
    rows = [{**_row(flow="CHOOSE"), "case_key": "c1", "gold_transcript": "one", "transcript": "one"}]
    triples, final, retries = wait_with_retries(client, [("choice", "e1", "orig")], rows, "name", {"label": "x"})
    assert triples == [("choice", "e1", kept)]
    assert runs.deleted == [deleted]
    assert retries[0]["kept"] == kept and final["choice"].id == kept


def test_collect_resumes_submission_and_deletes_replaced_runs_once_complete(tmp_path, monkeypatch):
    import json

    import sire_eval.runs as runs_module

    client, runs = _fake_client(0)
    runs.runs["new"] = runs._run("new", 0)
    row = {**_row(flow="CHOOSE"), "case_key": "c1", "gold_transcript": "one", "transcript": "one"}
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"label": "x", "suite": "smoke"}), encoding="utf-8")
    kept_e2e = {"group": "e2e", "eval_id": "e2", "run_id": "kept-e2e"}
    pending_entities = {"group": "entities", "eval_id": "e3", "run_id": "old-entities"}
    (tmp_path / "foundry_runs.json").write_text(json.dumps({
        "submitted": [kept_e2e, {"group": "choice", "eval_id": "e1", "run_id": "new"}],
        "runs": {"e2e": {"status": "completed"}},
        "replacing": [{"group": "choice", "eval_id": "e1", "run_id": "orig"}, pending_entities],
    }), encoding="utf-8")
    monkeypatch.setattr(runs_module, "project_client", lambda: _Obj(get_openai_client=lambda: client))

    summary = runs_module.collect_run_dir(tmp_path, groups=["choice"])

    assert runs.deleted == ["orig"]
    assert summary["replacing"] == [pending_entities]
    assert summary["submitted"][0] == kept_e2e and summary["submitted"][1]["run_id"] == "new"
    assert set(summary["runs"]) == {"e2e", "choice"}
    assert json.loads((tmp_path / "foundry_runs.json").read_text(encoding="utf-8")) == summary
    assert (tmp_path / "foundry_output_choice.json").exists()
