"""SIRE evaluator catalog and the three Foundry evaluation scorecards.

Custom evaluators are registered in the project's evaluator catalog as versioned code-based
evaluators whose code is ``textnorm.py`` + ``grader_functions.py``. Each eval group (scorecard)
only receives the rows its metrics apply to, so pass rates have honest denominators.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import grader_functions, textnorm
from .settings import EVALUATIONS_PREVIEW_HEADERS, EVALUATORS_DIR


@dataclass(frozen=True)
class SireEvaluator:
    name: str
    display_name: str
    description: str
    function: str
    fields: dict[str, str]
    threshold: float
    category: str = "quality"


EVALUATORS: list[SireEvaluator] = [
    SireEvaluator("sire-stt-word-accuracy", "SIRE STT word accuracy (1 - WER)",
                  "1 - word error rate between the gold transcript and the voice agent's input transcription, "
                  "after production-style normalization (numbers, spelled letters, fillers, punctuation).",
                  "grade_stt_word_accuracy", {"gold_transcript": "string", "transcript": "string"}, 0.8),
    SireEvaluator("sire-stt-exact-match", "SIRE STT exact transcript match",
                  "1 when the normalized transcription matches the gold transcript exactly (spacing-insensitive).",
                  "grade_stt_exact_match", {"gold_transcript": "string", "transcript": "string"}, 0.5),
    SireEvaluator("sire-stt-entity-recall", "SIRE STT entity word recall",
                  "Share of the spoken entity words (names, groups) present in the transcription; same-sounding "
                  "spellings count except in spelled lanes.",
                  "grade_stt_entity_recall", {"entity_mentions": "array", "transcript": "string", "lane": "string"}, 1.0),
    SireEvaluator("sire-intent-accuracy", "SIRE intent accuracy",
                  "1 when the intent passed to the tools equals the gold intent label.",
                  "grade_intent_accuracy", {"expected_intent": "string", "predicted_intent": "string"}, 0.5, "agents"),
    SireEvaluator("sire-tool-selection", "SIRE tool selection",
                  "1 when the set of tools called equals the expected set (search_user / search_group / route_command / pick_choice).",
                  "grade_tool_selection", {"expected_tools": "array", "called_tools": "array"}, 0.5, "agents"),
    SireEvaluator("sire-entity-top1", "SIRE entity resolution top-1",
                  "Share of gold entity slots resolved by the top-1 directory search result.",
                  "grade_entity_top1", {"expected_entities": "array", "predicted_entities": "array"}, 1.0, "agents"),
    SireEvaluator("sire-entity-recall-at-5", "SIRE entity recall@5",
                  "Share of gold entity slots present anywhere in the returned search candidates.",
                  "grade_entity_recall_at_5", {"expected_entities": "array", "candidate_ids": "array"}, 1.0, "agents"),
    SireEvaluator("sire-no-confident-wrong", "SIRE no confident wrong entity",
                  "Safety: 0 when a search auto-confirmed (_confident) an entity that is not the gold entity.",
                  "grade_no_confident_wrong", {"expected_entities": "array", "predicted_entities": "array"}, 0.5, "safety"),
    SireEvaluator("sire-choice-accuracy", "SIRE constrained-choice accuracy",
                  "1 when pick_choice kind and value match the gold choice.",
                  "grade_choice_accuracy", {"expected_choice": "object", "predicted_choice": "object"}, 0.5, "agents"),
    SireEvaluator("sire-task-success", "SIRE end-to-end task success",
                  "1 when the request is fully handled: right choice in the CHOOSE flow, otherwise right intent and every entity slot resolved at top-1.",
                  "grade_task_success",
                  {"flow": "string", "expected_intent": "string", "predicted_intent": "string", "expected_entities": "array",
                   "predicted_entities": "array", "expected_choice": "object", "predicted_choice": "object"}, 0.5, "agents"),
    SireEvaluator("sire-latency-sla", "SIRE latency within budget",
                  "1 when end-of-speech to first tool call is within the run's latency budget (latency_budget_ms).",
                  "grade_latency_sla", {"latency_ms": "number", "latency_budget_ms": "number"}, 0.5),
]
BY_NAME = {e.name: e for e in EVALUATORS}


@dataclass(frozen=True)
class EvalGroup:
    key: str
    title: str
    evaluators: list[str]
    applies: Callable[[dict[str, Any]], bool]
    builtins: list[str] = field(default_factory=list)
    llm_judges: list[str] = field(default_factory=list)


EVAL_GROUPS: dict[str, EvalGroup] = {
    "e2e": EvalGroup(
        "e2e", "SIRE voice - end-to-end",
        ["sire-stt-word-accuracy", "sire-stt-exact-match", "sire-intent-accuracy", "sire-tool-selection",
         "sire-task-success", "sire-latency-sla"],
        lambda row: True,
        llm_judges=["intent_resolution", "tool_call_accuracy"],
    ),
    "entities": EvalGroup(
        "entities", "SIRE voice - entity resolution",
        ["sire-stt-entity-recall", "sire-entity-top1", "sire-entity-recall-at-5", "sire-no-confident-wrong"],
        lambda row: row.get("flow") != "CHOOSE" and bool(row.get("expected_entities")),
        builtins=["document_retrieval"],
    ),
    "choice": EvalGroup(
        "choice", "SIRE voice - constrained choice",
        ["sire-choice-accuracy", "sire-stt-word-accuracy", "sire-tool-selection"],
        lambda row: row.get("flow") == "CHOOSE",
        llm_judges=["tool_call_accuracy"],
    ),
}


def code_text(evaluator: SireEvaluator) -> str:
    return "\n".join([
        inspect.getsource(textnorm),
        inspect.getsource(grader_functions),
        f"\n\ndef grade(sample, item):\n    return float({evaluator.function}(sample, item or {{}}))\n",
    ])


def _definition(evaluator: SireEvaluator):
    from azure.ai.projects.models import CodeBasedEvaluatorDefinition, EvaluatorMetric

    return CodeBasedEvaluatorDefinition(
        code_text=code_text(evaluator),
        data_schema={
            "type": "object",
            "properties": {k: {"type": v} for k, v in evaluator.fields.items()},
            "required": list(evaluator.fields),
        },
        init_parameters={"type": "object", "properties": {"threshold": {"type": "number"}}, "required": []},
        metrics={"result": EvaluatorMetric(type="continuous", desirable_direction="increase", min_value=0.0,
                                           max_value=1.0, threshold=evaluator.threshold)},
    )


BUILTIN_EVALUATORS = ("builtin.document_retrieval", "builtin.intent_resolution", "builtin.tool_call_accuracy")


def logic_signature(code: str) -> str:
    """Code with comments and docstrings removed, so documentation-only edits do not mint new versions."""
    import ast

    tree = ast.parse(code)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def register_evaluators(project, force: bool = False) -> dict[str, str]:
    """Create (or reuse when the code is unchanged) a catalog version for every SIRE evaluator.

    Returns the versions to pin in scorecards, including the current built-in evaluator versions.
    """
    versions: dict[str, str] = {}
    EVALUATORS_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILTIN_EVALUATORS:
        builtin = project.evaluators.get_version(name=name, version="latest", headers=EVALUATIONS_PREVIEW_HEADERS)
        versions[name] = str(builtin.version)
    for ev in EVALUATORS:
        definition = _definition(ev)
        latest = None
        try:
            latest = project.evaluators.get_version(name=ev.name, version="latest", headers=EVALUATIONS_PREVIEW_HEADERS)
        except Exception:  # noqa: BLE001 - not registered yet
            latest = None
        if latest is not None and not force:
            current = (latest.as_dict().get("definition") or {}).get("code_text")
            if current and logic_signature(current) == logic_signature(definition.code_text):
                versions[ev.name] = str(latest.version)
                continue
        created = project.evaluators.create_version(
            name=ev.name,
            evaluator_version={
                "name": ev.name,
                "evaluator_type": "custom",
                "categories": [ev.category],
                "display_name": ev.display_name,
                "description": ev.description,
                "definition": definition,
                "metadata": {"source": "sire_eval/grader_functions.py", "function": ev.function},
            },
            headers=EVALUATIONS_PREVIEW_HEADERS,
        )
        versions[ev.name] = str(created.version)
        (EVALUATORS_DIR / f"{ev.name}-v{created.version}.json").write_text(
            json.dumps({"name": ev.name, "version": str(created.version), "display_name": ev.display_name,
                        "description": ev.description, "fields": ev.fields, "threshold": ev.threshold,
                        "function": ev.function}, indent=2), encoding="utf-8")
    return versions


def graders_fingerprint(versions: dict[str, str]) -> str:
    """Short, stable identifier of a set of pinned evaluator versions."""
    import hashlib

    text = ",".join(f"{k}@{v}" for k, v in sorted(versions.items()))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def testing_criteria(group: EvalGroup, judge_deployment: str | None, llm_judges: bool,
                     versions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Testing criteria for a scorecard, pinned to ``versions`` so every run is scored by the same code."""
    versions = versions or {}

    def pin(criterion: dict[str, Any]) -> dict[str, Any]:
        if criterion["evaluator_name"] in versions:
            criterion["evaluator_version"] = versions[criterion["evaluator_name"]]
        return criterion

    criteria: list[dict[str, Any]] = []
    for name in group.evaluators:
        ev = BY_NAME[name]
        criteria.append(pin({
            "type": "azure_ai_evaluator",
            "name": name.removeprefix("sire-"),
            "evaluator_name": name,
            "data_mapping": {f: "{{item.%s}}" % f for f in ev.fields},
            "initialization_parameters": {"threshold": ev.threshold},
        }))
    if "document_retrieval" in group.builtins:
        criteria.append(pin({
            "type": "azure_ai_evaluator",
            "name": "entity-retrieval-ndcg",
            "evaluator_name": "builtin.document_retrieval",
            "data_mapping": {"retrieval_ground_truth": "{{item.retrieval_ground_truth}}",
                             "retrieved_documents": "{{item.retrieved_documents}}"},
            "initialization_parameters": {"ground_truth_label_min": 0, "ground_truth_label_max": 1},
        }))
    if llm_judges and judge_deployment:
        if "intent_resolution" in group.llm_judges:
            criteria.append(pin({
                "type": "azure_ai_evaluator",
                "name": "llm-intent-resolution",
                "evaluator_name": "builtin.intent_resolution",
                "data_mapping": {"query": "{{item.query}}", "response": "{{item.response}}",
                                 "tool_definitions": "{{item.tool_definitions}}"},
                "initialization_parameters": {"deployment_name": judge_deployment},
            }))
        if "tool_call_accuracy" in group.llm_judges:
            criteria.append(pin({
                "type": "azure_ai_evaluator",
                "name": "llm-tool-call-accuracy",
                "evaluator_name": "builtin.tool_call_accuracy",
                "data_mapping": {"query": "{{item.query}}", "tool_calls": "{{item.tool_calls}}",
                                 "tool_definitions": "{{item.tool_definitions}}", "response": "{{item.response}}"},
                "initialization_parameters": {"deployment_name": judge_deployment},
            }))
    return criteria
