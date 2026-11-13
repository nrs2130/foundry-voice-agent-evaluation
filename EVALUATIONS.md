# Building voice-agent evaluations in Microsoft Foundry: step-by-step guide

This guide walks through the **evaluation half** of this repository: how a labelled audio corpus becomes
versioned datasets, custom code evaluators, built-in evaluators, frozen Foundry scorecards, scored runs and
regression gates. It is written for teams who want to reuse the approach for their own voice (or text) agent.

- **[README.md](README.md)**: what the demo shows, quick start and results.
- **[docs/evaluation-design.md](docs/evaluation-design.md)**: metric definitions and design rationale.
- **This guide**: the end-to-end build, in order, with the files and Foundry objects behind each step.

The examples use the demo scenario (SIRE, a clinician voice command router that detects an intent and resolves
a person or group against Azure AI Search), but every step names what is scenario-specific and what to change
for your own agent.

---

## Contents

1. [What you need from this repository](#1-what-you-need-from-this-repository)
2. [How the pieces map to Foundry](#2-how-the-pieces-map-to-foundry)
3. [Prerequisites and configuration](#3-prerequisites-and-configuration)
4. [Step 1 - Prepare the evaluation data](#step-1---prepare-the-evaluation-data)
5. [Step 2 - Register datasets in Foundry (optional)](#step-2---register-datasets-in-foundry-optional)
6. [Step 3 - Create custom code evaluators](#step-3---create-custom-code-evaluators)
7. [Step 4 - Add built-in evaluators](#step-4---add-built-in-evaluators)
8. [Step 5 - Create the evaluation framework (scorecards)](#step-5---create-the-evaluation-framework-scorecards)
9. [Step 6 - Produce results to score](#step-6---produce-results-to-score)
10. [Step 7 - Submit runs and collect results](#step-7---submit-runs-and-collect-results)
11. [Step 8 - Compare runs and apply regression gates](#step-8---compare-runs-and-apply-regression-gates)
12. [Step 9 - Design the tests (suites and experiments)](#step-9---design-the-tests-suites-and-experiments)
13. [Adapting this to your own agent](#adapting-this-to-your-own-agent)
14. [Running in CI](#running-in-ci)
15. [Troubleshooting](#troubleshooting)

---

## 1. What you need from this repository

### Core evaluation code (reuse or adapt)

| File | Role in the evaluation | Reuse |
|---|---|---|
| [`sire_eval/grader_functions.py`](sire_eval/grader_functions.py) | The custom metric logic: one `grade_*(sample, item) -> float` per metric. Shipped verbatim to Foundry as evaluator code and imported locally. | Pattern as-is; replace the metrics with yours. |
| [`sire_eval/textnorm.py`](sire_eval/textnorm.py) | Transcript normalization, WER, exact match and entity-word recall used by the STT graders. Prepended to every evaluator's code. | As-is for any speech-to-text scoring (tune the normalization rules to your domain). |
| [`sire_eval/evaluators.py`](sire_eval/evaluators.py) | Evaluator catalog (`EVALUATORS`), scorecards (`EVAL_GROUPS`), registration with versioning (`register_evaluators`), version pinning and testing-criteria builder (`testing_criteria`). | As-is; edit the two lists. |
| [`sire_eval/foundry_evals.py`](sire_eval/foundry_evals.py) | Eval item schema (`ITEM_FIELDS`), row projection (`to_eval_item`), scorecard creation (`ensure_eval_group`), run submission (inline or file upload), waiting, transient-error retries, output download. | As-is; edit `ITEM_FIELDS`. |
| [`sire_eval/runs.py`](sire_eval/runs.py) | Local run folders (`agent/.foundry/results/<run>/`), run naming/metadata, submit and collect orchestration. | As-is. |
| [`sire_eval/scoring.py`](sire_eval/scoring.py) | Local scoring with the same graders, per-lane breakdowns, paired McNemar test for regression gates. | As-is; edit `METRICS`. |
| [`sire_eval/settings.py`](sire_eval/settings.py) | `.env` loading, paths, credentials, the `Foundry-Features: Evaluations=V1Preview` header. | As-is. |
| [`sire_eval/dataset.py`](sire_eval/dataset.py) | Gold case model (`build_case`) and Voice Live harness rows (`harness_row`). | Adapt `build_case` to your corpus format. |
| [`tests/test_graders.py`](tests/test_graders.py) | Unit tests for normalizer, graders, shipped evaluator code, gate statistics, retry logic. | Pattern as-is. |

### Command-line steps

| Script | Step | Purpose |
|---|---|---|
| [`scripts/prepare_datasets.py`](scripts/prepare_datasets.py) | 1 | Build gold manifests and suites (smoke / regression / full) from the corpus. |
| [`scripts/register_datasets.py`](scripts/register_datasets.py) | 2 | Upload suites as versioned Foundry datasets. |
| [`scripts/register_evaluators.py`](scripts/register_evaluators.py) | 3-4 | Register custom evaluators in the project's evaluator catalog; resolve built-in versions. |
| [`scripts/submit_baseline_from_captures.py`](scripts/submit_baseline_from_captures.py) | 6 | Turn an existing production system's logs into a scored baseline run (no replay needed). |
| [`scripts/run_voice_eval.py`](scripts/run_voice_eval.py) | 6-7 | Replay the audio through the Foundry Voice Agent, score locally, submit to scorecards. |
| [`scripts/replay_search.py`](scripts/replay_search.py) | 6-7 | Re-run only the retrieval step of an existing run with another search profile. |
| [`scripts/submit_run.py`](scripts/submit_run.py) | 7 | Submit (or re-grade) an existing local run folder to the scorecards. |
| [`scripts/collect_runs.py`](scripts/collect_runs.py) | 7 | Wait for submitted runs, retry transient grader failures, download per-item results. |
| [`scripts/compare_runs.py`](scripts/compare_runs.py) | 8 | Paired comparison of two runs with noise-aware regression gates (CI exit code). |
| [`scripts/summarize_runs.py`](scripts/summarize_runs.py) | 8 | Markdown leaderboard of all runs with Foundry report links. |
| [`scripts/oracle_search_check.py`](scripts/oracle_search_check.py) | 1 (optional) | Retrieval-only ceiling using gold entity mentions (no speech, no LLM). |

### Configuration and agent definition

| File | Role |
|---|---|
| [`agent/regression-gates.yaml`](agent/regression-gates.yaml) | Allowed drop per metric, significance level, latency growth budget. |
| [`.env.example`](.env.example) | Project endpoint, subscription, judge deployment, search endpoint, corpus paths. |
| [`requirements.txt`](requirements.txt) / [`requirements-dev.txt`](requirements-dev.txt) | `azure-ai-projects[voice]`, `azure-identity`, `openai`, `pytest`, ... |
| [`agent/instructions.md`](agent/instructions.md), [`agent/intents.yaml`](agent/intents.yaml), [`sire_eval/agent_spec.py`](sire_eval/agent_spec.py) | Agent prompt, intent catalog and tool definitions. The tool definitions are also sent to the LLM judges. |
| [`azure.yaml`](azure.yaml), [`scripts/render_agent_yaml.py`](scripts/render_agent_yaml.py) | The agent under test (generated) and its renderer. Only needed if you evaluate a deployed Foundry Voice Agent. |

### Scenario-specific (replace for your agent)

| File | Why it is specific |
|---|---|
| [`sire_eval/voice_runner.py`](sire_eval/voice_runner.py), [`sire_eval/audio.py`](sire_eval/audio.py) | Drives a Foundry Voice Agent realtime session per utterance and records transcript, tool calls, search candidates and latency. Reusable for any Voice Agent with client-executed tools; the `result_row` mapping is SIRE-specific. |
| [`sire_eval/search_client.py`](sire_eval/search_client.py), [`sire_eval/search_schema.py`](sire_eval/search_schema.py), [`scripts/build_search_indexes.py`](scripts/build_search_indexes.py) | The retrieval system under test (Azure AI Search directory). Replace with your tools / RAG backend. |

### Data (not in the repository)

The labelled corpus lives in `test_data/` and is **git-ignored** (customer data). Generated datasets, evaluator
caches and run results live in `agent/.foundry/` and are also git-ignored. Bring your own corpus in the format
described in [Step 1](#step-1---prepare-the-evaluation-data). Never commit corpus content, and never copy names or
phrases from it into prompts, code or docs.

---

## 2. How the pieces map to Foundry

```text
  labelled corpus ──prepare──► gold cases (cases.jsonl) ──run agent──► result rows (results.jsonl)
                                                                          │ to_eval_item()
                                                                          ▼
  grader_functions.py + textnorm.py ──register──► Evaluator catalog      eval items (fixed schema)
  builtin.* evaluators ─────────────(versions)──► │                       │
                                                  ▼                       ▼
                                   Scorecard = Foundry evaluation  ◄── Run (one per configuration)
                                   (item_schema + pinned criteria)      compared side by side in the portal
```

| Concept | Foundry object | Created by |
|---|---|---|
| Custom metric | Evaluator catalog entry, type **code**, versioned (`sire-*`) | `register_evaluators.py` |
| Built-in metric | `builtin.document_retrieval`, `builtin.intent_resolution`, `builtin.tool_call_accuracy` | referenced by name + pinned version |
| Scorecard | An **evaluation** (`client.evals.create`) with a custom `item_schema` and fixed `testing_criteria` | created on first submission |
| Configuration under test | An evaluation **run** (`client.evals.runs.create`) in each applicable scorecard | `run_voice_eval.py`, `submit_run.py`, ... |
| Test data | Folder **dataset** `sire-voice-<suite>` (manifest + harness JSONL + audio), content-hash versioned | `register_datasets.py` |
| Local workspace | `agent/.foundry/` (`agent-metadata.yaml`, `datasets/`, `evaluators/`, `results/`) | all scripts |

Three scorecards are used, one per capability, so each row only lands where its metrics apply:

| Scorecard | Rows | Custom evaluators | Built-in evaluators |
|---|---|---|---|
| **SIRE voice - end-to-end** | all | stt-word-accuracy, stt-exact-match, intent-accuracy, tool-selection, task-success, latency-sla | intent_resolution, tool_call_accuracy (LLM judges) |
| **SIRE voice - entity resolution** | non-CHOOSE rows that reference a person/group | stt-entity-recall, entity-top1, entity-recall-at-5, no-confident-wrong | document_retrieval |
| **SIRE voice - constrained choice** | CHOOSE-flow rows | choice-accuracy, stt-word-accuracy, tool-selection | tool_call_accuracy (LLM judge) |

---

## 3. Prerequisites and configuration

**Azure**

- A Foundry project in a region that supports Voice Agents and evaluations (the demo used East US 2).
- A chat model deployment for the LLM judges (default `gpt-5.4-mini`). Judge runs are bound by tokens per minute:
  about 1.1M tokens per 120-row end-to-end run, so 500K TPM is recommended.
- Role **Azure AI User** (Foundry User) on the project.
- Only if you replay audio through an agent that calls Azure AI Search: an Azure AI Search service and the
  **Search Service Contributor** + **Search Index Data Contributor** roles. Scoring an existing system from its
  logs ([Step 6a](#6a-baseline-from-existing-production-logs)) does not need it.

**Local tooling**

- Python 3.11+, [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az login`).
- [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) with the
  Foundry agents extension, only if you deploy the agent under test from this repository.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Fill in `.env`:

| Variable | Used for |
|---|---|
| `AZURE_AI_PROJECT_ENDPOINT` | Every Foundry call (evaluators, datasets, evaluations, agent). |
| `AZURE_SUBSCRIPTION_ID`, `AZURE_TENANT_ID` | Scopes the Azure CLI credential to the right tenant. |
| `AZURE_CREDENTIAL` | `cli` (default, `AzureCliCredential`) or `default` (`DefaultAzureCredential`, e.g. managed / workload identity in CI). |
| `FOUNDRY_JUDGE_DEPLOYMENT` | Deployment name passed to the LLM-judge built-in evaluators. |
| `SIRE_AGENT_NAME` | Voice agent under test. |
| `AZURE_SEARCH_*` | Retrieval backend (only for live replay). |
| `SIRE_CORPUS_ROOT`, `SIRE_CAPTURES_ROOT` | Where the extracted corpus and production captures live. |

---

## Step 1 - Prepare the evaluation data

**Goal:** turn a labelled corpus into deterministic, stratified test suites whose every case carries the gold
answer for every metric.

### 1.1 Corpus input contract

`prepare_datasets.py` reads the corpus from `SIRE_CORPUS_ROOT`. To bring your own data, provide the same shape
(or adapt [`build_case`](sire_eval/dataset.py)):

```text
<corpus>/
  phrases-all.jsonl                 one labelled utterance per line
  <usage>/<audio>.wav               the utterance audio (any rate; resampled to 24 kHz PCM16 at run time)
  <usage>/<metadata>.json           optional per-phrase metadata (spoken entity words, caller context)
  entity-lists/users.jsonl.gz       directory entities: entity_urn, display_name, ...
  entity-lists/groups.jsonl.gz
  entity-lists/groupmembers.jsonl.gz
```

Each `phrases-all.jsonl` row needs at least (illustrative values):

```json
{"case_id": "call/0001", "usage": "01-production", "lane": "call",
 "phrase": "Call Doctor Alex Rivera", "intent": "call",
 "slots": {"CALLEE": ["u-0001"]}, "audio": "call/0001.wav", "metadata": "call/0001.json",
 "facility": "s-site-a", "asr_label": "clean"}
```

- `usage` and `lane` define the strata used for suite selection and per-lane reporting.
- `slots` values with a `u-` / `g-` prefix are directory entities to resolve; several URNs in one slot mean
  "any of these is acceptable" (deliberately ambiguous names).
- Constrained-choice rows add `choice_list.values` and the expected `pick_choice` arguments.
- `asr_label` marks phrases a previous speech recognizer misheard; suites over-sample them.

### 1.2 Build the suites

```powershell
# extract the corpus zips first (long paths on Windows: use Python, not Expand-Archive) - see README step 1
python scripts/prepare_datasets.py                 # smoke + regression + full
python scripts/prepare_datasets.py --suites smoke  # one suite
```

Output per suite in `agent/.foundry/datasets/sire-<suite>/`:

| File | Content |
|---|---|
| `cases.jsonl` | Gold manifest: `case_key`, `flow`, `gold_transcript`, `expected_intent`, `expected_entities` (`type`, `urns`, `names`), `expected_tools`, `expected_choice`, `choice_options`, `entity_mentions`, caller context, `asr_label`, `audio_path`. Field meanings: [evaluation-design.md](docs/evaluation-design.md#gold-case-model-casesjsonl). |
| `voicelive_harness.jsonl` | Same cases in the [Voice Live evaluation harness](https://learn.microsoft.com/azure/ai-services/speech-service/how-to-voice-live-evaluate) format (`WavPath`, `Question`, `Answer`, `system_prompt`, `tool_definitions`, `conversationID`). Usable directly with the harness's built-in conversational evaluators. |
| `audio/` | Short-path WAV copies (smoke and regression). |
| `summary.json` | Counts by usage, lane, flow, expected tools, ASR label. |

Selection is deterministic (hash ordered) and stratified per usage and lane (`SUITES` in
[`prepare_datasets.py`](scripts/prepare_datasets.py)); about one third of each stratum comes from ASR-flagged
phrases so recognition regressions show up even in the small suite. Production-usage cases follow the traffic
mix of intents.

### 1.3 Check the retrieval ceiling (optional)

```powershell
python scripts/build_search_indexes.py --recreate
python scripts/oracle_search_check.py --suite regression
```

The oracle feeds gold entity mentions straight into the search, which tells you how much of the end-to-end gap
is retrieval versus speech or LLM.

---

## Step 2 - Register datasets in Foundry (optional)

```powershell
python scripts/register_datasets.py --suites smoke regression
```

Each suite folder becomes a folder dataset `sire-voice-<suite>` via `project.datasets.upload_folder`. The version
is a hash of `cases.jsonl` + `voicelive_harness.jsonl` (which embeds the prompt and tool definitions), so
re-running is a no-op until data or agent definition changes. The suite, its scorecards and its evaluators are
recorded under `evaluationSuites` in `agent/.foundry/agent-metadata.yaml`.

Registration gives traceability (which data version a run used) and makes the data available in the portal.
Runs themselves send eval items inline or as an uploaded JSONL file; audio is never uploaded to the
evaluation service.

---

## Step 3 - Create custom code evaluators

**Goal:** deterministic, gold-label metrics that run inside Foundry and produce exactly the same scores locally.

### 3.1 Write the grader

Add a function to [`grader_functions.py`](sire_eval/grader_functions.py). Rules (the file is executed in the
Foundry evaluation sandbox):

- Signature `grade_<metric>(sample, item) -> float` in `[0, 1]`. `item` is the eval item (Step 5 schema).
- **Standard library only**, no I/O, no relative imports (the only import is the guarded `textnorm` import).
- Parse defensively: arrays and objects may arrive as JSON strings (`_as_list`, `_as_dict`).
- Return a score for every input; code graders should never raise.

```python
def grade_intent_accuracy(sample, item):
    expected = (item.get("expected_intent") or "").strip().lower()
    predicted = (item.get("predicted_intent") or "").strip().lower()
    return 1.0 if expected and expected == predicted else 0.0
```

### 3.2 Declare it in the catalog

Add an entry to `EVALUATORS` in [`evaluators.py`](sire_eval/evaluators.py): catalog name, display name,
description, grader function, the item fields it reads (with JSON types) and the pass threshold.

```python
SireEvaluator("sire-intent-accuracy", "SIRE intent accuracy",
              "1 when the intent passed to the tools equals the gold intent label.",
              "grade_intent_accuracy", {"expected_intent": "string", "predicted_intent": "string"}, 0.5, "agents"),
```

### 3.3 Register

```powershell
python scripts/register_evaluators.py          # new version only when grading logic changed
python scripts/register_evaluators.py --force  # always create a new version
```

What `register_evaluators()` does for each entry:

1. Builds `code_text` = source of `textnorm.py` + `grader_functions.py` + an entry point
   `def grade(sample, item): return float(grade_<metric>(sample, item or {}))`.
2. Wraps it in a `CodeBasedEvaluatorDefinition` with a `data_schema` (the declared fields, all required), an
   `init_parameters` schema accepting `threshold`, and one continuous `result` metric in `[0, 1]` with the threshold.
3. Compares with the latest catalog version **ignoring comments and docstrings** (`logic_signature`) and only
   calls `project.evaluators.create_version(...)` when the logic changed.
4. Writes a local record to `agent/.foundry/evaluators/<name>-v<version>.json` and returns all versions
   (including the current built-in versions) for pinning.

The evaluator catalog APIs are preview: every call carries the header `Foundry-Features: Evaluations=V1Preview`
and the client is created with `allow_preview=True`. A minimal standalone equivalent:

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeBasedEvaluatorDefinition, EvaluatorMetric
from azure.identity import AzureCliCredential

HEADERS = {"Foundry-Features": "Evaluations=V1Preview"}
project = AIProjectClient(endpoint="<project endpoint>", credential=AzureCliCredential(), allow_preview=True)

code = '''
def grade(sample, item):
    expected = (item.get("expected_intent") or "").strip().lower()
    predicted = (item.get("predicted_intent") or "").strip().lower()
    return 1.0 if expected and expected == predicted else 0.0
'''
definition = CodeBasedEvaluatorDefinition(
    code_text=code,
    data_schema={"type": "object",
                 "properties": {"expected_intent": {"type": "string"}, "predicted_intent": {"type": "string"}},
                 "required": ["expected_intent", "predicted_intent"]},
    init_parameters={"type": "object", "properties": {"threshold": {"type": "number"}}, "required": []},
    metrics={"result": EvaluatorMetric(type="continuous", desirable_direction="increase",
                                       min_value=0.0, max_value=1.0, threshold=0.5)},
)
created = project.evaluators.create_version(
    name="my-intent-accuracy",
    evaluator_version={"name": "my-intent-accuracy", "evaluator_type": "custom", "categories": ["agents"],
                       "display_name": "Intent accuracy", "description": "Gold intent equals predicted intent.",
                       "definition": definition},
    headers=HEADERS,
)
print(created.version)
```

### 3.4 Test before you register

```powershell
python -m pytest tests
```

The tests exercise the graders, the normalizer and the bundled `code_text` exactly as Foundry will run it.
Then do a local-only smoke run (`--no-submit`, [Step 6](#step-6---produce-results-to-score)) and check the
local summary before submitting anything.

### The custom evaluators in this repository

| Evaluator | Reads | Threshold |
|---|---|---|
| `sire-stt-word-accuracy` | `gold_transcript`, `transcript` | 0.8 |
| `sire-stt-exact-match` | `gold_transcript`, `transcript` | 0.5 |
| `sire-stt-entity-recall` | `entity_mentions`, `transcript`, `lane` | 1.0 |
| `sire-intent-accuracy` | `expected_intent`, `predicted_intent` | 0.5 |
| `sire-tool-selection` | `expected_tools`, `called_tools` | 0.5 |
| `sire-entity-top1` | `expected_entities`, `predicted_entities` | 1.0 |
| `sire-entity-recall-at-5` | `expected_entities`, `candidate_ids` | 1.0 |
| `sire-no-confident-wrong` (safety) | `expected_entities`, `predicted_entities` | 0.5 |
| `sire-choice-accuracy` | `expected_choice`, `predicted_choice` | 0.5 |
| `sire-task-success` | flow, intents, entities, choices | 0.5 |
| `sire-latency-sla` | `latency_ms`, `latency_budget_ms` | 0.5 |

Definitions: [evaluation-design.md - Metrics](docs/evaluation-design.md#metrics).

---

## Step 4 - Add built-in evaluators

Built-in evaluators are referenced by name in a scorecard's testing criteria; `register_evaluators()` resolves
their current versions so they can be pinned like the custom ones.

| Criterion | Built-in | Data mapping | Initialization | Notes |
|---|---|---|---|---|
| `entity-retrieval-ndcg` | `builtin.document_retrieval` | `retrieval_ground_truth` <- gold URNs with `query_relevance_label: 1`; `retrieved_documents` <- merged search candidates with `relevance_score` | `ground_truth_label_min: 0`, `ground_truth_label_max: 1` | Reports NDCG, XDCG, fidelity and top-1 relevance. Scores 0 with no ground truth, which is one reason entity metrics live in their own scorecard. |
| `llm-intent-resolution` | `builtin.intent_resolution` | `query`, `response`, `tool_definitions` | `deployment_name: <judge>` | LLM judge. Does not know the gold answer. |
| `llm-tool-call-accuracy` | `builtin.tool_call_accuracy` | `query`, `tool_calls`, `tool_definitions`, `response` | `deployment_name: <judge>` | LLM judge. Marks rows *not applicable* (counted as errored) when no tool or an unknown tool was called. |

Formats the built-ins expect (produced by the runner, see `result_row` in
[`voice_runner.py`](sire_eval/voice_runner.py)):

```json
"tool_calls": [{"type": "tool_call", "tool_call_id": "call_0", "name": "search_user",
                "arguments": {"intent": "call", "name": "<name>"}}],
"tool_definitions": [{"name": "search_user", "description": "...", "parameters": {"type": "object", "properties": {}}}],
"retrieval_ground_truth": [{"document_id": "u-0001", "query_relevance_label": 1}],
"retrieved_documents": [{"document_id": "u-0001", "relevance_score": 0.92}]
```

Guidance:

- **Gate on the deterministic code evaluators; use the LLM judges as context.** In testing the tool-call judge gave
  5/5 to a call with the wrong intent because it looked plausible.
- LLM judges dominate cost and duration. Use `--no-llm-judges` for regression and full suites; those runs go to
  separate "(no LLM judges)" scorecards so pass rates stay comparable.
- For spoken-response quality, enable `--with-audio-output` and consider the preview audio evaluators
  (`builtin.task_completion_audio`, `builtin.coherence_audio`).

---

## Step 5 - Create the evaluation framework (scorecards)

**Goal:** one Foundry evaluation per capability, with a fixed item schema and pinned evaluator versions, so every
run added later is scored by exactly the same code and is comparable side by side.

### 5.1 Define the eval item schema

`ITEM_FIELDS` in [`foundry_evals.py`](sire_eval/foundry_evals.py) is the contract between your runs and every
evaluator. Every row must contain **every** field with a **non-null** value: missing keys become NaN in the
service and fail the whole run. `to_eval_item()` enforces this (type defaults, `query`/`response` placeholders,
sentinel retrieval documents for empty lists).

```python
ITEM_FIELDS = {
    "case_id": "string", "case_key": "string", "usage": "string", "lane": "string", "flow": "string", ...
    "gold_transcript": "string", "transcript": "string", "query": "string", "response": "string",
    "expected_intent": "string", "predicted_intent": "string",
    "expected_tools": "array", "called_tools": "array",
    "expected_entities": "array", "predicted_entities": "array", "candidate_ids": "array",
    "expected_choice": "object", "predicted_choice": "object",
    "tool_calls": "array", "tool_definitions": "array",
    "retrieval_ground_truth": "array", "retrieved_documents": "array",
    "latency_ms": "number", "latency_budget_ms": "number", ...
}
```

### 5.2 Define the scorecards

`EVAL_GROUPS` in [`evaluators.py`](sire_eval/evaluators.py) lists, per scorecard, its title, its custom
evaluators, its built-ins / LLM judges, and an `applies(row)` filter that decides which rows it receives.
Filtering keeps denominators honest (no "not applicable" rows inflating pass rates).

### 5.3 Build pinned testing criteria

`testing_criteria()` produces one criterion per evaluator, each with an explicit `evaluator_version`:

```python
{"type": "azure_ai_evaluator",
 "name": "intent-accuracy",
 "evaluator_name": "sire-intent-accuracy",
 "evaluator_version": "3",
 "data_mapping": {"expected_intent": "{{item.expected_intent}}", "predicted_intent": "{{item.predicted_intent}}"},
 "initialization_parameters": {"threshold": 0.5}}
```

Always pin versions: a criterion without `evaluator_version` resolves to the latest catalog version when each run
executes, so a grader fix would silently change how later runs in the same scorecard are scored.

### 5.4 Create the evaluation

Scorecards are created lazily on the first submission by `ensure_eval_group()`; you do not need a separate
command. It is equivalent to:

```python
client = project.get_openai_client()
evaluation = client.evals.create(
    name="SIRE voice - end-to-end",
    data_source_config={"type": "custom", "item_schema": item_schema(), "include_sample_schema": False},
    testing_criteria=testing_criteria(EVAL_GROUPS["e2e"], judge_deployment, llm_judges=True, versions=versions),
    metadata={"scorecard": "e2e", "llm_judges": "true", "graders": graders_fingerprint(versions)},
)
```

The evaluation id is cached under `evaluationGroups` in `agent/.foundry/agent-metadata.yaml` (and found by name
if the cache is missing).

### 5.5 Scorecards are frozen

The scorecard's metadata stores a fingerprint of its pinned evaluator versions. When grading logic changes and
`register_evaluators.py` creates new versions, submissions to the old scorecard are refused with an explanation.
Start a new series and re-baseline:

```powershell
python scripts/submit_run.py <baseline-run> --scorecard-suffix v2
python scripts/submit_run.py <candidate-run> --scorecard-suffix v2
```

Changing a scorecard's criteria likewise means a new scorecard name, never an edit of the existing one.

---

## Step 6 - Produce results to score

Every producer writes the same run folder, `agent/.foundry/results/<timestamp>-<suite>-<label>/`, containing
`results.jsonl` (one row per case: gold fields + predictions + diagnostics) and `run.json` (configuration, local
summary, per-lane breakdown). Any system that can emit this row format can be scored in the same scorecards.

### 6a. Baseline from existing production logs

```powershell
python scripts/submit_baseline_from_captures.py --suite smoke
```

Maps per-phrase production session captures (`<captures>/<usage>/<audio>.capture.json`: transcription, tool
call, search candidates and confidence, server timeline) onto the result schema. The current production system
becomes the first run in every scorecard at zero inference cost, which is the bar each candidate must clear.
Adapt the mapping functions in this script to your system's log format.

### 6b. Replay the corpus through the Foundry agent

```powershell
python scripts/run_voice_eval.py --suite smoke --label baseline              # run, score locally, submit
python scripts/run_voice_eval.py --suite smoke --label my-change --no-submit # local only
```

Per case the runner opens a realtime session with the case's structured inputs (dialog flow, choice options),
streams the WAV plus trailing silence (semantic VAD closes the turn), executes the tool calls client-side so
search candidates are captured exactly, and records transcript, tool calls, reply, tokens and latency
(end-of-speech to first tool call). A **preflight** session aborts the run if a requested override (for example a
per-session STT model) is not honored, so runs are never mislabelled.

Useful options: `--agent-name`, `--search-profile`, `--latency-budget-ms`, `--usage`, `--lane`, `--case`,
`--limit`, `--concurrency`, `--with-audio-output`, `--no-llm-judges`.

### 6c. Replay only the retrieval step

```powershell
python scripts/replay_search.py latest:smoke --search-profile sire-rrf-v2
```

Keeps transcript, intent and tool arguments from the source run and re-executes only the searches, so retrieval
changes are measured in seconds with speech and LLM held constant.

### From result row to eval item

| Eval item field | Source in the run |
|---|---|
| `transcript`, `query` | Service input transcription |
| `response` | Agent's text reply |
| `predicted_intent` | `intent` argument of the first tool call (`pick_choice` kinds map to intents) |
| `called_tools`, `tool_calls` | Every function call the agent made |
| `predicted_entities` | Top result of each search call (`id`, `confident`) |
| `candidate_ids`, `retrieved_documents` | All returned candidates (score / 100 as relevance) |
| `retrieval_ground_truth` | Gold URNs, relevance 1 |
| `predicted_choice` | `pick_choice` arguments |
| `latency_ms`, `latency_budget_ms` | First-tool-call latency and the run's budget |

---

## Step 7 - Submit runs and collect results

`run_voice_eval.py` and `submit_baseline_from_captures.py` submit automatically (unless `--no-submit`). To submit
or re-grade an existing folder:

```powershell
python scripts/submit_run.py latest:smoke                              # all applicable scorecards, waits
python scripts/submit_run.py latest:smoke --no-wait                    # submit and return
python scripts/collect_runs.py latest:smoke                            # wait / resume and download results
python scripts/submit_run.py latest:smoke --groups entities --replace  # re-grade one scorecard, drop old run
```

What happens:

1. Evaluators are (re)registered and versions resolved; each scorecard is found or created (Step 5).
2. Rows are filtered by each scorecard's `applies()`, projected with `to_eval_item()`, sorted in a fixed
   pseudo-random order (hash of `case_key`) and sent with `client.evals.runs.create`. Payloads above ~2 MB are
   uploaded as a JSONL file (`purpose="evals"`) and referenced by `file_id` (inline payloads hit HTTP 413).
3. The run name and metadata carry label, suite, agent name/version, LLM, STT model and search profile, so runs
   are self-describing in the portal.
4. Run ids are written to `<run>/foundry_runs.json` before waiting; an interrupted wait resumes with
   `collect_runs.py` instead of resubmitting duplicates.
5. If more than 2% of a run's code-grader results errored (transient `ProviderGradingError` under load), the run
   is resubmitted once, the cleaner run kept and the other deleted.
6. Per-item results are downloaded to `<run>/foundry_output_<scorecard>.json`, and `lastEval` is updated in
   `agent-metadata.yaml`.

Submit one configuration at a time: with many runs in flight, code graders lose a large share of their results.

---

## Step 8 - Compare runs and apply regression gates

**In the portal:** open **Evaluation**, pick a scorecard, select runs and choose **Compare**.

**Locally / in CI:**

```powershell
python scripts/compare_runs.py <baseline-run> <candidate-run> --markdown report.md   # exit code 1 on regression
python scripts/summarize_runs.py --suite smoke                                       # leaderboard of all runs
```

`compare_runs.py` re-scores both runs with the same graders, pairs them case by case, and for each metric in
[`agent/regression-gates.yaml`](agent/regression-gates.yaml) counts flipped cases (pass -> fail vs fail -> pass).
A gate fails only if the metric drops by more than `max_drop` **and** a one-sided exact McNemar test gives
`p <= alpha`:

```yaml
alpha: 0.05
metrics:
  task_success:       {max_drop: 0.02}
  entity_top1:        {max_drop: 0.02}
  no_confident_wrong: {max_drop: 0.0, alpha: 0.2}   # safety: more sensitive
  ...
latency:
  p95_ms: {max_increase_pct: 25}
```

- Drops beyond `max_drop` that are within noise are **warnings**: re-run or confirm on the regression suite.
- Significant drops within `max_drop` are flagged **watch**.
- Run an **A/A check** (same configuration twice) after changing models or the runner to calibrate the noise.

---

## Step 9 - Design the tests (suites and experiments)

### Suites

| Suite | Size | When |
|---|---|---|
| smoke | 120 cases, ~4 min + grading | Every change |
| regression | 1,200 cases, all intents and lanes | Before a release (use `--no-llm-judges`) |
| full | Whole corpus | Periodic coverage, model upgrades (use `--no-llm-judges`) |

### Test dimensions inside a suite

| Usage | What it stresses |
|---|---|
| Production mix | Real traffic distribution of intents, multi-entity requests |
| User lanes | How a person is named: full name, with department, "doctor", single name, spelled, unusual shapes |
| Group lanes | How a group is named: name, spoken form, spelled, member singular/plural, "the manager of ..." |
| Intent acceptance | Dialog states: main menu, login, partial call, choose, yes/no |
| Constrained choice | Picking from a list under audio stress: SNR 5/10/20 dB, speed, clipping, telephone band, voices, drift, prompt injection |

Every metric is reported per lane (`run.json`, `compare_runs.py`), so a regression in one way of speaking is
visible even when the average holds.

### Experiments: change one variable per run

| Change | How to test it |
|---|---|
| Prompt / tool descriptions | Edit `agent/instructions.md` or `agent/intents.yaml`, `python scripts/render_agent_yaml.py`, `azd deploy sire-voice`, then `run_voice_eval.py --label <name>`. |
| Voice LLM | `render_agent_yaml.py --candidate-model <model>`, `azd deploy sire-voice-candidate`, `run_voice_eval.py --agent-name <candidate> --label <name>`. |
| STT model | Fixed per agent version: render a candidate with `--candidate-transcription-model <model>`, deploy, run against it. |
| Search algorithm | Add a profile to `PROFILES` in [`search_client.py`](sire_eval/search_client.py) (never edit the baseline profile), then `replay_search.py <run> --search-profile <profile>`. |
| Index schema | Edit [`search_schema.py`](sire_eval/search_schema.py), rebuild indexes, `oracle_search_check.py`, then replay. |

Then `compare_runs.py <previous> <candidate>`. Keep prompt examples generic (`<name>`, `<group>`): names copied
from test cases inflate scores on those cases and realistic example names get reused by the model (the
`no-confident-wrong` gate caught exactly this in the demo).

---

## Adapting this to your own agent

1. **Corpus**: labelled utterances with gold transcript, intent/outcome, expected tools and expected retrieved
   entities (any-of sets for ambiguous ones). Adapt `build_case` in [`dataset.py`](sire_eval/dataset.py) and
   `SUITES` in [`prepare_datasets.py`](scripts/prepare_datasets.py).
2. **Metrics**: write `grade_*` functions (stdlib only), add `EVALUATORS` entries, add tests, run
   `python -m pytest tests`.
3. **Item schema**: add any new fields to `ITEM_FIELDS`; make your producer emit them (`to_eval_item` fills
   defaults).
4. **Scorecards**: group metrics by capability in `EVAL_GROUPS` with an `applies()` filter each; pick built-ins
   (retrieval, intent resolution, tool-call accuracy, or others from the catalog).
5. **Producer**: reuse [`voice_runner.py`](sire_eval/voice_runner.py) for a Foundry Voice Agent with
   client-executed tools, or write a converter that turns your system's logs into `results.jsonl` rows
   (pattern: [`submit_baseline_from_captures.py`](scripts/submit_baseline_from_captures.py)).
6. **Local metrics**: add the new metrics to `METRICS` in [`scoring.py`](sire_eval/scoring.py) and to
   [`regression-gates.yaml`](agent/regression-gates.yaml).
7. **Baseline first**: submit the current system as the first run in every scorecard, run an A/A check, then
   start experimenting.

---

## Running in CI

- Smoke suite on every pull request that touches the agent definition or evaluation code; regression suite before
  a release.
- Authenticate with workload identity (`azure/login`) and `AZURE_CREDENTIAL=cli`, or `AZURE_CREDENTIAL=default`.
- `python scripts/compare_runs.py <baseline> latest:smoke --markdown report.md` exits 1 on a failed gate;
  publish `report.md` as the job summary.
- The corpus is not in the repository: restore it from secure storage in the pipeline.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Whole run fails right after submission | A row is missing a field or has a null. Always go through `to_eval_item()`; add new fields to `ITEM_FIELDS`. |
| Evaluator catalog calls are rejected | Missing preview opt-in. Use `allow_preview=True` and the `Foundry-Features: Evaluations=V1Preview` header. |
| "Scorecard ... was created with evaluator versions ..." | Grader logic changed since the scorecard was created. Use `--scorecard-suffix <name>` and resubmit the baseline. |
| Many code-grader results *errored* | Transient `ProviderGradingError` under load. Submit one configuration at a time; `collect_runs.py` retries once above 2%. |
| `document_retrieval` scores 0 | No ground truth on the row. Keep retrieval metrics in a scorecard whose `applies()` only admits rows with gold entities. |
| Tool-call judge rows *errored* | Judge marked them not applicable (no tool call or unknown tool). Expected; read alongside `tool-selection`. |
| End-to-end scorecard takes very long | LLM judges limited by judge TPM. Increase capacity or use `--no-llm-judges`. |
| HTTP 413 on submission | Payload too large inline; `submit_run` switches to a file upload above ~2 MB. |
| STT override has no effect | Voice Agent transcription model is fixed per agent version; deploy a candidate agent instead. The preflight aborts such runs. |
| Interrupted wait | `python scripts/collect_runs.py <run>` resumes using `foundry_runs.json`. |
