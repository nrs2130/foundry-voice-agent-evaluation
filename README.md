# Foundry voice agent evaluation: SIRE regression demo

This repository shows how to set up repeatable, **regression-style evaluations for a Voice-RAG agent**
in Microsoft Foundry. The scenario is **SIRE**, a hands-free voice command router: a clinician speaks to a
badge, a real-time voice model separates the **intent** ("call", "page", "log in", ...) from the
**entities** (a person such as "Doctor Smith" or a group such as "NICU"), the entities are resolved against
a directory in **Azure AI Search** with a multi-strategy RRF search, and the agent confirms the result.

When the team changes **the LLM, the STT model, the prompt or the search algorithm**, one command replays a
labelled audio corpus through the agent and adds a new run to three Foundry evaluation scorecards, where it
can be compared side by side with the production baseline and earlier runs.

> **Building evaluations for your own agent?** [EVALUATIONS.md](EVALUATIONS.md) is a step-by-step guide to the
> evaluation part of this repository: data preparation, custom code evaluators, built-in evaluators, scorecards,
> runs, regression gates and test design.

## What gets evaluated

| Capability | Metric (evaluator) | How it is scored | Scorecard |
|---|---|---|---|
| Speech-to-text | `stt-word-accuracy` (custom code) | 1 - WER vs. the gold transcript after production-style normalization (numbers, spelled letters, fillers) | end-to-end, choice |
| Speech-to-text | `stt-exact-match` (custom code) | Normalized transcript equals gold (spacing-insensitive) | end-to-end |
| Speech-to-text on names | `stt-entity-recall` (custom code) | Share of the spoken entity words heard; same-sounding spellings allowed except in spelled lanes | entity resolution |
| Intent | `intent-accuracy` (custom code) | Intent passed to the tools equals the gold intent (71 labels) | end-to-end |
| Tool calling | `tool-selection` (custom code) | Set of tools called equals the expected set | end-to-end, choice |
| Tool calling | `llm-tool-call-accuracy` (built-in `tool_call_accuracy`, LLM judge) | Tool choice and arguments, judged from the transcript and tool definitions | end-to-end, choice |
| Entity search | `entity-top1`, `entity-recall-at-5` (custom code) | Gold entity URN is the top result / in the top 5 for each referenced person or group | entity resolution |
| Entity search | `entity-retrieval-ndcg` (built-in `document_retrieval`) | NDCG, XDCG, fidelity and top-1 relevance of the returned candidates | entity resolution |
| Safety | `no-confident-wrong` (custom code) | Fails when the search auto-confirms (`_confident`) the wrong person or group | entity resolution |
| Constrained choice | `choice-accuracy` (custom code) | `pick_choice` kind and value match (e.g. "call me back" -> `callback`, noisy audio, prompt injection) | constrained choice |
| End to end | `task-success` (custom code) | Right choice, or right intent **and** every entity resolved at top 1 | end-to-end |
| Conversation quality | `llm-intent-resolution` (built-in `intent_resolution`, LLM judge) | Whether the spoken reply resolves the request | end-to-end |
| Latency | `latency-sla` (custom code) | End of speech to first tool call within the run's budget (default 1,500 ms) | end-to-end |

The custom evaluators are registered in the project's **evaluator catalog** as versioned code-based
evaluators; their code is [`sire_eval/textnorm.py`](sire_eval/textnorm.py) plus
[`sire_eval/grader_functions.py`](sire_eval/grader_functions.py), which the local tooling also imports, so
local reports and Foundry scores always agree. Rows only go to the scorecards their metrics apply to (no
"not applicable" rows inflating pass rates). See [docs/evaluation-design.md](docs/evaluation-design.md).

## Architecture

```text
 labelled audio corpus (WAV + gold transcript, intent, entity URNs, choice, dialog flow)
        |  scripts/prepare_datasets.py  ->  smoke / regression / full suites
        |                                   (+ Voice Live evaluation-harness JSONL)
        v
 scripts/run_voice_eval.py  ---- realtime session per utterance ---->  Foundry Voice Agent "sire-voice-agent"
        ^      (structured inputs = dialog flow, optional STT override)  (kind: voice, gpt-realtime-mini,
        |                                                                  mai-transcribe, semantic VAD)
        |  function calls: search_user / search_group / route_command / pick_choice
        v
 sire_eval/search_client.py  -- multi-strategy RRF -->  Azure AI Search: sire-gamma-users, sire-gamma-groups
        |
        v
 agent/.foundry/results/<run>/results.jsonl  (transcript, tool calls, candidates, latency, tokens)
        |
        +--> Foundry Evaluations: 3 scorecards x N runs   <-- evaluator catalog (custom code + built-ins)
        +--> scripts/compare_runs.py: per-lane deltas + regression gates (CI exit code)
```

The production pipeline's own session captures are converted into the same result format
([`scripts/submit_baseline_from_captures.py`](scripts/submit_baseline_from_captures.py)), so the current
production system is the first run in every scorecard - the bar each candidate configuration is measured against.

## Repository layout

| Path | Purpose |
|---|---|
| [`azure.yaml`](azure.yaml) | azd project: the SIRE voice agent (`kind: voice`) and, optionally, a candidate agent. Generated by `scripts/render_agent_yaml.py`. |
| [`agent/`](agent) | Agent root: [`intents.yaml`](agent/intents.yaml) (71-intent catalog), [`instructions.md`](agent/instructions.md) (prompt template), [`regression-gates.yaml`](agent/regression-gates.yaml), and the `.foundry/` workspace (datasets, evaluators, results; git-ignored). |
| [`sire_eval/`](sire_eval) | Library: agent spec, dataset model, audio, search schema/client, voice runner, graders, evaluator catalog, Foundry submission, scoring. |
| [`scripts/`](scripts) | Command-line entry points (below). |
| [`docs/evaluation-design.md`](docs/evaluation-design.md) | Metric definitions, data formats, design decisions and how to extend. |
| [`EVALUATIONS.md`](EVALUATIONS.md) | Step-by-step guide to building the Foundry evaluations (data, evaluators, scorecards, runs, gates). |

## Prerequisites

- Python 3.11+, [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az login`), and the
  [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) with the
  Foundry agents extension (`azd extension install azure.ai.agents`).
- A Foundry project in a region that supports Voice Agents and evaluations (this demo used East US 2), with
  a chat model deployment for the LLM judges (default `gpt-5.4-mini`, set `FOUNDRY_JUDGE_DEPLOYMENT`).
- An Azure AI Search service (Basic tier is enough).
- Roles: **Foundry User** (Azure AI User) on the project; **Search Service Contributor** and
  **Search Index Data Contributor** on the search service (or pass `--use-admin-key` to the index builder).
- The regression corpus: extract `test_data/tenant-gamma-regression-2026-11-11.zip` and its inner zips into
  `test_data/extracted/` (step 1). The corpus is customer data and is git-ignored - never commit it.

## Quick start

```powershell
# 0. Environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # fill in the project endpoint, subscription, search endpoint

# 1. Extract the inner corpus zips (long paths: use Python rather than Expand-Archive on Windows)
python -c "import zipfile,glob,os; out='\\\\?\\'+os.path.abspath('test_data/extracted'); [zipfile.ZipFile(z).extractall(out) for z in glob.glob('test_data/tenant-gamma-regression-2026-11-11/tenant-gamma-regression-2026-11-11/*.zip')]"

# 2. Directory search: create the user/group indexes and upload the tenant entity lists
python scripts/build_search_indexes.py --recreate
python scripts/oracle_search_check.py --suite regression      # optional: search-only ceiling with gold mentions

# 3. Datasets: gold manifests + Voice Live harness JSONL for smoke / regression / full
python scripts/prepare_datasets.py
python scripts/register_datasets.py                           # optional: versioned Foundry datasets

# 4. Voice agent: render azure.yaml from agent/* and deploy with azd
python scripts/render_agent_yaml.py
azd deploy sire-voice

# 5. Evaluators: register the SIRE code evaluators in the project's evaluator catalog
python scripts/register_evaluators.py

# 6. Baseline: score the production pipeline from its captures (no audio replay needed)
python scripts/submit_baseline_from_captures.py --suite smoke

# 7. Run the agent on the smoke suite and add it to the scorecards
python scripts/run_voice_eval.py --suite smoke --label baseline

# Optional: run locally first, then submit (one configuration at a time - the evaluation service's code
# graders fail items under heavy concurrency) and collect. Run ids are recorded in foundry_runs.json as soon
# as they are submitted, so collect_runs.py also resumes an interrupted wait. --replace swaps out a folder's
# earlier Foundry runs once the new ones complete
python scripts/run_voice_eval.py --suite smoke --label my-change --no-submit
python scripts/submit_run.py latest:smoke --no-wait
python scripts/collect_runs.py latest:smoke
python scripts/submit_run.py latest:smoke --groups entities --replace   # re-grade one scorecard
```

Before the first `azd deploy`, bind an azd environment to your existing project: in a scratch folder run
`azd ai agent init --no-prompt --kind prompt-voice --agent-name sire-voice-agent --project-id <project ARM id>`,
copy the generated `sire-voice-agent/.azure/` folder to the repo root, then run
`azd env set AZURE_AI_PROJECT_ENDPOINT <project endpoint>` in the repo (azure.yaml reads the endpoint from the azd
environment, so no environment details are committed).

## Regression playbook

Each experiment changes one variable and becomes a new run in the same three scorecards. In the Foundry
portal open **Evaluation**, pick a scorecard, select runs and choose **Compare**; or compare locally with gates:

```powershell
python scripts/compare_runs.py <baseline-run> <candidate-run> --markdown report.md   # exit code 1 on regression
python scripts/summarize_runs.py --suite smoke                                       # leaderboard of all runs
```

A gate fails only when a metric drops beyond its tolerance in
[`agent/regression-gates.yaml`](agent/regression-gates.yaml) **and** the drop is statistically significant on the
paired cases (exact McNemar test on the cases that flipped), because two runs of the same realtime agent differ by
a few cases. Drops within noise are reported as warnings; see
[Regression workflow](docs/evaluation-design.md#regression-workflow).

| Change | How to test it |
|---|---|
| **Prompt / tool descriptions** | Edit `agent/instructions.md` or `agent/intents.yaml`, `python scripts/render_agent_yaml.py`, `azd deploy sire-voice` (new agent version), then `run_voice_eval.py --label <name>`. |
| **LLM** (voice model) | `python scripts/render_agent_yaml.py --candidate-model gpt-realtime` adds service `sire-voice-candidate`; `azd deploy sire-voice-candidate`; `run_voice_eval.py --agent-name sire-voice-agent-candidate --label <name>`. Use `--candidate-model-type self_deployed` with a deployment name for your own model deployments. |
| **STT model** | The transcription model is fixed per agent version (the runner's preflight aborts if a per-session `--stt-model` override is not honored). Render a candidate with `--candidate-name sire-voice-agent-stt-4o --candidate-transcription-model gpt-4o-transcribe`, `azd deploy sire-voice-candidate`, then `run_voice_eval.py --agent-name sire-voice-agent-stt-4o --label <name>`. |
| **Search algorithm** | Add or edit a profile in `PROFILES` in [`sire_eval/search_client.py`](sire_eval/search_client.py), then `python scripts/replay_search.py <run> --search-profile <profile>`: re-executes only the searches of an existing run (speech and LLM held constant) in seconds. Confirm with a full `run_voice_eval.py --search-profile <profile>`. |
| **Index schema / analyzers** | Edit [`sire_eval/search_schema.py`](sire_eval/search_schema.py), rebuild the indexes, run `oracle_search_check.py`, then replay. |

Suites: **smoke** (120 cases, about 4 minutes) for every change, **regression** (1,200 cases across all 71
intents and lanes) before a release, **full** (15,485 cases) for periodic coverage. The two LLM judges make the
end-to-end scorecard slow (about 1.1M judge tokens per 120 rows - 11+ minutes at 100K TPM; size the judge
deployment accordingly), so run regression and full suites with `--no-llm-judges`; those runs go to separate
"(no LLM judges)" scorecards.

**Scorecards are frozen.** Each scorecard pins the evaluator versions (custom and built-in) it was created with.
When grading logic changes (comment and docstring edits do not count), `register_evaluators.py` creates new
catalog versions and submissions to the old scorecards stop with an explanation; start a new series with
`--scorecard-suffix <name>` and resubmit the baseline into it, so every run in a scorecard is always scored by
the same code.

## Demo results

What the scorecards showed in the demo environment (synthetic TTS audio; local scores, which match the Foundry
evaluators). Smoke suite, 120 cases; the gate column compares each candidate with the run it changes.

| Run | Variable | Task success | Intent | Entity top-1 | No confident-wrong | STT word acc. | Latency p50 / p95 | Gate |
|---|---|---|---|---|---|---|---|---|
| Production pipeline (captures) | baseline | 91.5% | 98.3% | 90.9% | 98.9% | 83.9% | 570 / 1115 ms | - |
| Agent v1: initial prompt | prompt | 57.5% | 87.5% | 57.3% | 92.1% | 82.5% | 572 / 813 ms | - |
| Agent v2: tool-use rules | prompt | 63.3% | 87.5% | 66.3% | 96.6% | 84.2% | 566 / 704 ms | PASS vs v1 |
| v2 + search profile `sire-rrf-v2` | search | 64.2% | 87.5% | 67.4% | 96.6% | 84.2% | 566 / 704 ms | PASS vs v2 |
| v2 with LLM `gpt-realtime` | LLM | 71.7% | 91.7% | 73.0% | 96.6% | 81.7% | 739 / 1694 ms | **FAIL**: latency p95 +141% |
| v2 with STT `gpt-4o-transcribe` | STT | 65.0% | 89.2% | 66.3% | 94.4% | 76.7% | 523 / 693 ms | **FAIL**: STT word -7.5, STT entity -10.1 pts |
| Agent v3: fictional example names | prompt | 67.5% | 88.3% | 70.8% | 91.0% | 82.5% | 561 / 756 ms | **FAIL**: 6 new confident-wrong (p=0.06) |
| Agent v4: placeholder examples | prompt | 61.7% | 86.7% | 65.2% | 95.5% | 82.5% | 555 / 706 ms | PASS vs v2 (drops within noise) |
| Agent v4 again (A/A check) | none | 64.2% | 87.5% | 66.3% | 94.4% | 82.5% | 496 / 682 ms | PASS vs v4 |

Regression suite (1,200 cases): production 90.2% task success / 89.3% entity top-1; agent v2 64.8% / 68.0%;
agent v4 63.6% / 67.5% - v2 -> v4 passes, with intent accuracy flagged *watch* (-1.8 points, consistent across
46 flipped cases).

What this demonstrates:

- **A bigger voice model is not automatically better for this workload.** `gpt-realtime` lifts task success by
  8 points but more than doubles tail latency; the gate blocks it until latency is addressed.
- **STT changes are isolated by the STT metrics.** The speech-to-speech model acts on the audio, so swapping
  the transcription model barely moves intent or entity metrics, while transcript and name accuracy drop.
- **Prompt hygiene matters and the safety gate catches it.** Agent v2's prompt contained names copied from test
  cases. Replacing them with fictional names (v3) made the model reuse an example name in a login search, which
  auto-confirmed the wrong user; placeholders (v4) removed the leak without a significant regression.
- **Noise-aware gates.** Two runs of the same agent differ by a few cases (A/A row); gates use paired
  significance tests so that noise does not block changes and real regressions are not waved through.
- **The prototype agent is a stand-in.** The production pipeline (same realtime model, its own prompt,
  `mai-transcribe-2`, a single command tool and a cross-type entity search) leads by about 26 points on task
  success; the harness quantifies exactly where (tool selection, entity resolution per lane) - see
  `agent/.foundry/results/reports/` after a run.

## Data formats prepared for Foundry

`scripts/prepare_datasets.py` writes, per suite, to `agent/.foundry/datasets/sire-<suite>/`:

- **`voicelive_harness.jsonl`**: the format of the
  [Voice Live evaluation harness](https://learn.microsoft.com/azure/ai-services/speech-service/how-to-voice-live-evaluate):
  `WavPath` (relative to the file), `Question` (gold transcript), `Answer` (expected outcome: intent and resolved
  entity names/IDs, or the expected choice), `system_prompt` (the agent prompt with the case's dialog flow
  filled in), `tool_definitions`, and `conversationID`, plus the SIRE gold fields. It can be fed to the
  harness directly to score its built-in conversational evaluators.
- **`cases.jsonl`**: the gold manifest for the runner: transcript, intent, entity slots (each with the set of
  acceptable URNs), expected tools, expected choice, dialog flow, facility, caller context, ASR watchlist label.
- **`audio/`**: short-path copies of the suite's WAVs (smoke and regression).

Every run sends one eval item per case with a fixed schema (`ITEM_FIELDS` in
[`sire_eval/foundry_evals.py`](sire_eval/foundry_evals.py)); audio is never uploaded to the evaluation service.

## Notes and limitations

- **Turn-taking**: each utterance is streamed into its own realtime session with trailing silence; the agent's
  semantic VAD closes the turn as in production. Latency is measured from the service's end-of-speech event to
  the first tool call (comparable to the production pipeline's "brain" time) and excludes the VAD silence window.
- **Output**: runs are text-only by default (`--with-audio-output` keeps TTS); speech synthesis quality is not scored.
- **Speech-to-speech models** act on the audio itself, so an STT change mainly moves the STT metrics; with a
  cascaded model (STT -> text LLM) it also moves intent and entity metrics.
- **LLM judges** vary run to run; gate releases on the deterministic code evaluators and use the judges as
  supporting signal. The tool-call judge marks rows *not applicable* when the agent made no tool call or called a
  tool that is not in its definitions; Foundry counts those as *errored*, separately from pass/fail. Transient
  service-side failures of the code evaluators are retried automatically (see
  [docs/evaluation-design.md](docs/evaluation-design.md#mapping-to-foundry)). The audio is synthetic TTS, so
  absolute numbers are an upper bound for real speech.
- **Production baseline**: production uses a single `command` tool with a cross-type entity search; its calls are
  expressed in this demo's tool vocabulary according to what its search resolved for each slot.
