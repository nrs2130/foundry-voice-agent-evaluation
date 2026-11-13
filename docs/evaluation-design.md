# Evaluation design

This document explains how the SIRE voice evaluation is put together, what each metric means, and how the
pieces map onto Microsoft Foundry. The [README](../README.md) covers setup and day-to-day commands.

## Principles

1. **Regression first.** The goal is to detect whether a change (LLM, STT model, prompt, search algorithm) makes
   the system better or worse on a fixed, labelled corpus, not to produce a single absolute quality number.
2. **Deterministic gates, judged context.** Release gates use deterministic, gold-label evaluators
   (WER, intent match, entity top-1, ...). LLM judges (intent resolution, tool-call accuracy) add context but vary
   run to run and do not know the gold answer - in testing, the tool-call judge gave 5/5 to a call with the wrong
   intent because the call was *plausible*.
3. **One scorecard per capability.** End-to-end, entity resolution and constrained choice are separate Foundry
   evaluations. Each row goes only to the scorecards whose metrics apply, so a pass rate is never inflated by
   "not applicable" rows (the Foundry evaluation service also rejects rows with missing fields, so mixing
   schemas in one evaluation is not an option).
4. **Same code locally and in the cloud.** The custom evaluators' code is the repository's
   `textnorm.py` + `grader_functions.py`; local summaries, regression gates and Foundry scores therefore agree
   exactly (verified on every run during development).
5. **Change one variable per run.** Agent versions (prompt), candidate agents (LLM, STT), and search profiles
   (search) are independent knobs; every run records the agent name/version, session model, transcription
   model and search profile in its Foundry run name and metadata.
6. **No test-set leakage.** Prompt and tool-description examples use placeholders (`<name>`, `<group>`), not
   names. Names copied from test cases inflate scores on exactly those cases, and realistic example names get
   reused by the model: with fictional example names (agent v3) the agent searched for the example's name on a
   spelled-name login and auto-confirmed the wrong user - caught by the `no-confident-wrong` gate. Fix failure
   *patterns* found by the evaluation (spelling preservation, "<name> in <unit>") with general rules, not with
   the failing names.

## Corpus and suites

The corpus (`tenant-gamma` regression set, 15,501 synthetic phrases, 15,485 with TTS audio) is organised by usage:

| Usage | What it stresses | Lanes |
|---|---|---|
| `01-production` | Real traffic mix over 69 intents (`call` dominant), multi-entity requests | one per intent |
| `02-user-lanes` | How a person is named | `name`, `dept` ("Zephyrine in Food Services"), `doctor`, `single`, `spelled`, `name_shape` |
| `03-group-lanes` | How a group is named | `name`, `spoken`, `spelled`, `member_singular`, `member_plural`, `joiner` ("the manager of ...") |
| `04-intent-acceptance` | Dialog flows | `main_menu`, `login`, `partialcall`, `choose`, `yesno` |
| `05-constrained-choice` | Picking from a list under audio stress | `clean`, `snr5/10/20`, `speed090-110`, `clip150`, `tel`, `voice1-3`, `drift` |

`scripts/prepare_datasets.py` builds three suites with deterministic, hash-ordered, stratified selection; roughly a
third of every stratum comes from phrases that the corpus ASR watchlist flags as misheard, so recognition
regressions are visible even in the small suite.

| Suite | Cases | Use |
|---|---|---|
| smoke | 120 (25 intents, 103 entity slots) | every change, a few minutes |
| regression | 1,200 (all 71 intents, 944 entity slots) | before a release |
| full | 15,485 | periodic coverage, model upgrades |

### Gold case model (`cases.jsonl`)

| Field | Meaning |
|---|---|
| `case_key` | Unique key (`<usage>-<lane>-<hash>`); `case_id` repeats across usages. |
| `flow` | Dialog state the badge is in: `MAIN_MENU`, `LOGIN`, `PARTIAL_CALL`, `YES_NO`, `CHOOSE`. Passed to the agent as a structured input. |
| `gold_transcript` | What was said (the phrase). |
| `expected_intent` | One of 71 intent labels. |
| `expected_entities` | One entry per slot that references the directory: `type` (user/group), `urns` (any is acceptable - some names are ambiguous by design), `names`. |
| `expected_tools` | `search_user` / `search_group` per entity type, `route_command` when no entity, `pick_choice` in the CHOOSE flow. |
| `expected_choice`, `choice_options` | CHOOSE flow only: `{kind: choice|none|exit, value}` and the options read to the user. |
| `entity_mentions` | Entity words as spoken (from the phrase metadata), used for STT entity recall. |
| `facility_ids`, `caller_department_ids` | Session context: the caller's site (+ global) and small departments, used by the search for scoping and disambiguation. |
| `asr_label` | Corpus ASR watchlist label (`clean`, `consistent-mismatch`, ...). |

Only user (`u-`) and group (`g-`) entities are scored; locations and facilities (under 1% of slots) are out of scope.

## The agent under test

`sire-voice-agent` is a Foundry **Voice Agent** (`kind: voice`, managed `gpt-realtime-mini`, `mai-transcribe`,
Azure semantic VAD - the production configuration) defined in `azure.yaml` and deployed with `azd`.
Its instructions are rendered from `agent/instructions.md` and the intent catalog `agent/intents.yaml`.

Tools (executed by the client, so search results are captured exactly):

| Tool | Arguments | Purpose |
|---|---|---|
| `search_user` | `intent`, `name`, `department`, `doctor` | Resolve a person |
| `search_group` | `intent`, `group` | Resolve a group, unit, role or room/bed group |
| `route_command` | `intent`, `details` | Requests that name no person or group |
| `pick_choice` | `kind`, `value` | CHOOSE flow only |

`{{flow}}` and `{{flow_guidance}}` are Voice Agent **structured inputs**: the runner passes the case's dialog state
(and, in the CHOOSE flow, the option list) when it connects, mirroring how the badge sets context in production.

## Run protocol

For every case `scripts/run_voice_eval.py`:

1. Opens a realtime session (`project.beta.voice_agents.realtime.connect`) with the case's structured inputs.
2. Optionally switches output to text only (skips TTS). A **preflight** session first verifies that the service
   acknowledges every requested override; transcription models, for example, are fixed per agent version and a
   per-session override is silently ignored - the preflight aborts instead of mislabelling the run.
3. Streams the WAV (resampled to 24 kHz PCM16) followed by trailing silence. The agent's semantic VAD closes the
   turn and starts the response, as in production; manual commit is only a fallback.
4. Executes function calls with `sire_eval/search_client.py` (multi-strategy RRF over Azure AI Search, scoped to
   the caller's facility) and returns compact results (`id`, `name`, `_match_score`, `_confident`) until the
   agent's final reply.
5. Records the input transcription, every tool call (arguments, timing, full candidate list), the reply, token
   usage, session/conversation IDs and latencies.

**Latency** is measured from the service's end-of-speech event (`speech_stopped`/`committed`) to the first
function call - the decision time, comparable to the production pipeline's end-of-speech-to-tool-call "brain"
time - and excludes the VAD silence window. Search latency is reported separately.

## Metrics

All custom metrics are in `[0, 1]` and pass at the listed threshold.

| Metric | Definition | Threshold |
|---|---|---|
| `stt-word-accuracy` | `1 - WER(normalize(gold), normalize(transcript))` | 0.8 |
| `stt-exact-match` | `compact(normalize(gold)) == compact(normalize(transcript))` | 0.5 (binary) |
| `stt-entity-recall` | Share of entity-mention tokens found in the transcript; a mention whose compact form appears in the compact transcript counts in full; other tokens may match a same-sounding token (Soundex) except in `spelled` lanes | 1.0 |
| `intent-accuracy` | Intent of the first tool call (`pick_choice` kinds map to choose / nointent / goodbye) equals gold | 0.5 |
| `tool-selection` | Set of tool names called equals `expected_tools` | 0.5 |
| `entity-top1` | Share of gold entity slots whose acceptable URNs include the top result of some search call | 1.0 |
| `entity-recall-at-5` | Share of gold entity slots with an acceptable URN among all returned candidates | 1.0 |
| `no-confident-wrong` | 0 if any search marked a non-gold top result `_confident` (the agent would auto-confirm the wrong person) | 0.5 |
| `choice-accuracy` | `pick_choice` kind matches and, for `choice`, the normalized value matches | 0.5 |
| `task-success` | CHOOSE: `choice-accuracy`; otherwise intent correct and every entity slot resolved at top 1 | 0.5 |
| `latency-sla` | `0 <= latency_ms <= latency_budget_ms` (default 1,500 ms, set per run) | 0.5 |
| `entity-retrieval-ndcg` | Built-in `document_retrieval` over the merged candidate list with gold URNs as relevant | built-in default |
| `llm-intent-resolution` | Built-in LLM judge on transcript + reply + tool definitions | built-in default |
| `llm-tool-call-accuracy` | Built-in LLM judge on transcript + tool calls + tool definitions | built-in default |

**Normalization** (`textnorm.py`) mirrors the rules used to label the corpus: lower-casing, number words to
digits ("seventy seven hundred and eighty one" -> `7781`, "eight six two" -> `862`), digit/letter splitting,
joined spelled letters ("o t t o" -> `otto`), "Dr." -> "doctor", contractions expanded, fillers and
punctuation removed, "12:00" == "12". As a self-check, the normalized gold phrase equals the normalized TTS script
for 98.8% of the corpus; the remainder are deliberate TTS pronunciation spellings ("nick you" for NICU).

## Mapping to Foundry

| Concept | Foundry object |
|---|---|
| Custom metric | Evaluator catalog entry, type **code** (`sire-*`), versioned; a new version is created only when the grading logic changes (comment and docstring edits are ignored). Thresholds are passed as `initialization_parameters.threshold`. |
| Built-in metric | `builtin.document_retrieval`, `builtin.intent_resolution`, `builtin.tool_call_accuracy` (judge deployment from `FOUNDRY_JUDGE_DEPLOYMENT`). |
| Scorecard | An evaluation with a custom `item_schema` (`ITEM_FIELDS`) and fixed testing criteria: "SIRE voice - end-to-end", "- entity resolution", "- constrained choice". Every criterion pins `evaluator_version`; the scorecard's metadata stores a fingerprint of those versions and submissions with different versions are refused. |
| Configuration under test | An evaluation **run** in each scorecard. Run name and metadata carry label, suite, agent name/version, LLM, STT model and search profile. Compare runs in the portal. |
| Test data | Folder datasets `sire-voice-smoke` / `sire-voice-regression` (harness JSONL + gold manifest + audio), version = hash of the manifest and the harness file (so a prompt change yields a new version). |
| Local cache | `agent/.foundry/` (Foundry workspace convention): `agent-metadata.yaml` (suites, scorecard IDs, last run), `datasets/`, `evaluators/`, `results/`. |

Implementation notes learned while building this:

- The evaluator catalog API needs the `Foundry-Features: Evaluations=V1Preview` header.
- Every row in a run must carry every field with a non-null value; absent keys become NaN in the service and
  fail the whole run. `to_eval_item()` enforces this, including sentinel documents for empty retrieval lists.
- `document_retrieval` with no ground truth scores 0, which is why entity metrics live in their own scorecard.
- Criteria that reference an evaluator without `evaluator_version` resolve to the latest catalog version when a run
  executes, so a grader fix would silently change how later runs in the same scorecard are scored. Pin versions.
- A per-session `session.update` of the transcription model is accepted but not applied by Voice Agents (the
  acknowledged session keeps the agent version's model). The runner's preflight checks the acknowledgement.
- Under load (many runs queued at once) code evaluators fail items with `ProviderGradingError`, mostly at the
  start of a run: with about 30 runs in flight, some scorecard runs lost 80-100% of their code-grader results.
  Submit one configuration at a time (the default: `run_voice_eval.py` and `submit_run.py` wait for their runs)
  rather than many `--no-wait` submissions. The code graders never legitimately error, so `submit_run.py` (when
  waiting) and `collect_runs.py` resubmit a scorecard run once when more than 2% of its code-grader results
  errored, keep the cleaner run and delete the other. Items are sent in a fixed pseudo-random order (hash of
  `case_key`), so errors that remain do not remove whole lanes from a pass rate.
- Waiting for grading can take tens of minutes, so `submit_run.py` records the new run ids (and, with
  `--replace`, the runs to delete) in `foundry_runs.json` before it waits. If the wait is interrupted,
  `collect_runs.py` resumes it instead of a resubmission leaving duplicate runs in the scorecards; replaced runs
  are deleted only once their replacement has completed.
- LLM-judge runs are bound by the judge deployment's tokens per minute: each row carries the full tool definitions,
  so a 120-row end-to-end run sends about 240 judge calls / 1.1M tokens - about 11 minutes at 100K TPM, much longer
  when runs share the deployment. Give the judge deployment enough capacity (this demo uses 500K TPM).
- Inline run payloads are size-limited (HTTP 413 above a few MB); larger runs are uploaded as a JSONL file
  (`purpose="evals"`) and referenced by `file_id`.

## Baselines

`scripts/submit_baseline_from_captures.py` converts the production pipeline's per-phrase session captures into the
run format: transcription, intent, tool call, the slot search's candidates and confidence, and the server timeline
(end of speech -> tool call). Production uses one `command` tool with a cross-type entity search; for tool selection
and the tool-call judge each slot is expressed as `search_user` or `search_group` according to what production's
search resolved. This makes the production system the first run in every scorecard at zero inference cost.

## Regression workflow

```text
change -> render/deploy (or new search profile) -> run_voice_eval.py / replay_search.py -> 3 Foundry runs
       -> compare_runs.py <baseline> <candidate>  (exit 1 if a gate in agent/regression-gates.yaml fails)
```

`agent/regression-gates.yaml` holds the allowed drop per metric (for example 2 points on task success and entity
top-1, any drop on `no-confident-wrong`) and the allowed latency growth.

**Gates account for run-to-run noise.** The realtime model is sampled, so the same agent version scores
differently on a second run: in an A/A check (agent v4 twice on the smoke suite) STT pass rates were identical
(only a few transcripts differed, e.g. "at 9 a.m." vs "at 9:00 AM") while agent metrics moved by up to 6 points
(choice accuracy: one case is 5.9 points on 17 cases). A threshold-only gate would fail that A/A comparison.
`compare_runs.py` therefore pairs the two runs case by case, counts the cases that flipped (pass -> fail vs
fail -> pass) and runs a one-sided exact McNemar (sign) test; a metric fails only if it drops beyond `max_drop`
**and** `p <= alpha` (0.05; 0.2 for the safety metric, so a handful of new confident-wrong cases already fails).
Drops beyond `max_drop` that are within noise are listed as warnings: re-run, or decide on the regression suite,
whose 1,200 cases give far more power. Drops within `max_drop` that are nevertheless significant are flagged
*watch* (on the regression suite, agent v4 lost 34 intent cases and gained 12: -1.8 points, p = 0.001).
Re-run an A/A check after changing models or the runner to re-calibrate.

In CI, run the smoke suite on every pull request that touches `agent/` or `sire_eval/`, and the regression suite
before a release; authenticate with workload identity (`azure/login` + `AZURE_CREDENTIAL=cli`) and publish the
Markdown report as a job summary.

## Extending

- **New metric**: add a `grade_*` function to `grader_functions.py`, an entry in `EVALUATORS`, attach it to a
  scorecard in `EVAL_GROUPS`, and (if it needs new data) a field in `ITEM_FIELDS` and `voice_runner.result_row`.
  Existing scorecards keep their criteria, so create the new scorecard version under a new name.
- **New search strategy**: add it in `SireSearchClient`, give it weight 0 in the baseline profile and a weight in
  a new profile, then `replay_search.py` against an existing run.
- **New lane or tenant**: point `SIRE_CORPUS_ROOT` at the new corpus, rebuild the indexes, re-run
  `prepare_datasets.py`; suites are defined in `SUITES` in that script.

## Limitations and next steps

- The audio is synthetic TTS; add real recorded utterances (with consent) to the corpus for production-like numbers.
- Runs are text-only by default; enable `--with-audio-output` and the preview audio evaluators
  (`builtin.task_completion_audio`, `builtin.coherence_audio`) to score spoken responses.
- Single-utterance sessions only; multi-turn disambiguation ("which Jennifer?") can be added with
  `conversationID`-grouped cases or Foundry's simulation-based voice evaluation.
- For production monitoring, enable conversation storage and tracing on the agent and use Foundry's trace-based
  evaluation with the same catalog evaluators.
- Findings to act on: the search marks some junk queries as confident (spelled letters passed unjoined, e.g.
  "X Y Z ...", matched an unrelated short name with `_confident`), which is what turns an LLM slip into a confident-wrong
  result - a candidate for a new search profile (`replay_search.py` measures it in seconds); and agent v4's small
  intent-accuracy dip (mostly requests where the model made no tool call) is the next prompt iteration's target.
