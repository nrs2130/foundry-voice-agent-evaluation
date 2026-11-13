# AGENTS.md

Guidance for coding agents working in this repository.

This project was built with the microsoft-foundry skill. Before working on or answering questions about foundry
agents, read the microsoft-foundry skill first. If you are in VS Code, read the vscode-microsoft-foundry skill first.

## What this repository is

A regression-evaluation harness for a Foundry **Voice Agent** (`sire-voice-agent`, `kind: voice`) that routes
spoken clinician commands: intent detection plus person/group resolution against Azure AI Search. See
[README.md](README.md) for the workflow, [docs/evaluation-design.md](docs/evaluation-design.md) for metric
definitions and [EVALUATIONS.md](EVALUATIONS.md) for the step-by-step evaluation build guide (keep it in sync when
evaluation scripts, `ITEM_FIELDS`, `EVALUATORS` or `EVAL_GROUPS` change).

## Map

- `azure.yaml` is generated: edit `agent/instructions.md`, `agent/intents.yaml` or `sire_eval/agent_spec.py`, then
  run `python scripts/render_agent_yaml.py` and `azd deploy sire-voice`. Always set
  `AZURE_DEV_USER_AGENT=microsoft_foundry_skill` inline for azd commands.
- `sire_eval/textnorm.py` and `sire_eval/grader_functions.py` are shipped verbatim as Foundry code evaluators:
  keep them standard-library only, with no I/O and no relative imports beyond the guarded one in `grader_functions.py`.
  Changing their logic creates new evaluator versions on the next `scripts/register_evaluators.py` (comment and
  docstring edits are ignored).
- Scorecards (`EVAL_GROUPS` in `sire_eval/evaluators.py`) have fixed testing criteria; changing criteria means a
  new scorecard, otherwise runs stop being comparable.
- Rows sent to Foundry must contain every field in `ITEM_FIELDS` (`sire_eval/foundry_evals.py`) with non-null
  values; use `to_eval_item()`.
- Search behaviour is controlled by `PROFILES` in `sire_eval/search_client.py`; add a profile instead of changing
  the baseline `sire-rrf-v1` so existing runs stay reproducible.

## Conventions

- Never commit `test_data/`, `agent/.foundry/datasets/`, `agent/.foundry/results/`, `.env` or `.azure/` (customer
  data and environment identifiers; the repository is public). Do not copy names, URNs or phrases from the corpus
  into code, docs, tests or prompts: use placeholders (`<name>`) in prompts and fictional names elsewhere (prompt
  examples taken from test cases also leak the test set into the agent).
- Prompt changes are judged with `scripts/compare_runs.py` against the current agent's run; its gates separate real
  regressions from run-to-run noise, so read the "within noise" warnings before re-running.
- Configuration comes from `.env` (see `.env.example`); credentials are Entra ID via `AzureCliCredential` scoped to
  `AZURE_SUBSCRIPTION_ID` (`AZURE_CREDENTIAL=default` switches to `DefaultAzureCredential`).
- Validate grader or normalizer changes with `python -m pytest tests`, then a smoke run with `--no-submit`.
