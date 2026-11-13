"""Run a SIRE evaluation suite through the Foundry Voice Agent and score it in Foundry.

    # baseline: the deployed agent as configured in azure.yaml
    python scripts/run_voice_eval.py --suite smoke --label baseline

    # STT or LLM experiment: a candidate agent rendered with a different transcription / voice model
    python scripts/render_agent_yaml.py --candidate-name sire-voice-agent-stt-4o --candidate-transcription-model gpt-4o-transcribe
    azd deploy sire-voice-candidate
    python scripts/run_voice_eval.py --suite smoke --label stt-gpt-4o-transcribe --agent-name sire-voice-agent-stt-4o

    # search experiment: same agent, different client-side search profile
    python scripts/run_voice_eval.py --suite smoke --label search-v2 --search-profile sire-rrf-v2

Results land in agent/.foundry/results/<run>/ (results.jsonl, run.json) and, unless --no-submit,
one run per scorecard is added to the Foundry evaluations "SIRE voice - end-to-end",
"SIRE voice - entity resolution" and "SIRE voice - constrained choice" so runs can be compared
side by side in the portal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.dataset import read_jsonl  # noqa: E402
from sire_eval.runs import new_run_dir, print_local_summary, save_run, submit_run_dir  # noqa: E402
from sire_eval.search_client import PROFILES, SireSearchClient  # noqa: E402
from sire_eval.settings import DATASETS_DIR, get_async_credential, load_settings  # noqa: E402
from sire_eval.voice_runner import RunConfig, preflight, resolve_audio, run_cases  # noqa: E402


def agent_info(settings, agent_name: str) -> dict:
    from sire_eval.settings import project_client

    agent = project_client(settings).agents.get(agent_name=agent_name).as_dict()
    latest = (agent.get("versions") or {}).get("latest") or {}
    definition = latest.get("definition") or {}
    transcription = ((definition.get("audio") or {}).get("input") or {}).get("transcription") or {}
    return {"agent_name": agent_name, "agent_version": str(latest.get("version", "")),
            "model": definition.get("model", ""), "model_type": definition.get("model_type", ""),
            "agent_stt_model": transcription.get("model", "")}


def select_cases(cases: list[dict], args) -> list[dict]:
    if args.usage:
        cases = [c for c in cases if c["usage"].startswith(args.usage)]
    if args.lane:
        cases = [c for c in cases if c["lane"] == args.lane]
    if args.case:
        cases = [c for c in cases if c["case_key"] in args.case or c["case_id"] in args.case]
    if args.limit:
        cases = cases[: args.limit]
    return cases


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="smoke", help="Dataset suite under agent/.foundry/datasets/sire-<suite>.")
    parser.add_argument("--label", required=True, help="Short run label shown in Foundry (e.g. baseline, stt-4o).")
    parser.add_argument("--agent-name", default=None, help="Voice agent name (default: SIRE_AGENT_NAME).")
    parser.add_argument("--stt-model", default=None,
                        help="Per-session transcription override; the run aborts if the agent does not honor it "
                             "(deploy a candidate agent with --candidate-transcription-model instead).")
    parser.add_argument("--search-profile", default="sire-rrf-v1", choices=list(PROFILES))
    parser.add_argument("--with-audio-output", action="store_true", help="Keep TTS output (slower, costs more).")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--latency-budget-ms", type=int, default=1500)
    parser.add_argument("--usage", default=None, help="Filter by usage prefix, e.g. 02-user-lanes.")
    parser.add_argument("--lane", default=None)
    parser.add_argument("--case", nargs="*", default=None, help="Specific case_key(s) or case_id(s).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-submit", action="store_true", help="Run and score locally only.")
    parser.add_argument("--no-llm-judges", action="store_true",
                        help="Skip the LLM-judge evaluators (uses separate 'no LLM judges' scorecards).")
    args = parser.parse_args()

    settings = load_settings()
    settings.require("project_endpoint", "search_endpoint")
    agent_name = args.agent_name or settings.agent_name
    manifest = DATASETS_DIR / f"sire-{args.suite}" / "cases.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found. Run scripts/prepare_datasets.py first.")
    cases = select_cases(read_jsonl(manifest), args)
    resolve_audio(cases, manifest)

    info = {"label": args.label, "suite": args.suite, "source": "voice-agent-replay", **agent_info(settings, agent_name),
            "stt_model": args.stt_model or "", "search_profile": args.search_profile,
            "text_only": not args.with_audio_output, "latency_budget_ms": args.latency_budget_ms}
    if not info["stt_model"]:
        info["stt_model"] = info["agent_stt_model"]
    print(f"{len(cases)} cases | agent {agent_name} v{info['agent_version']} "
          f"({info['model']}) | stt={info['stt_model']} | search={args.search_profile}")

    cfg = RunConfig(agent_name=agent_name, run_label=args.label, stt_model=args.stt_model,
                    text_only=not args.with_audio_output, concurrency=args.concurrency,
                    latency_budget_ms=args.latency_budget_ms)
    started = time.perf_counter()
    done = 0

    def progress(_i, row):
        nonlocal done
        done += 1
        if done % 10 == 0 or done == len(cases) or row.get("status") == "error":
            flag = " ERROR " + "; ".join(row.get("errors", []))[:160] if row.get("status") == "error" else ""
            print(f"  [{done}/{len(cases)}] {time.perf_counter() - started:6.1f}s{flag}")

    from azure.ai.projects.aio import AIProjectClient

    credential = get_async_credential(settings)
    async with credential, AIProjectClient(endpoint=settings.project_endpoint, credential=credential, allow_preview=True) as project:
        check = await preflight(project, cfg, cases[0])
        if check["problems"]:
            raise SystemExit("Preflight failed:\n  - " + "\n  - ".join(check["problems"]))
        info["session_model"] = check["model"]
        info["session_transcription_model"] = check["transcription_model"]
        run_dir = new_run_dir(args.suite, args.label)
        print(f"Run {run_dir.name}: preflight ok (session model={check['model']}, stt={check['transcription_model']})")
        async with SireSearchClient(settings, profile=args.search_profile) as search:
            await search.warmup()
            rows = await run_cases(project, cases, cfg, search, on_result=progress)
    info["duration_s"] = round(time.perf_counter() - started, 1)
    save_run(run_dir, rows, info)
    print_local_summary(rows, f"\nLocal scores ({run_dir.name}):")

    if not args.no_submit:
        submit_run_dir(run_dir, llm_judges=not args.no_llm_judges)
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
