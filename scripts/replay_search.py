"""Re-run only the directory search of an existing run with a different search profile.

    python scripts/replay_search.py latest:smoke --search-profile sire-rrf-v2

The transcript, intent and tool-call arguments are kept from the source run (speech and LLM
behaviour held constant); every search_user / search_group call is re-executed with the new
profile and the entity-resolution fields are recomputed. Search-algorithm changes can then be
regression tested in seconds, without replaying audio, and compared in the same Foundry scorecards.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.agent_spec import SEARCH_TOOLS  # noqa: E402
from sire_eval.runs import load_run, new_run_dir, print_local_summary, resolve_run_dir, save_run, submit_run_dir  # noqa: E402
from sire_eval.search_client import PROFILES, SireSearchClient, tool_payload  # noqa: E402
from sire_eval.settings import load_settings  # noqa: E402


async def replay_row(row: dict, search: SireSearchClient, sem: asyncio.Semaphore, label: str) -> dict:
    predicted, candidates, retrieved, tool_results, search_ms = [], [], {}, [], 0.0
    for idx, call in enumerate(row.get("tool_calls") or []):
        if call.get("name") not in SEARCH_TOOLS:
            continue
        args = call.get("arguments") or {}
        async with sem:
            if call["name"] == "search_user":
                res = await search.search_user(name=str(args.get("name") or ""), department=str(args.get("department") or ""),
                                               doctor=bool(args.get("doctor")), facility_ids=row.get("facility_ids"),
                                               caller_department_ids=row.get("caller_department_ids"))
            else:
                res = await search.search_group(group=str(args.get("group") or ""), facility_ids=row.get("facility_ids"))
        search_ms += res["latency_ms"]
        results = res["results"]
        tool_results.append({"call_id": call.get("tool_call_id"), "name": call["name"], "arguments": args,
                             "output": tool_payload(res), "search": {k: v for k, v in res.items() if k != "strategies"}})
        if results:
            top = results[0]
            predicted.append({"id": top["id"], "name": top["name"], "call_index": idx,
                              "type": "user" if call["name"] == "search_user" else "group",
                              "score": top["_match_score"], "confident": bool(top["_confident"])})
        for r in results:
            candidates.append(r["id"])
            retrieved[r["id"]] = max(retrieved.get(r["id"], 0.0), r["_match_score"] / 100.0)
    return {**row, "run_label": label, "predicted_entities": predicted, "candidate_ids": candidates,
            "retrieved_documents": [{"document_id": k, "relevance_score": round(v, 4)} for k, v in sorted(retrieved.items(), key=lambda kv: -kv[1])],
            "tool_results": tool_results or row.get("tool_results", []), "search_ms": round(search_ms, 1)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Source run folder, its name, 'latest' or 'latest:<suite>'.")
    parser.add_argument("--search-profile", required=True, choices=list(PROFILES))
    parser.add_argument("--label", default=None)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--no-llm-judges", action="store_true")
    args = parser.parse_args()

    source = resolve_run_dir(args.run)
    rows, info = load_run(source)
    label = args.label or f"{info.get('label', source.name)} + search {args.search_profile}"
    started = time.perf_counter()
    async with SireSearchClient(load_settings(), profile=args.search_profile) as search:
        await search.warmup()
        sem = asyncio.Semaphore(6)
        new_rows = await asyncio.gather(*(replay_row(r, search, sem, label) for r in rows))
    run_dir = new_run_dir(info.get("suite", "replay"), label)
    new_info = {**{k: v for k, v in info.items() if k not in ("summary", "by_lane")}, "label": label,
                "search_profile": args.search_profile, "source": f"search-replay of {source.name}",
                "duration_s": round(time.perf_counter() - started, 1)}
    save_run(run_dir, list(new_rows), new_info)
    print_local_summary(list(new_rows), f"Local scores ({run_dir.name}):")
    if not args.no_submit:
        submit_run_dir(run_dir, llm_judges=not args.no_llm_judges)
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
