"""Oracle search check: feed gold entity mentions straight into the RRF search (no STT / LLM).

Measures the ceiling of the entity-resolution stage on its own:

    python scripts/oracle_search_check.py --suite regression --profile sire-rrf-v1
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.dataset import read_jsonl  # noqa: E402
from sire_eval.search_client import PROFILES, SireSearchClient  # noqa: E402
from sire_eval.settings import DATASETS_DIR  # noqa: E402

_DEPT = re.compile(r"^(?P<name>.+?)\s+(?:in|from)\s+(?P<dept>.+)$")


def oracle_args(mention: str, entity_type: str, phrase: str) -> dict:
    if entity_type == "group":
        return {"group": mention}
    m = _DEPT.match(mention)
    name, dept = (m.group("name"), m.group("dept")) if m else (mention, "")
    doctor = bool(re.search(r"\b(doctor|dr)\b", phrase.lower()))
    return {"name": name, "department": dept, "doctor": doctor}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="regression")
    parser.add_argument("--profile", default="sire-rrf-v1", choices=list(PROFILES))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = [c for c in read_jsonl(DATASETS_DIR / f"sire-{args.suite}" / "cases.jsonl") if c["expected_entities"]]
    if args.limit:
        cases = cases[: args.limit]
    stats = collections.defaultdict(lambda: collections.Counter())
    misses = []
    async with SireSearchClient(profile=args.profile) as client:
        sem = asyncio.Semaphore(6)

        async def run(case):
            mentions = list(case["entity_mentions"])
            for i, ent in enumerate(case["expected_entities"]):
                mention = mentions[i] if i < len(mentions) else ent["names"][0]
                q = oracle_args(mention, ent["type"], case["gold_transcript"])
                async with sem:
                    if ent["type"] == "user":
                        res = await client.search_user(facility_ids=case["facility_ids"],
                                                       caller_department_ids=case.get("caller_department_ids"), **q)
                    else:
                        res = await client.search_group(facility_ids=case["facility_ids"], **q)
                ids = [r["id"] for r in res["results"]]
                key = f"{case['usage'][:5]}/{case['lane']}"
                s = stats[key]
                s["n"] += 1
                s["top1"] += bool(ids and ids[0] in ent["urns"])
                s["top5"] += any(i in ent["urns"] for i in ids)
                conf = bool(res["results"] and res["results"][0]["_confident"])
                s["confident"] += conf
                s["confident_wrong"] += conf and not (ids and ids[0] in ent["urns"])
                s["latency"] += int(res["latency_ms"])
                if not (ids and ids[0] in ent["urns"]) and len(misses) < 25:
                    misses.append((key, q, ent["names"][0], [r["name"] for r in res["results"][:3]]))

        await asyncio.gather(*(run(c) for c in cases))

    total = collections.Counter()
    print(f"profile={args.profile}  suite={args.suite}")
    print(f"{'lane':24s} {'n':>5s} {'top1':>6s} {'top5':>6s} {'conf':>6s} {'conf_wrong':>10s} {'ms':>6s}")
    for key in sorted(stats):
        s = stats[key]
        total.update(s)
        print(f"{key:24s} {s['n']:5d} {s['top1']/s['n']:6.1%} {s['top5']/s['n']:6.1%} {s['confident']/s['n']:6.1%} "
              f"{s['confident_wrong']/s['n']:10.1%} {s['latency']/s['n']:6.0f}")
    n = total["n"]
    print(f"{'TOTAL':24s} {n:5d} {total['top1']/n:6.1%} {total['top5']/n:6.1%} {total['confident']/n:6.1%} "
          f"{total['confident_wrong']/n:10.1%} {total['latency']/n:6.0f}")
    print("\nSample misses (lane, query, gold, top3):")
    for m in misses:
        print("  ", m)


if __name__ == "__main__":
    asyncio.run(main())
