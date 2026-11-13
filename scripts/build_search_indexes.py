"""Create the tenant directory indexes in Azure AI Search and upload the entity lists.

    python scripts/build_search_indexes.py --recreate

Auth: Entra ID by default (needs Search Service Contributor + Search Index Data Contributor on
the search service). ``--use-admin-key`` instead fetches the admin key through the management
plane (Owner/Contributor) and keeps it in memory only.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sire_eval.search_schema import build_documents, group_index_definition, user_index_definition  # noqa: E402
from sire_eval.settings import get_credential, load_settings  # noqa: E402

API_VERSION = "2024-07-01"


def admin_key(settings, service_name: str) -> str:
    rg = os.getenv("AZURE_RESOURCE_GROUP")
    if not (settings.subscription_id and rg):
        raise SystemExit("--use-admin-key needs AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP in .env")
    token = get_credential(settings).get_token("https://management.azure.com/.default").token
    url = (f"https://management.azure.com/subscriptions/{settings.subscription_id}/resourceGroups/{rg}"
           f"/providers/Microsoft.Search/searchServices/{service_name}/listAdminKeys?api-version=2023-11-01")
    resp = httpx.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["primaryKey"]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the indexes if they exist.")
    parser.add_argument("--use-admin-key", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    settings = load_settings()
    settings.require("search_endpoint")
    base = settings.search_endpoint.rstrip("/")
    service_name = base.split("//", 1)[1].split(".", 1)[0]
    if args.use_admin_key:
        headers = {"api-key": admin_key(settings, service_name), "Content-Type": "application/json"}
    elif settings.search_api_key:
        headers = {"api-key": settings.search_api_key, "Content-Type": "application/json"}
    else:
        token = get_credential(settings).get_token("https://search.azure.com/.default").token
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    entity_dir = settings.corpus_root / "entity-lists"
    print(f"Building documents from {entity_dir} ...")
    started = time.perf_counter()
    user_docs, group_docs = build_documents(entity_dir)
    print(f"  {len(user_docs):,} users, {len(group_docs):,} groups in {time.perf_counter() - started:.1f}s")

    plan = [
        (user_index_definition(settings.search_user_index), user_docs),
        (group_index_definition(settings.search_group_index), group_docs),
    ]
    async with httpx.AsyncClient(timeout=120) as http:
        for definition, docs in plan:
            name = definition["name"]
            url = f"{base}/indexes/{name}?api-version={API_VERSION}"
            exists = (await http.get(url, headers=headers)).status_code == 200
            if exists and args.recreate:
                (await http.delete(url, headers=headers)).raise_for_status()
                print(f"Deleted index {name}")
                exists = False
            if not exists:
                resp = await http.put(url, headers=headers, json=definition)
                if resp.status_code >= 300:
                    raise SystemExit(f"Creating {name} failed ({resp.status_code}): {resp.text[:800]}")
                print(f"Created index {name}")

            sem = asyncio.Semaphore(args.parallel)
            done = 0

            async def upload(batch: list[dict]) -> None:
                nonlocal done
                payload = {"value": [{"@search.action": "mergeOrUpload", **d} for d in batch]}
                delay = 1.0
                for attempt in range(6):
                    async with sem:
                        resp = await http.post(f"{base}/indexes/{name}/docs/index?api-version={API_VERSION}", headers=headers, json=payload)
                    if resp.status_code in (200, 201):
                        failed = [r for r in resp.json()["value"] if not r.get("status")]
                        if failed:
                            raise RuntimeError(f"{len(failed)} documents failed in {name}: {failed[:3]}")
                        done += len(batch)
                        if done % (args.batch_size * 10) == 0 or done == len(docs):
                            print(f"  {name}: {done:,}/{len(docs):,}")
                        return
                    if resp.status_code in (207, 429, 503) and attempt < 5:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise RuntimeError(f"Upload to {name} failed ({resp.status_code}): {resp.text[:500]}")

            started = time.perf_counter()
            batches = [docs[i:i + args.batch_size] for i in range(0, len(docs), args.batch_size)]
            await asyncio.gather(*(upload(b) for b in batches))
            print(f"Uploaded {len(docs):,} docs to {name} in {time.perf_counter() - started:.1f}s")

        await asyncio.sleep(5)
        for definition, docs in plan:
            name = definition["name"]
            count = (await http.get(f"{base}/indexes/{name}/docs/$count?api-version={API_VERSION}", headers=headers)).text
            print(f"{name}: {count.strip()} documents indexed (expected {len(docs):,})")


if __name__ == "__main__":
    asyncio.run(main())
