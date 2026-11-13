"""Multi-strategy Azure AI Search client with weighted Reciprocal Rank Fusion (RRF).

Port of the SIRE_demo search pipeline (exact, fuzzy, field-targeted, Double Metaphone /
Soundex / Beider-Morse phonetic and number-normalized strategies fused with weighted RRF,
K=10, plus the "confident" auto-confirm rule), extended for the tenant directory:
facility scoping, department qualifiers, doctor titles, spelled-out (compact) names,
group member-role names and English-stemmed group fields.

Every knob lives in a ``SearchProfile`` so search-algorithm changes can be regression
tested exactly like model changes (see ``scripts/replay_search.py``).
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from .settings import Settings, get_credential, load_settings
from .textnorm import compact, normalize_text


@dataclass(frozen=True)
class SearchProfile:
    name: str
    rrf_k: int = 10
    top: int = 5
    per_strategy_top: int = 15
    confident_min_score: float = 70.0
    confident_min_gap: float = 25.0
    weights: dict[str, float] = field(default_factory=dict)
    disabled: frozenset[str] = frozenset()

    def weight(self, label: str) -> float:
        return 0.0 if label in self.disabled else self.weights.get(label, 0.0)


_BASE_WEIGHTS = {
    # people
    "user-exact": 1.0, "user-fuzzy": 0.7, "user-targeted": 2.0, "user-targeted-fuzzy": 1.6,
    "user-targeted-phonetic": 1.2, "user-phonetic-dm": 0.5, "user-phonetic-soundex": 0.4,
    "user-phonetic-bm": 0.35, "user-compact": 2.0, "user-department": 3.0, "user-department-all": 2.0,
    "user-department-phonetic": 1.0, "user-doctor": 1.0, "user-doctor-phonetic": 0.5,
    "user-caller-department": 2.0, "user-caller-department-phonetic": 1.0,
    # groups
    "group-exact-all": 1.0, "group-exact-any": 0.5, "group-fuzzy": 0.7, "group-normalised": 0.85,
    "group-normalised-any": 0.51, "group-fuzzy-norm": 0.63, "group-norm-fields": 0.9,
    "group-compact": 2.0, "group-stemmed": 0.8, "group-role-rewrite": 1.0, "group-role-rewrite-exact": 2.0,
    # Off in v1; v2 prefers groups whose member role name ("lactation nurse") was spoken.
    "group-member-exact": 0.0,
}
_PHONETIC = frozenset({"user-phonetic-dm", "user-phonetic-soundex", "user-phonetic-bm", "user-targeted-phonetic",
                       "user-department-phonetic", "user-doctor-phonetic", "user-caller-department-phonetic"})
_CONTEXT = frozenset({"user-caller-department", "user-caller-department-phonetic"})

PROFILES: dict[str, SearchProfile] = {
    # SIRE_demo defaults (K=10, weights, 70/25 confidence rule) plus tenant-specific strategies.
    "sire-rrf-v1": SearchProfile(name="sire-rrf-v1", weights=dict(_BASE_WEIGHTS)),
    # Textbook RRF constant: flatter rank sensitivity, typically weaker confidence separation.
    "sire-rrf-k60": SearchProfile(name="sire-rrf-k60", rrf_k=60, weights=dict(_BASE_WEIGHTS)),
    # Cheaper variant without phonetic strategies (fewer queries per lookup).
    "sire-lexical-only": SearchProfile(name="sire-lexical-only", weights=dict(_BASE_WEIGHTS), disabled=_PHONETIC),
    # Ignores who is speaking (no caller-department context), like the original SIRE_demo.
    "sire-no-caller-context": SearchProfile(name="sire-no-caller-context", weights=dict(_BASE_WEIGHTS), disabled=_CONTEXT),
    # Candidate v2: member-role names outrank groups that merely share the spoken name; a department
    # match on a same-sounding name ("Brynlee" in Pediatric PT) and an exact compact full-name match
    # ("otto line" == "Ottoline") count as stronger evidence.
    "sire-rrf-v2": SearchProfile(name="sire-rrf-v2", weights={**_BASE_WEIGHTS, "group-member-exact": 3.0,
                                                             "user-department-phonetic": 2.5, "user-compact": 3.0}),
}
DEFAULT_PROFILE = "sire-rrf-v1"

_TOKEN = re.compile(r"[a-z0-9]+")
_LUCENE_WORDS = {"and", "or", "not", "to"}
_GROUP_LEAD = {
    "a", "an", "any", "some", "the", "all", "of", "someone", "somebody", "anyone", "anybody", "nearest",
    "closest", "member", "members", "person", "people", "in", "from", "group", "team",
}
_ROLE_FOR_UNIT = re.compile(r"^(?:the\s+)?(?P<role>[a-z0-9 ]{2,40}?)\s+(?:for|of)\s+(?:the\s+)?(?P<unit>.+)$")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _fuzzy(tokens: list[str], long_edits: int, short_edits: int = 1) -> str:
    out = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t in _LUCENE_WORDS:
            out.append(f'"{t}"')
        else:
            out.append(f"{t}~{long_edits if len(t) >= 4 else short_edits}")
    return " ".join(out)


def clean_group_query(text: str) -> str:
    """Strip carrier phrasing ("a member of", "someone in", "all the members of") from a group reference."""
    tokens = _tokens(text)
    while len(tokens) > 1 and tokens[0] in _GROUP_LEAD:
        tokens.pop(0)
    return " ".join(tokens)


def role_rewrite(text: str) -> str | None:
    """"the Coordinator for Inventory" -> "inventory coordinator" (groups are named <unit> <role>)."""
    m = _ROLE_FOR_UNIT.match(" ".join(_tokens(text)))
    if not m:
        return None
    role, unit = m.group("role").strip(), m.group("unit").strip()
    if not role or not unit or len(role.split()) > 3:
        return None
    return f"{unit} {role}"


class _TokenCache:
    def __init__(self, credential) -> None:
        self._credential = credential
        self._token: str | None = None
        self._expires = 0.0
        self._lock = asyncio.Lock()

    async def get(self) -> str:
        async with self._lock:
            if not self._token or time.time() > self._expires - 300:
                token = await asyncio.to_thread(self._credential.get_token, "https://search.azure.com/.default")
                self._token, self._expires = token.token, float(token.expires_on)
            return self._token


class SireSearchClient:
    """Async multi-strategy search over the user and group indexes."""

    def __init__(self, settings: Settings | None = None, profile: str | SearchProfile = DEFAULT_PROFILE,
                 max_concurrency: int = 24) -> None:
        self.settings = settings or load_settings()
        self.settings.require("search_endpoint")
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self._base = self.settings.search_endpoint.rstrip("/")
        self._tokens = None if self.settings.search_api_key else _TokenCache(get_credential(self.settings))
        self._http = httpx.AsyncClient(timeout=20, limits=httpx.Limits(max_connections=max_concurrency))
        self._sem = asyncio.Semaphore(max_concurrency)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "SireSearchClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def with_profile(self, profile: str | SearchProfile) -> "SireSearchClient":
        clone = object.__new__(SireSearchClient)
        clone.__dict__.update(self.__dict__)
        clone.profile = PROFILES[profile] if isinstance(profile, str) else profile
        return clone

    async def warmup(self) -> None:
        """Acquire the search token and open connections before timing anything."""
        await self._headers()
        await self._post(self.settings.search_user_index, {"search": "*", "top": 1, "select": "id"})
        await self._post(self.settings.search_group_index, {"search": "*", "top": 1, "select": "id"})

    async def _headers(self) -> dict[str, str]:
        if self.settings.search_api_key:
            return {"api-key": self.settings.search_api_key, "Content-Type": "application/json"}
        return {"Authorization": f"Bearer {await self._tokens.get()}", "Content-Type": "application/json"}

    async def _post(self, index: str, body: dict) -> list[dict]:
        url = f"{self._base}/indexes/{index}/docs/search?api-version={self.settings.search_api_version}"
        delay = 0.5
        for attempt in range(5):
            async with self._sem:
                resp = await self._http.post(url, headers=await self._headers(), json=body)
            if resp.status_code == 200:
                return resp.json().get("value", [])
            if resp.status_code in (429, 503) and attempt < 4:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Search {index} failed ({resp.status_code}): {resp.text[:300]}")
        return []

    @staticmethod
    def _facility_filter(facility_ids: list[str] | None, extra: str | None = None) -> str | None:
        parts = []
        if facility_ids:
            parts.append("(" + " or ".join(f"FacilityID eq '{f}'" for f in facility_ids) + ")")
        if extra:
            parts.append(extra)
        return " and ".join(parts) or None

    def _body(self, search: str, fields: str, select: str, flt: str | None, *, full: bool = False, all_terms: bool = False) -> dict:
        body = {"search": search, "searchFields": fields, "select": select, "top": self.profile.per_strategy_top,
                "queryType": "full" if full else "simple", "searchMode": "all" if all_terms else "any"}
        if flt:
            body["filter"] = flt
        return body

    def _user_strategies(self, name: str, department: str, doctor: bool, flt: str | None,
                         caller_department_ids: list[str] | None = None) -> list[tuple[str, dict]]:
        sel = "id,DisplayName,FacilityID"
        nt = _tokens(name)
        strategies: list[tuple[str, dict]] = []
        if not nt:
            return strategies
        q = " ".join(nt)
        names = "FirstName,LastName,FullName,SpokenNames"
        phon_fields = "FirstName_phonetic,LastName_phonetic,FullName_phonetic"
        strategies.append(("user-exact", self._body(q, names, sel, flt)))
        fz = _fuzzy(nt, 2)
        if fz:
            strategies.append(("user-fuzzy", self._body(fz, "FirstName,LastName,FullName", sel, flt, full=True)))
        if len(nt) >= 2:
            first, last = nt[0], " ".join(nt[1:])
            strategies.append(("user-targeted", self._body(f'FirstName:"{first}" AND LastName:"{last}"', names, sel, flt, full=True)))
            if len(nt) == 2 and len(first) > 1 and len(nt[1]) > 1:
                ff = f"{first}~2" if len(first) >= 4 else f"{first}~1"
                strategies.append(("user-targeted-fuzzy", self._body(f"FirstName:{ff} AND LastName:{nt[1]}~1", names, sel, flt, full=True)))
            strategies.append(("user-targeted-phonetic", self._body(
                f'FirstName_phonetic:"{first}" AND LastName_phonetic:"{last}"', "FirstName_phonetic,LastName_phonetic", sel, flt, full=True)))
        strategies.append(("user-phonetic-dm", self._body(q, phon_fields, sel, flt)))
        strategies.append(("user-phonetic-soundex", self._body(q, "FirstName_phonetic_soundex,LastName_phonetic_soundex", sel, flt)))
        strategies.append(("user-phonetic-bm", self._body(q, "FirstName_phonetic_bm,LastName_phonetic_bm", sel, flt)))
        cq = compact(name)
        if len(cq) >= 3:
            strategies.append(("user-compact", self._body(cq, "NameCompact", sel, flt)))
        dept_raw = " ".join(_tokens(department))
        if dept_raw:
            phon = " OR ".join(f'FirstName_phonetic:"{t}" OR LastName_phonetic:"{t}"' for t in nt)
            dept_forms = sorted({dept_raw, normalize_text(department)} - {""})
            dept_phrase = " OR ".join(f'Departments:"{d}"' for d in dept_forms)
            dept_all = " ".join(f"+{t}" for t in _tokens(dept_forms[0]))
            strategies.append(("user-department", self._body(f"({q}) AND ({dept_phrase})", names + ",Departments", sel, flt, full=True)))
            strategies.append(("user-department-all", self._body(f"({q}) AND Departments:({dept_all})", names + ",Departments", sel, flt, full=True)))
            strategies.append(("user-department-phonetic", self._body(
                f"({phon}) AND ({dept_phrase})", "FirstName_phonetic,LastName_phonetic,Departments", sel, flt, full=True)))
        if doctor:
            dflt = f"{flt} and IsDoctor eq true" if flt else "IsDoctor eq true"
            strategies.append(("user-doctor", self._body(q, names, sel, dflt)))
            strategies.append(("user-doctor-phonetic", self._body(q, phon_fields, sel, dflt)))
        if caller_department_ids:
            ids = ",".join(caller_department_ids)
            cflt = f"DepartmentIds/any(d: search.in(d, '{ids}', ','))"
            cflt = f"{flt} and {cflt}" if flt else cflt
            strategies.append(("user-caller-department", self._body(q, names, sel, cflt)))
            strategies.append(("user-caller-department-phonetic", self._body(q, phon_fields, sel, cflt)))
        return strategies

    def _group_strategies(self, group: str, flt: str | None) -> list[tuple[str, dict]]:
        sel = "id,GroupName,FacilityID"
        cleaned = clean_group_query(group)
        gt = _tokens(cleaned)
        strategies: list[tuple[str, dict]] = []
        if not gt:
            return strategies
        q = " ".join(gt)
        fields = "GroupName,AlternateNames,MemberNames"
        strategies.append(("group-exact-all", self._body(q, fields, sel, flt, all_terms=True)))
        strategies.append(("group-exact-any", self._body(q, fields, sel, flt)))
        fz = _fuzzy(gt, 1)
        if fz:
            strategies.append(("group-fuzzy", self._body(fz, "GroupName,AlternateNames", sel, flt, full=True, all_terms=True)))
        nq = normalize_text(cleaned)
        if nq and nq != q:
            strategies.append(("group-normalised", self._body(nq, fields, sel, flt, all_terms=True)))
            strategies.append(("group-normalised-any", self._body(nq, fields, sel, flt)))
            nfz = _fuzzy(nq.split(), 1)
            if nfz:
                strategies.append(("group-fuzzy-norm", self._body(nfz, "GroupName,AlternateNames", sel, flt, full=True, all_terms=True)))
        if nq:
            strategies.append(("group-norm-fields", self._body(f'"{nq}"', "AllNamesNormalized", sel, flt)))
            strategies.append(("group-member-exact", self._body(f'"{nq}"', "MemberNamesNormalized", sel, flt)))
        cq = compact(cleaned)
        if len(cq) >= 3:
            strategies.append(("group-compact", self._body(cq, "NameCompact", sel, flt)))
        strategies.append(("group-stemmed", self._body(q, "GroupNameStd,AlternateNamesStd,MemberNames", sel, flt)))
        rewrite = role_rewrite(group)
        if rewrite:
            strategies.append(("group-role-rewrite", self._body(rewrite, "GroupNameStd,AlternateNamesStd,MemberNames", sel, flt, all_terms=True)))
            strategies.append(("group-role-rewrite-exact", self._body(f'"{normalize_text(rewrite)}"', "AllNamesNormalized", sel, flt)))
        return strategies

    async def _fuse(self, index: str, strategies: list[tuple[str, dict]], name_field: str) -> dict[str, Any]:
        active = [(label, body) for label, body in strategies if self.profile.weight(label) > 0]
        started = time.perf_counter()
        responses = await asyncio.gather(*(self._post(index, body) for _, body in active), return_exceptions=True)
        latency_ms = (time.perf_counter() - started) * 1000
        merged: dict[str, dict[str, Any]] = {}
        errors = []
        for (label, _), docs in zip(active, responses):
            if isinstance(docs, Exception):
                errors.append(f"{label}: {docs}")
                continue
            weight = self.profile.weight(label)
            for rank, doc in enumerate(docs, start=1):
                key = doc.get("id")
                if not key:
                    continue
                entry = merged.setdefault(key, {"doc": doc, "rrf": 0.0, "ranks": {}})
                entry["rrf"] += weight / (self.profile.rrf_k + rank)
                entry["ranks"][label] = rank
        best = max((e["rrf"] for e in merged.values()), default=0.0) or 1.0
        ranked = sorted(merged.values(), key=lambda e: e["rrf"], reverse=True)
        results = []
        for i, entry in enumerate(ranked[: self.profile.top], start=1):
            doc = entry["doc"]
            results.append({
                "id": doc["id"],
                "name": doc.get(name_field, ""),
                "facility": doc.get("FacilityID", ""),
                "rank": i,
                "_match_score": round(entry["rrf"] / best * 100.0, 1),
                "_rrf": round(entry["rrf"], 5),
                "_strategies": ", ".join(f"{k}(#{v})" for k, v in sorted(entry["ranks"].items(), key=lambda kv: kv[1])),
            })
        confident = False
        if results:
            gap = results[0]["_match_score"] - (results[1]["_match_score"] if len(results) > 1 else 0.0)
            confident = results[0]["_match_score"] >= self.profile.confident_min_score and gap >= self.profile.confident_min_gap
            results[0]["_gap_to_next"] = round(gap, 1)
        for r in results:
            r["_confident"] = confident and r["rank"] == 1
        return {"results": results, "strategies": [label for label, _ in active], "latency_ms": round(latency_ms, 1),
                "errors": errors, "profile": self.profile.name}

    async def search_user(self, name: str, department: str = "", doctor: bool = False,
                          facility_ids: list[str] | None = None,
                          caller_department_ids: list[str] | None = None) -> dict[str, Any]:
        flt = self._facility_filter(facility_ids)
        strategies = self._user_strategies(name, department or "", bool(doctor), flt, caller_department_ids)
        out = await self._fuse(self.settings.search_user_index, strategies, "DisplayName")
        out["query"] = {"name": name, "department": department or "", "doctor": bool(doctor)}
        return out

    async def search_group(self, group: str, facility_ids: list[str] | None = None) -> dict[str, Any]:
        flt = self._facility_filter(facility_ids)
        strategies = self._group_strategies(group, flt)
        out = await self._fuse(self.settings.search_group_index, strategies, "GroupName")
        out["query"] = {"group": group, "cleaned": clean_group_query(group), "rewrite": role_rewrite(group)}
        return out


def tool_payload(search_result: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    """Compact payload returned to the voice model as the function-call output."""
    return {
        "results": [
            {"id": r["id"], "name": r["name"], "_match_score": r["_match_score"], "_confident": r["_confident"]}
            for r in search_result.get("results", [])[:limit]
        ],
        "count": len(search_result.get("results", [])),
    }


def profile_from_args(name: str, **overrides) -> SearchProfile:
    base = PROFILES[name]
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})
