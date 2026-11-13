"""Azure AI Search index schemas and document builders for the tenant directory.

The analyzers replicate the production SIRE indexes (keyword names + Double Metaphone /
Beider-Morse / Soundex phonetic fields for people; word-delimiter + shingle + number-split
char filters for groups) and add fields the tenant-gamma phrases need: facility scoping,
group memberships ("Zephyrine in Food Services"), member role names ("the charge nurse"),
compact forms for spelled-out names, and English-stemmed fields for plurals ("techs").
"""

from __future__ import annotations

import base64
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .textnorm import compact, normalize_text

_PHONETIC_FILTERS = [
    {"@odata.type": "#Microsoft.Azure.Search.PhoneticTokenFilter", "name": "phonetic_dm", "encoder": "doubleMetaphone", "replace": True},
    {"@odata.type": "#Microsoft.Azure.Search.PhoneticTokenFilter", "name": "phonetic_bm", "encoder": "beiderMorse", "replace": True},
    {"@odata.type": "#Microsoft.Azure.Search.PhoneticTokenFilter", "name": "phonetic_soundex", "encoder": "soundex", "replace": True},
]


def _custom(name: str, tokenizer: str, filters: list[str], char_filters: list[str] | None = None) -> dict:
    return {
        "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
        "name": name,
        "tokenizer": tokenizer,
        "tokenFilters": filters,
        "charFilters": char_filters or [],
    }


def _field(name: str, *, type_: str = "Edm.String", key: bool = False, searchable: bool = True,
           filterable: bool = False, retrievable: bool = True, analyzer: str | None = None,
           index_analyzer: str | None = None, search_analyzer: str | None = None) -> dict:
    field = {"name": name, "type": type_, "key": key, "searchable": searchable, "filterable": filterable,
             "retrievable": retrievable, "sortable": False, "facetable": False}
    if analyzer:
        field["analyzer"] = analyzer
    if index_analyzer:
        field["indexAnalyzer"] = index_analyzer
        field["searchAnalyzer"] = search_analyzer
    return field


def user_index_definition(name: str) -> dict:
    coll = "Collection(Edm.String)"
    return {
        "name": name,
        "fields": [
            _field("DocKey", key=True, searchable=False),
            _field("id", searchable=False, filterable=True),
            _field("DisplayName", analyzer="standard.lucene"),
            _field("FirstName", analyzer="user_name_analyzer"),
            _field("LastName", analyzer="user_name_analyzer"),
            _field("FullName", analyzer="standard.lucene"),
            _field("SpokenNames", type_=coll, analyzer="standard.lucene"),
            _field("FirstName_phonetic", analyzer="phonetic_dm_analyzer", retrievable=False),
            _field("LastName_phonetic", analyzer="phonetic_dm_analyzer", retrievable=False),
            _field("FullName_phonetic", analyzer="phonetic_dm_tokenized_analyzer", retrievable=False),
            _field("FirstName_phonetic_bm", analyzer="phonetic_bm_analyzer", retrievable=False),
            _field("LastName_phonetic_bm", analyzer="phonetic_bm_analyzer", retrievable=False),
            _field("FirstName_phonetic_soundex", analyzer="phonetic_soundex_analyzer", retrievable=False),
            _field("LastName_phonetic_soundex", analyzer="phonetic_soundex_analyzer", retrievable=False),
            _field("NameCompact", type_=coll, analyzer="keyword_lowercase_plain", retrievable=False),
            _field("Departments", type_=coll, analyzer="en.lucene", retrievable=False),
            _field("DepartmentIds", type_=coll, searchable=False, filterable=True, retrievable=False),
            _field("FacilityID", searchable=False, filterable=True),
            _field("FacilityName", searchable=False),
            _field("IsDoctor", type_="Edm.Boolean", searchable=False, filterable=True),
        ],
        "analyzers": [
            _custom("user_name_analyzer", "keyword_v2", ["lowercase", "asciifolding"]),
            _custom("phonetic_dm_analyzer", "keyword_v2", ["lowercase", "phonetic_dm"]),
            _custom("phonetic_dm_tokenized_analyzer", "standard_v2", ["lowercase", "phonetic_dm"]),
            _custom("phonetic_bm_analyzer", "keyword_v2", ["lowercase", "phonetic_bm"]),
            _custom("phonetic_soundex_analyzer", "keyword_v2", ["lowercase", "phonetic_soundex"]),
            _custom("keyword_lowercase_plain", "keyword_v2", ["lowercase"]),
        ],
        "tokenFilters": _PHONETIC_FILTERS,
    }


_GROUP_CHAR_FILTERS = [
    {"@odata.type": "#Microsoft.Azure.Search.PatternReplaceCharFilter", "name": "collapse_single_runs",
     "pattern": "(?<=\\b[A-Za-z0-9])\\s+(?=[A-Za-z0-9]\\b)", "replacement": ""},
    {"@odata.type": "#Microsoft.Azure.Search.PatternReplaceCharFilter", "name": "collapse_sequential_singles",
     "pattern": "\\b([A-Za-z0-9])\\s+([A-Za-z0-9])\\b", "replacement": "$1$2"},
    {"@odata.type": "#Microsoft.Azure.Search.PatternReplaceCharFilter", "name": "digit_letter_split",
     "pattern": "(?<=\\d)(?=[A-Za-z])", "replacement": " "},
    {"@odata.type": "#Microsoft.Azure.Search.PatternReplaceCharFilter", "name": "letter_digit_split",
     "pattern": "(?<=[A-Za-z])(?=\\d)", "replacement": " "},
    {"@odata.type": "#Microsoft.Azure.Search.MappingCharFilter", "name": "hospital_char_mapping",
     "mappings": ["&=> and ", "'=> ", "-=> ", "_=> ", ".=> ", ",=> ", "(=> ", ")=> "]},
]
_GROUP_CHAIN = ["collapse_single_runs", "collapse_sequential_singles", "digit_letter_split", "letter_digit_split", "hospital_char_mapping"]


def group_index_definition(name: str) -> dict:
    coll = "Collection(Edm.String)"
    return {
        "name": name,
        "fields": [
            _field("DocKey", key=True, searchable=False),
            _field("id", searchable=False, filterable=True),
            _field("GroupName", analyzer="hospital_group_analyzer"),
            _field("GroupNameStd", analyzer="en.lucene", retrievable=False),
            _field("AlternateNames", type_=coll, analyzer="hospital_group_analyzer"),
            _field("AlternateNamesStd", type_=coll, analyzer="en.lucene", retrievable=False),
            _field("MemberNames", type_=coll, analyzer="en.lucene"),
            _field("MemberNamesNormalized", type_=coll, analyzer="keyword_lowercase", retrievable=False),
            _field("AllNamesNormalized", type_=coll, analyzer="keyword_lowercase", retrievable=False),
            _field("NameCompact", type_=coll, analyzer="keyword_lowercase_plain", retrievable=False),
            _field("GroupName_edge", index_analyzer="front_edge_ngram_analyzer", search_analyzer="keyword_lowercase", retrievable=False),
            _field("FacilityID", searchable=False, filterable=True),
            _field("FacilityName", searchable=False),
        ],
        "analyzers": [
            _custom("hospital_group_analyzer", "keyword_v2",
                    ["lowercase", "asciifolding", "hospital_wd_filter", "hospital_shingle_filter"], _GROUP_CHAIN),
            _custom("keyword_lowercase", "keyword_v2", ["lowercase"], _GROUP_CHAIN),
            _custom("keyword_lowercase_plain", "keyword_v2", ["lowercase"]),
            _custom("front_edge_ngram_analyzer", "front_edge_ngram_tok", ["lowercase"]),
        ],
        "tokenizers": [
            {"@odata.type": "#Microsoft.Azure.Search.EdgeNGramTokenizer", "name": "front_edge_ngram_tok",
             "minGram": 2, "maxGram": 25, "tokenChars": []},
        ],
        "tokenFilters": [
            {"@odata.type": "#Microsoft.Azure.Search.WordDelimiterTokenFilter", "name": "hospital_wd_filter",
             "generateWordParts": True, "generateNumberParts": True, "catenateWords": True, "catenateNumbers": True,
             "catenateAll": True, "splitOnCaseChange": False, "preserveOriginal": True, "splitOnNumerics": True,
             "stemEnglishPossessive": True, "protectedWords": []},
            {"@odata.type": "#Microsoft.Azure.Search.ShingleTokenFilter", "name": "hospital_shingle_filter",
             "maxShingleSize": 4, "minShingleSize": 2, "outputUnigrams": True, "outputUnigramsIfNoShingles": False,
             "tokenSeparator": "", "filterToken": "_"},
        ],
        "charFilters": _GROUP_CHAR_FILTERS,
    }


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def doc_key(urn: str) -> str:
    """Search keys allow only [A-Za-z0-9_-=]; entity URNs can contain accents (e.g. "é")."""
    return base64.urlsafe_b64encode(urn.encode("utf-8")).decode("ascii")


def _dedupe(values) -> list[str]:
    seen, out = set(), []
    for v in values:
        v = (v or "").strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def load_memberships(entity_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return (groups by URN, group URNs per user URN)."""
    groups = {g["entity_urn"]: g for g in read_jsonl_gz(entity_dir / "groups.jsonl.gz")}
    memberships: dict[str, list[str]] = defaultdict(list)
    for edge in read_jsonl_gz(entity_dir / "groupmembers.jsonl.gz"):
        if edge["group_urn"] in groups:
            memberships[edge["user_urn"]].append(edge["group_urn"])
    return groups, memberships


def build_documents(entity_dir: Path) -> tuple[list[dict], list[dict]]:
    """Build (user_docs, group_docs) from the tenant entity lists (schema v1.1)."""
    groups, memberships = load_memberships(entity_dir)
    learned: dict[str, list[str]] = defaultdict(list)
    learned_path = entity_dir / "learnednames.jsonl.gz"
    if learned_path.exists():
        for rec in read_jsonl_gz(learned_path):
            learned[rec["entity_urn"]].append(rec["display_name"])

    user_docs = []
    for u in read_jsonl_gz(entity_dir / "users.jsonl.gz"):
        first, last = (u.get("first_name") or "").strip(), (u.get("last_name") or "").strip()
        full = (u.get("display_name") or f"{first} {last}").strip()
        spoken = _dedupe(list(u.get("spoken_names") or []) + learned.get(u["entity_urn"], []) + [u.get("identifying_phrase") or ""])
        user_docs.append({
            "DocKey": doc_key(u["entity_urn"]),
            "id": u["entity_urn"],
            "DisplayName": full,
            "FirstName": first,
            "LastName": last,
            "FullName": full,
            "SpokenNames": spoken,
            "FirstName_phonetic": first,
            "LastName_phonetic": last,
            "FullName_phonetic": full,
            "FirstName_phonetic_bm": first,
            "LastName_phonetic_bm": last,
            "FirstName_phonetic_soundex": first,
            "LastName_phonetic_soundex": last,
            "NameCompact": _dedupe([compact(full), compact(f"{first} {last}"), compact(first), compact(last)] + [compact(s) for s in spoken]),
            "Departments": _dedupe(
                [groups[g]["display_name"] for g in memberships.get(u["entity_urn"], [])]
                + [normalize_text(groups[g]["display_name"]) for g in memberships.get(u["entity_urn"], [])]
            ),
            "DepartmentIds": sorted(set(memberships.get(u["entity_urn"], []))),
            "FacilityID": u.get("facility_urn") or "",
            "FacilityName": u.get("facility_name") or "",
            "IsDoctor": bool(u.get("doctor_prefix")),
        })

    group_docs = []
    for g in groups.values():
        name = (g.get("display_name") or "").strip()
        alternates = _dedupe([s for s in (g.get("spoken_names") or []) if s.strip().lower() != name.lower()])
        members = _dedupe([g.get("member_name_singular") or "", g.get("member_name_plural") or ""])
        all_names = _dedupe([name] + alternates + members)
        group_docs.append({
            "DocKey": doc_key(g["entity_urn"]),
            "id": g["entity_urn"],
            "GroupName": name,
            "GroupNameStd": name,
            "AlternateNames": alternates,
            "AlternateNamesStd": alternates,
            "MemberNames": members,
            "MemberNamesNormalized": _dedupe(normalize_text(m) for m in members),
            "AllNamesNormalized": _dedupe(normalize_text(n) for n in all_names),
            "NameCompact": _dedupe(compact(n) for n in all_names),
            "GroupName_edge": normalize_text(name),
            "FacilityID": g.get("facility_urn") or "",
            "FacilityName": g.get("facility_name") or "",
        })
    return user_docs, group_docs
