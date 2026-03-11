"""
Taxonomic Name Resolution Service (TNRS) with SQLite caching.

Responsibilities:
- Resolve raw scientific name strings to canonical names + OpenTree OTT IDs
- Cache all TNRS API responses in a local SQLite database (tnrs_cache.db)
- Process names in batches (TNRS API accepts up to 500 names per request)
- Flag unresolved names for manual review
- Never use raw strings as class labels — always use OTT IDs after resolution

CRITICAL: All taxonomic name resolution must go through this module before training.
Budget 2–4 hours for full NAFlora-1M resolution (~15K unique names).

Usage:
    resolver = TNRSResolver("data/taxonomy/tnrs_cache.db")
    resolved = resolver.resolve(["Clarkia gracilis", "Quercus lobata"])
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

# opentree is a core dep; import OT at module level so tests can patch it.
from opentree import OT

if TYPE_CHECKING:
    import pandas as pd

TNRS_BATCH_SIZE = 500
TNRS_CONTEXT = "Land plants"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tnrs_cache (
    input_name   TEXT PRIMARY KEY,
    matched_name TEXT,
    ott_id       INTEGER,
    score        REAL,
    resolved     INTEGER NOT NULL DEFAULT 0,
    flags        TEXT    NOT NULL DEFAULT '[]'
)
"""


class TNRSResolver:
    """TNRS name resolver with persistent SQLite cache."""

    def __init__(self, cache_db_path: str):
        """Initialize resolver with path to SQLite cache database.

        Creates the database and schema if it does not exist.
        """
        self.cache_db_path = cache_db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, names: list[str]) -> list[dict]:
        """Resolve a list of scientific name strings.

        Returns a list of dicts, one per input name, in the same order:
            {
                'input_name': str,
                'matched_name': str | None,
                'ott_id': int | None,
                'score': float,   # 0–1 match confidence
                'resolved': bool,
                'flags': list[str],  # e.g. ['misspelled', 'synonym']
            }
        """
        results: list[dict | None] = []
        to_fetch: list[str] = []
        to_fetch_indices: list[int] = []

        for i, name in enumerate(names):
            cached = self._cache_lookup(name)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                to_fetch.append(name)
                to_fetch_indices.append(i)

        if to_fetch:
            fetched: list[dict] = []
            for start in range(0, len(to_fetch), TNRS_BATCH_SIZE):
                batch = to_fetch[start : start + TNRS_BATCH_SIZE]
                fetched.extend(self._resolve_batch(batch))
            for idx, result in zip(to_fetch_indices, fetched):
                self._cache_store(result)
                results[idx] = result

        return results  # type: ignore[return-value]

    def resolve_dataframe(self, df: "pd.DataFrame", name_col: str = "scientific_name") -> "pd.DataFrame":
        """Resolve names in a DataFrame column, adding 'ott_id' and 'matched_name' columns."""
        import pandas as pd  # noqa: F811

        names = df[name_col].tolist()
        resolved = self.resolve(names)
        df = df.copy()
        df["ott_id"] = [r["ott_id"] for r in resolved]
        df["matched_name"] = [r["matched_name"] for r in resolved]
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def _cache_lookup(self, name: str) -> dict | None:
        """Look up a name in the SQLite cache. Returns None if not cached."""
        with sqlite3.connect(self.cache_db_path) as conn:
            row = conn.execute(
                "SELECT matched_name, ott_id, score, resolved, flags "
                "FROM tnrs_cache WHERE input_name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        matched_name, ott_id, score, resolved, flags_json = row
        return {
            "input_name": name,
            "matched_name": matched_name,
            "ott_id": ott_id,
            "score": score if score is not None else 0.0,
            "resolved": bool(resolved),
            "flags": json.loads(flags_json) if flags_json else [],
        }

    def _cache_store(self, result: dict) -> None:
        """Store a resolution result in the SQLite cache (upsert)."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tnrs_cache "
                "(input_name, matched_name, ott_id, score, resolved, flags) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    result["input_name"],
                    result.get("matched_name"),
                    result.get("ott_id"),
                    result.get("score", 0.0),
                    int(result.get("resolved", False)),
                    json.dumps(result.get("flags", [])),
                ),
            )

    def _resolve_batch(self, names: list[str]) -> list[dict]:
        """Call TNRS API for a single batch (up to TNRS_BATCH_SIZE names)."""
        response = OT.tnrs_match(names, context_name=TNRS_CONTEXT)
        api_results = response.response_dict.get("results", [])

        # Build a lookup from name → result for ordering
        name_to_result: dict[str, dict] = {}
        for item in api_results:
            input_name = item.get("name", "")
            matches = item.get("matches", [])
            if matches:
                best = matches[0]
                taxon = best.get("taxon", {})
                ott_id = taxon.get("ott_id")
                matched_name = best.get("matched_name") or taxon.get("name")
                score = best.get("score", 0.0)
                flags = taxon.get("flags", [])
                resolved = ott_id is not None
            else:
                ott_id = None
                matched_name = None
                score = 0.0
                flags = ["unmatched"]
                resolved = False
            name_to_result[input_name] = {
                "input_name": input_name,
                "matched_name": matched_name,
                "ott_id": ott_id,
                "score": score,
                "resolved": resolved,
                "flags": flags,
            }

        # Return in the same order as the input names, filling missing names
        out = []
        for name in names:
            if name in name_to_result:
                out.append(name_to_result[name])
            else:
                out.append({
                    "input_name": name,
                    "matched_name": None,
                    "ott_id": None,
                    "score": 0.0,
                    "resolved": False,
                    "flags": ["unmatched"],
                })
        return out
