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

import sqlite3

# TODO(phase2): implement SQLite cache schema (create table if not exists)
# TODO(phase2): implement batch TNRS API calls via opentree.OT.tnrs_match_names
# TODO(phase2): implement fallback fuzzy matching for near-misses
# TODO(phase2): implement unresolved name flagging and logging

TNRS_BATCH_SIZE = 500
TNRS_CONTEXT = "Land plants"


class TNRSResolver:
    """TNRS name resolver with persistent SQLite cache."""

    def __init__(self, cache_db_path: str):
        """Initialize resolver with path to SQLite cache database.

        Creates the database and schema if it does not exist.
        """
        self.cache_db_path = cache_db_path
        # TODO(phase2): self._init_db()

    def resolve(self, names: list[str]) -> list[dict]:
        """Resolve a list of scientific name strings.

        Returns a list of dicts, one per input name:
            {
                'input_name': str,
                'matched_name': str | None,
                'ott_id': int | None,
                'score': float,   # 0–1 match confidence
                'resolved': bool,
                'flags': list[str],  # e.g. ['misspelled', 'synonym']
            }
        """
        raise NotImplementedError

    def resolve_dataframe(self, df, name_col: str = "scientific_name") -> "pd.DataFrame":
        """Resolve names in a DataFrame column, adding 'ott_id' and 'matched_name' columns."""
        raise NotImplementedError

    def _resolve_batch(self, names: list[str]) -> list[dict]:
        """Call TNRS API for a single batch (up to TNRS_BATCH_SIZE names)."""
        raise NotImplementedError

    def _cache_lookup(self, name: str) -> dict | None:
        """Look up a name in the SQLite cache. Returns None if not cached."""
        raise NotImplementedError

    def _cache_store(self, result: dict) -> None:
        """Store a resolution result in the SQLite cache."""
        raise NotImplementedError
