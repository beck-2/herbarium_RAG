"""
Tests for src/taxonomy/: tnrs, opentree, gnn.

Phase 0 stubs: import checks.
Phase 2: unit tests (no network) + network tests (marked with network).

Network tests require internet access and are skipped by default.
Run them explicitly: pytest -m network
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Import / API surface (Phase 0)
# ---------------------------------------------------------------------------

def test_tnrs_module_importable():
    from src.taxonomy import tnrs
    assert hasattr(tnrs, "TNRSResolver")
    assert hasattr(tnrs, "TNRS_BATCH_SIZE")
    assert hasattr(tnrs, "TNRS_CONTEXT")


def test_opentree_module_importable():
    from src.taxonomy import opentree
    assert hasattr(opentree, "fetch_induced_subtree")
    assert hasattr(opentree, "compute_patristic_distances")
    assert hasattr(opentree, "get_lca_rank")
    assert hasattr(opentree, "export_subtree_json")


def test_gnn_module_importable():
    pytest.importorskip("torch", reason="torch not installed; run: pip install 'hyperbolic-herbarium[ml]'")
    from src.taxonomy import gnn
    assert hasattr(gnn, "TaxonomyGNNRegularizer")


# ---------------------------------------------------------------------------
# TNRS — Phase 2 unit tests (no network)
# ---------------------------------------------------------------------------

def _make_fake_tnrs_response(names):
    """Return a mock OT.tnrs_match response dict for given names."""
    results = []
    for name in names:
        genus = name.split()[0] if name else "Unknown"
        ott_id = abs(hash(name)) % 1_000_000 + 1
        results.append({
            "name": name,
            "matches": [{
                "matched_name": name,
                "score": 1.0,
                "is_approximate_match": False,
                "is_synonym": False,
                "taxon": {
                    "name": name,
                    "ott_id": ott_id,
                    "rank": "species",
                    "flags": [],
                },
            }],
        })
    return {"results": results}


class TestTNRSResolverUnit:
    def test_init_creates_db(self, tmp_path):
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)
        # DB file must exist and have the expected table
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "tnrs_cache" in tables

    def test_cache_miss_then_hit(self, tmp_path):
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        # Nothing in cache yet
        assert resolver._cache_lookup("Quercus alba") is None

        # Store a result
        result = {
            "input_name": "Quercus alba",
            "matched_name": "Quercus alba",
            "ott_id": 791112,
            "score": 1.0,
            "resolved": True,
            "flags": [],
        }
        resolver._cache_store(result)

        # Now it should be in cache
        cached = resolver._cache_lookup("Quercus alba")
        assert cached is not None
        assert cached["ott_id"] == 791112
        assert cached["resolved"] is True
        assert cached["flags"] == []

    def test_cache_store_idempotent(self, tmp_path):
        """Storing the same name twice should not raise."""
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)
        result = {
            "input_name": "Acer rubrum",
            "matched_name": "Acer rubrum",
            "ott_id": 1039827,
            "score": 1.0,
            "resolved": True,
            "flags": [],
        }
        resolver._cache_store(result)
        resolver._cache_store(result)  # second store should upsert cleanly
        cached = resolver._cache_lookup("Acer rubrum")
        assert cached["ott_id"] == 1039827

    def test_resolve_uses_cache_on_second_call(self, tmp_path):
        """resolve() should not call the API for a name that's already cached."""
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        names = ["Quercus alba"]
        fake_response = MagicMock()
        fake_response.response_dict = _make_fake_tnrs_response(names)

        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.return_value = fake_response
            results1 = resolver.resolve(names)  # hits API

        # Second call — API should NOT be called again
        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.return_value = fake_response
            results2 = resolver.resolve(names)
            mock_ot.tnrs_match.assert_not_called()

        assert results1[0]["ott_id"] == results2[0]["ott_id"]

    def test_resolve_returns_correct_structure(self, tmp_path):
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        names = ["Clarkia gracilis", "Quercus lobata"]
        fake_response = MagicMock()
        fake_response.response_dict = _make_fake_tnrs_response(names)

        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.return_value = fake_response
            results = resolver.resolve(names)

        assert len(results) == 2
        for r in results:
            assert "input_name" in r
            assert "matched_name" in r
            assert "ott_id" in r
            assert "score" in r
            assert "resolved" in r
            assert "flags" in r
            assert isinstance(r["flags"], list)

    def test_resolve_unmatched_name(self, tmp_path):
        """Names with no matches should have resolved=False and ott_id=None."""
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        no_match_response = MagicMock()
        no_match_response.response_dict = {
            "results": [{"name": "Zzz notaspecies", "matches": []}]
        }

        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.return_value = no_match_response
            results = resolver.resolve(["Zzz notaspecies"])

        assert results[0]["resolved"] is False
        assert results[0]["ott_id"] is None

    def test_resolve_dataframe(self, tmp_path):
        import pandas as pd
        from src.taxonomy.tnrs import TNRSResolver
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        import pandas as pd
        df = pd.DataFrame({"scientific_name": ["Quercus alba", "Acer rubrum"]})
        names = df["scientific_name"].tolist()
        fake_response = MagicMock()
        fake_response.response_dict = _make_fake_tnrs_response(names)

        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.return_value = fake_response
            result_df = resolver.resolve_dataframe(df)

        assert "ott_id" in result_df.columns
        assert "matched_name" in result_df.columns
        assert len(result_df) == 2

    def test_batch_splitting(self, tmp_path):
        """Names beyond TNRS_BATCH_SIZE should be split into multiple API calls."""
        from src.taxonomy.tnrs import TNRSResolver, TNRS_BATCH_SIZE
        db_path = str(tmp_path / "tnrs_cache.db")
        resolver = TNRSResolver(db_path)

        # Create more names than one batch
        names = [f"Species number{i}" for i in range(TNRS_BATCH_SIZE + 3)]

        def fake_tnrs_match(batch, context_name=None, **kwargs):
            r = MagicMock()
            r.response_dict = _make_fake_tnrs_response(batch)
            return r

        with patch("src.taxonomy.tnrs.OT") as mock_ot:
            mock_ot.tnrs_match.side_effect = fake_tnrs_match
            results = resolver.resolve(names)

        # Should have called API at least twice
        assert mock_ot.tnrs_match.call_count >= 2
        assert len(results) == len(names)


# ---------------------------------------------------------------------------
# OpenTree — Phase 2 unit tests (no network)
# ---------------------------------------------------------------------------

# Minimal Newick fixture for two leaves sharing a common ancestor.
# Layout:  ((Acer_rubrum_ott1039827,Acer_saccharum_ott1039830)ott790665,Quercus_alba_ott791112);
_FIXTURE_NEWICK = (
    "((Acer_rubrum_ott1039827,Acer_saccharum_ott1039830)ott790665,Quercus_alba_ott791112)root;"
)

_FIXTURE_SUBTREE = {
    "newick": _FIXTURE_NEWICK,
    "ott_ids": [1039827, 1039830, 791112],
    "broken": [],
    "lineages": {
        "1039827": [
            {"rank": "genus", "name": "Acer", "ott_id": 790665},
            {"rank": "family", "name": "Sapindaceae", "ott_id": 123},
        ],
        "1039830": [
            {"rank": "genus", "name": "Acer", "ott_id": 790665},
            {"rank": "family", "name": "Sapindaceae", "ott_id": 123},
        ],
        "791112": [
            {"rank": "genus", "name": "Quercus", "ott_id": 791121},
            {"rank": "family", "name": "Fagaceae", "ott_id": 267713},
        ],
    },
}


class TestOpenTreeUnit:
    def test_fetch_induced_subtree_uses_cache(self, tmp_path):
        """Second call with same OTT IDs should not hit the API."""
        from src.taxonomy.opentree import fetch_induced_subtree

        db_path = str(tmp_path / "opentree.db")

        fake_resp = MagicMock()
        fake_resp.response_dict = {
            "newick": _FIXTURE_NEWICK,
            "broken": [],
            "supporting_studies": [],
        }

        with patch("src.taxonomy.opentree.OT") as mock_ot:
            mock_ot.synth_induced_tree.return_value = fake_resp
            subtree1 = fetch_induced_subtree([1039827, 1039830, 791112], db_path)

        with patch("src.taxonomy.opentree.OT") as mock_ot:
            mock_ot.synth_induced_tree.return_value = fake_resp
            subtree2 = fetch_induced_subtree([1039827, 1039830, 791112], db_path)
            mock_ot.synth_induced_tree.assert_not_called()

        assert subtree1["newick"] == subtree2["newick"]

    def test_fetch_induced_subtree_structure(self, tmp_path):
        from src.taxonomy.opentree import fetch_induced_subtree

        db_path = str(tmp_path / "opentree.db")
        fake_resp = MagicMock()
        fake_resp.response_dict = {
            "newick": _FIXTURE_NEWICK,
            "broken": [],
            "supporting_studies": [],
        }

        with patch("src.taxonomy.opentree.OT") as mock_ot:
            mock_ot.synth_induced_tree.return_value = fake_resp
            subtree = fetch_induced_subtree([1039827, 1039830, 791112], db_path)

        assert "newick" in subtree
        assert "ott_ids" in subtree
        assert "broken" in subtree
        assert isinstance(subtree["ott_ids"], list)

    def test_compute_patristic_distances_returns_dict(self):
        from src.taxonomy.opentree import compute_patristic_distances

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            distances = compute_patristic_distances(
                [1039827, 1039830, 791112],
                _FIXTURE_SUBTREE,
                f.name,
            )

        assert isinstance(distances, dict)
        # Each pair should have a distance
        assert (1039827, 1039830) in distances or (1039830, 1039827) in distances
        assert (1039827, 791112) in distances or (791112, 1039827) in distances

    def test_compute_patristic_distances_genus_closer_than_family(self):
        """Two species in the same genus should be closer than cross-genus."""
        from src.taxonomy.opentree import compute_patristic_distances

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            distances = compute_patristic_distances(
                [1039827, 1039830, 791112],
                _FIXTURE_SUBTREE,
                f.name,
            )

        def get_dist(a, b):
            return distances.get((a, b), distances.get((b, a)))

        same_genus = get_dist(1039827, 1039830)    # Acer rubrum vs Acer saccharum
        cross_genus = get_dist(1039827, 791112)    # Acer vs Quercus
        assert same_genus is not None
        assert cross_genus is not None
        assert same_genus < cross_genus

    def test_get_lca_rank_same_genus(self):
        from src.taxonomy.opentree import get_lca_rank

        rank = get_lca_rank(1039827, 1039830, _FIXTURE_SUBTREE)
        assert rank == "genus"

    def test_get_lca_rank_different_genus_same_family(self):
        """If two taxa have different genera but same family, LCA is 'family'."""
        subtree = {
            "newick": _FIXTURE_NEWICK,
            "ott_ids": [1039827, 791112],
            "broken": [],
            "lineages": {
                "1039827": [
                    {"rank": "genus", "name": "Acer", "ott_id": 790665},
                    {"rank": "family", "name": "Sapindaceae", "ott_id": 123},
                ],
                "791112": [
                    {"rank": "genus", "name": "Quercus", "ott_id": 791121},
                    {"rank": "family", "name": "Sapindaceae", "ott_id": 123},  # same family
                ],
            },
        }
        from src.taxonomy.opentree import get_lca_rank
        rank = get_lca_rank(1039827, 791112, subtree)
        assert rank == "family"

    def test_get_lca_rank_different_family(self):
        from src.taxonomy.opentree import get_lca_rank
        rank = get_lca_rank(1039827, 791112, _FIXTURE_SUBTREE)
        # Different family (Sapindaceae vs Fagaceae) → higher than family
        assert rank not in ("species", "genus", "family")

    def test_export_subtree_json(self, tmp_path):
        from src.taxonomy.opentree import export_subtree_json

        out = tmp_path / "subtree.json"
        export_subtree_json(_FIXTURE_SUBTREE, str(out))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["newick"] == _FIXTURE_SUBTREE["newick"]
        assert loaded["ott_ids"] == _FIXTURE_SUBTREE["ott_ids"]


# ---------------------------------------------------------------------------
# Network tests — require internet, marked explicitly
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_tnrs_resolve_real_names(tmp_path):
    """Integration: actually call the TNRS API."""
    from src.taxonomy.tnrs import TNRSResolver

    db_path = str(tmp_path / "tnrs_cache.db")
    resolver = TNRSResolver(db_path)
    results = resolver.resolve(["Quercus alba", "Acer rubrum"])

    assert len(results) == 2
    assert results[0]["resolved"] is True
    assert results[0]["ott_id"] == 791112       # Quercus alba
    assert results[1]["ott_id"] == 1039827      # Acer rubrum


@pytest.mark.network
def test_fetch_induced_subtree_real(tmp_path):
    """Integration: actually call the OpenTree API."""
    from src.taxonomy.opentree import fetch_induced_subtree

    db_path = str(tmp_path / "opentree.db")
    subtree = fetch_induced_subtree([791112, 1039827], db_path)
    assert "newick" in subtree
    assert len(subtree["newick"]) > 10
