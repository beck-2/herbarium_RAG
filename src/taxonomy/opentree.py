"""
OpenTree of Life taxonomy utilities.

Responsibilities:
- Fetch subtree for a set of OTT IDs (regional subset) from the OpenTree API
- Compute pairwise patristic distances between taxa from the synoptic tree
- Cache distances in opentree_distances.db (SQLite) — all API results cached
- Export relevant subtree as JSON for bundle (opentree_subtree.json)
- Provide LCA (lowest common ancestor) rank lookups for graph aggregation

CRITICAL: Cache all API results. OpenTree API is rate-limited and
          must be available offline during inference.

Subtree dict schema
-------------------
{
    "newick":   str,            # raw Newick from synth_induced_tree
    "ott_ids":  list[int],      # requested OTT IDs
    "broken":   list[int],      # OTT IDs not in the synthetic tree
    "lineages": {               # str(ott_id) → ancestry list (from taxon_info)
        "<ott_id>": [
            {"rank": str, "name": str, "ott_id": int},
            ...
        ],
        ...
    },
}
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING

from opentree import OT

if TYPE_CHECKING:
    import networkx as nx

# ----- SQLite schema ---------------------------------------------------

_CREATE_SUBTREE_TABLE = """
CREATE TABLE IF NOT EXISTS subtree_cache (
    key     TEXT PRIMARY KEY,
    payload TEXT NOT NULL
)
"""

_CREATE_DISTANCE_TABLE = """
CREATE TABLE IF NOT EXISTS patristic_distances (
    ott_id_a INTEGER NOT NULL,
    ott_id_b INTEGER NOT NULL,
    distance REAL    NOT NULL,
    PRIMARY KEY (ott_id_a, ott_id_b)
)
"""


def _init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_SUBTREE_TABLE)
        conn.execute(_CREATE_DISTANCE_TABLE)


def _subtree_cache_key(ott_ids: list[int]) -> str:
    """Deterministic cache key from sorted OTT ID list."""
    ids_str = ",".join(str(i) for i in sorted(ott_ids))
    return hashlib.sha256(ids_str.encode()).hexdigest()


# ----- Public API ------------------------------------------------------


def fetch_induced_subtree(ott_ids: list[int], cache_path: str) -> dict:
    """Fetch the induced subtree for a set of OTT IDs.

    Returns a subtree dict (see module docstring for schema).
    Results are cached in SQLite to avoid repeated API calls.

    Args:
        ott_ids: List of OpenTree OTT IDs for the taxa of interest.
        cache_path: Path to SQLite cache file (created if absent).
    """
    _init_db(cache_path)
    key = _subtree_cache_key(ott_ids)

    # Cache lookup
    with sqlite3.connect(cache_path) as conn:
        row = conn.execute(
            "SELECT payload FROM subtree_cache WHERE key = ?", (key,)
        ).fetchone()
    if row is not None:
        return json.loads(row[0])

    # Fetch from API
    response = OT.synth_induced_tree(ott_ids=ott_ids, ignore_unknown_ids=True)
    rd = response.response_dict
    broken_nodes = rd.get("broken", [])
    # broken is a dict node_id → list in some API versions; normalise to list[int]
    if isinstance(broken_nodes, dict):
        broken_ids = [int(k.replace("ott", "")) for k in broken_nodes.keys()]
    else:
        broken_ids = [int(str(b).replace("ott", "")) for b in broken_nodes if b]

    subtree: dict = {
        "newick": rd.get("newick", ""),
        "ott_ids": ott_ids,
        "broken": broken_ids,
        "lineages": {},
    }

    # Store in cache
    payload = json.dumps(subtree)
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO subtree_cache (key, payload) VALUES (?, ?)",
            (key, payload),
        )

    return subtree


def compute_patristic_distances(
    ott_ids: list[int],
    subtree: dict,
    cache_db_path: str,
) -> dict[tuple[int, int], float]:
    """Compute pairwise patristic distances from a subtree.

    Uses topological (hop-count) distance along the Newick tree.  Pairs that
    share a genus have smaller distances than cross-family pairs.

    Results are cached in the ``patristic_distances`` table.  If a pair is
    already cached it is read directly without re-parsing the tree.

    Args:
        ott_ids: Taxa to compute distances for.
        subtree: Subtree dict from :func:`fetch_induced_subtree`.
        cache_db_path: Path to SQLite cache.

    Returns:
        Dict mapping ``(ott_id_a, ott_id_b)`` pairs (a < b) to float distances.
    """
    import networkx as nx

    _init_db(cache_db_path)

    ott_ids = sorted(set(ott_ids))
    pairs = [(a, b) for i, a in enumerate(ott_ids) for b in ott_ids[i + 1 :]]

    # Check cache for all pairs
    results: dict[tuple[int, int], float] = {}
    missing_pairs: list[tuple[int, int]] = []
    with sqlite3.connect(cache_db_path) as conn:
        for a, b in pairs:
            row = conn.execute(
                "SELECT distance FROM patristic_distances WHERE ott_id_a=? AND ott_id_b=?",
                (a, b),
            ).fetchone()
            if row is not None:
                results[(a, b)] = row[0]
            else:
                missing_pairs.append((a, b))

    if not missing_pairs:
        return results

    # Parse newick and compute distances for missing pairs
    newick = subtree.get("newick", "")
    G, _ = _newick_to_digraph(newick)
    undirected = G.to_undirected()

    # Build a map from ott_id → node label in the tree
    ott_to_node: dict[int, str] = {}
    for node in G.nodes:
        node_str = str(node)
        if "_ott" in node_str:
            try:
                ott_part = node_str.rsplit("_ott", 1)[-1]
                ott_int = int(ott_part)
                ott_to_node[ott_int] = node_str
            except ValueError:
                pass
        elif node_str.startswith("ott") and not node_str.startswith("ott_"):
            try:
                ott_int = int(node_str[3:])
                ott_to_node[ott_int] = node_str
            except ValueError:
                pass

    new_results: dict[tuple[int, int], float] = {}
    for a, b in missing_pairs:
        node_a = ott_to_node.get(a)
        node_b = ott_to_node.get(b)
        if node_a is not None and node_b is not None and nx.has_path(undirected, node_a, node_b):
            dist = float(nx.shortest_path_length(undirected, node_a, node_b))
        else:
            # Fall back: estimate from lineage depth difference
            dist = _lineage_distance(a, b, subtree.get("lineages", {}))
        new_results[(a, b)] = dist
        results[(a, b)] = dist

    # Cache new results
    with sqlite3.connect(cache_db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO patristic_distances (ott_id_a, ott_id_b, distance) VALUES (?,?,?)",
            [(a, b, d) for (a, b), d in new_results.items()],
        )

    return results


def get_lca_rank(ott_id_a: int, ott_id_b: int, subtree: dict) -> str:
    """Return the taxonomic rank of the LCA of two taxa.

    Uses the lineage data stored in the subtree dict (populated by
    :func:`fetch_induced_subtree` + taxon_info enrichment).

    Returns one of: 'species', 'genus', 'family', 'order', 'class',
    'phylum', 'kingdom', or 'no rank'.
    """
    lineages = subtree.get("lineages", {})
    lin_a = lineages.get(str(ott_id_a), [])
    lin_b = lineages.get(str(ott_id_b), [])

    # Build ancestor set for a (rank → ott_id)
    ancestors_a: dict[int, str] = {entry["ott_id"]: entry["rank"] for entry in lin_a}

    # Walk up b's lineage and find the first ancestor shared with a
    for entry in lin_b:
        shared_ott = entry["ott_id"]
        if shared_ott in ancestors_a:
            return ancestors_a[shared_ott]

    return "no rank"


def export_subtree_json(subtree: dict, output_path: str) -> None:
    """Serialize the subtree dict to JSON for inclusion in a regional bundle."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subtree, f, indent=2)


# ----- Newick parser ---------------------------------------------------


def _newick_to_digraph(newick: str) -> tuple["nx.DiGraph", str]:
    """Parse a topological Newick string into a directed graph (parent → child).

    Handles the OpenTree format where internal nodes may have ``ott{id}`` or
    ``mrcaott…`` labels and leaves have ``Name_ott{id}`` labels.  Branch
    lengths (``:``) are skipped if present.

    Returns:
        (G, root_node_id) — directed graph and the root node label.
    """
    import networkx as nx

    G: nx.DiGraph = nx.DiGraph()
    _counter = [0]

    def fresh() -> str:
        _counter[0] += 1
        return f"__internal_{_counter[0]}"

    newick = newick.strip().rstrip(";")
    pos = [0]
    n = len(newick)

    def read_label() -> str:
        start = pos[0]
        while pos[0] < n and newick[pos[0]] not in ",();":
            if newick[pos[0]] == ":":
                # branch length — discard
                pos[0] += 1
                while pos[0] < n and newick[pos[0]] not in ",();":
                    pos[0] += 1
                break
            pos[0] += 1
        return newick[start : pos[0]].strip()

    def parse_node() -> str:
        if pos[0] < n and newick[pos[0]] == "(":
            pos[0] += 1  # consume '('
            children: list[str] = []
            while True:
                child = parse_node()
                children.append(child)
                if pos[0] >= n or newick[pos[0]] == ")":
                    break
                if newick[pos[0]] == ",":
                    pos[0] += 1  # consume ','
            if pos[0] < n and newick[pos[0]] == ")":
                pos[0] += 1  # consume ')'
            label = read_label()
            node_id = label if label else fresh()
            G.add_node(node_id)
            for child in children:
                G.add_edge(node_id, child)
            return node_id
        else:
            label = read_label()
            node_id = label if label else fresh()
            G.add_node(node_id)
            return node_id

    root = parse_node()
    return G, root


def _lineage_distance(ott_id_a: int, ott_id_b: int, lineages: dict) -> float:
    """Estimate distance from lineage depths when the tree node isn't found.

    Counts the number of lineage steps to the shared ancestor.
    Falls back to a large constant if no lineage data is available.
    """
    lin_a = lineages.get(str(ott_id_a), [])
    lin_b = lineages.get(str(ott_id_b), [])
    if not lin_a or not lin_b:
        return 20.0

    ancestors_a = {entry["ott_id"]: i for i, entry in enumerate(lin_a)}
    for depth_b, entry in enumerate(lin_b):
        if entry["ott_id"] in ancestors_a:
            depth_a = ancestors_a[entry["ott_id"]]
            return float(depth_a + depth_b + 2)  # +2 for the two leaf edges

    return 20.0
