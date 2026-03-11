"""
One-time script to fetch a small OpenTree subtree fixture for Phase 7 tests.

Resolves ~10 species via TNRS, fetches the induced subtree, populates lineages,
and saves to tests/fixtures/opentree_mini_subtree.json.

Run once:
    python3.14 scripts/fetch_opentree_fixture.py

The result is committed to the repo; tests never need network access.
"""

import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opentree import OT

# Species chosen to exercise:
#   - same-genus pairs (Poa pratensis / Poa annua)
#   - same-family pairs (Poa / Festuca, both Poaceae)
#   - convergent-pair families (Cactaceae vs Euphorbiaceae, Droseraceae vs Nepenthaceae)
#   - outgroup (Fagaceae, Asteraceae)
SPECIES = [
    "Poa pratensis",
    "Poa annua",
    "Festuca rubra",
    "Quercus robur",
    "Helianthus annuus",
    "Carnegiea gigantea",
    "Euphorbia milii",
    "Drosera rotundifolia",
    "Nepenthes mirabilis",
    "Taraxacum officinale",
]

OUTPUT = Path(__file__).parent.parent / "tests" / "fixtures" / "opentree_mini_subtree.json"


def main():
    print("Resolving names via TNRS...")
    response = OT.tnrs_match(SPECIES, context_name="Land plants")
    rd = response.response_dict

    name_to_ott: dict[str, int] = {}
    for result in rd.get("results", []):
        name = result.get("name", "")
        matches = result.get("matches", [])
        if not matches:
            print(f"  WARNING: no match for '{name}'")
            continue
        # Take the first (best) match
        best = matches[0]
        taxon = best.get("taxon", {})
        ott_id = taxon.get("ott_id")
        matched_name = taxon.get("unique_name", taxon.get("name", name))
        if ott_id:
            name_to_ott[name] = int(ott_id)
            print(f"  {name} → {matched_name} (ott{ott_id})")
        else:
            print(f"  WARNING: no ott_id for '{name}'")

    ott_ids = list(name_to_ott.values())
    print(f"\nFetching induced subtree for {len(ott_ids)} OTT IDs...")
    tree_response = OT.synth_induced_tree(ott_ids=ott_ids, ignore_unknown_ids=True)
    rd2 = tree_response.response_dict

    broken_nodes = rd2.get("broken", [])
    if isinstance(broken_nodes, dict):
        broken_ids = [int(k.replace("ott", "")) for k in broken_nodes.keys()]
    else:
        broken_ids = [int(str(b).replace("ott", "")) for b in broken_nodes if b]

    subtree = {
        "newick": rd2.get("newick", ""),
        "ott_ids": ott_ids,
        "broken": broken_ids,
        "lineages": {},
        "name_to_ott": name_to_ott,
    }

    print("Fetching lineages via taxon_info...")
    for name, ott_id in name_to_ott.items():
        try:
            info = OT.taxon_info(ott_id=ott_id, include_lineage=True)
            lineage = info.response_dict.get("lineage", [])
            subtree["lineages"][str(ott_id)] = [
                {"rank": e.get("rank", "no rank"), "name": e.get("name", ""), "ott_id": e.get("ott_id", 0)}
                for e in lineage
            ]
            print(f"  {name}: {len(lineage)} lineage nodes")
        except Exception as exc:
            print(f"  WARNING: taxon_info failed for {name} (ott{ott_id}): {exc}")
            subtree["lineages"][str(ott_id)] = []

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(subtree, f, indent=2)

    print(f"\nSaved fixture to {OUTPUT}")
    print(f"  Species resolved: {len(name_to_ott)}")
    print(f"  Broken IDs: {broken_ids}")
    print(f"  Newick length: {len(subtree['newick'])} chars")


if __name__ == "__main__":
    main()
