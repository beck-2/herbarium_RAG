# Hyperbolic Herbarium — Design Decisions Log

Format: `DECISION-N | Status | Date | Choice | Rationale`

---

## DECISION-1 | OPEN | 2026-02-23 | BioCLIP-2 ViT-L/14 (default)

**Question**: BioCLIP-2 ViT-L/14 (~300MB int8) vs BioCLIP-1 ViT-B/16 (~86MB int8)?

**Default**: BioCLIP-2 ViT-L/14.

**Flag**: Target platform is React Native mobile. BioCLIP-2 int8 + 1 region bundle ≈ 515–620MB, which exceeds the 500MB threshold noted in the spec. Revisit once we have real bundle size measurements from Phase 5. If total exceeds 500MB on target device, switch to BioCLIP-1.

**Config key**: `backbone.model_id` in `config/backbone.yaml`.

---

## DECISION-2 | OPEN | 2026-02-23 | Hyperbolic dim 512 (default)

**Question**: Output dim 512 vs 128 or 256?

**Default**: 512. Smaller dim → smaller FAISS index but lower retrieval precision for similar species.

**Trigger**: Revisit if index exceeds 400MB per-region bundle budget or retrieval latency > 3s on mobile CPU.

**Config key**: `model.hyperbolic_dim` in `config/default.yaml`.

---

## DECISION-3 | OPEN | 2026-02-23 | Fixed curvature -1.0 (default)

**Question**: Fixed curvature c = -1.0 vs learned curvature?

**Default**: Fixed -1.0. Simpler, no risk of curvature collapse during training.

**Trigger**: Switch to learned if calibration ECE > 0.08 after Phase 7 evaluation.

**Config key**: `model.curvature` and `model.learn_curvature` in `config/default.yaml`.

---

## DECISION-4 | DECIDED | 2026-02-23 | State lines for prototype

**Question**: Regional boundaries via EPA Level III Ecoregion polygons vs state/province lines vs floristic provinces?

**Decision**: State lines (DarwinCore `stateProvince` field) for prototype. Bounding boxes defined in `config/regions.yaml`.

**Rationale**: Simpler to implement and debug in prototype phase. Spec recommends this for initial development. Switch to ecological provinces when data density analysis is available.

---

## DECISION-5 | OPEN | 2026-02-23 | NAFlora-1M only in public bundles

**Question**: Include CCH2 images in the FAISS retrieval index for public distribution?

**Default**: NAFlora-1M only in public bundles. CCH2 for training only.

**Status**: Awaiting legal clarity on CCH2 redistribution rights before app release.

---

## DECISION-6 | OPEN | 2026-02-23 | LoRA rank 16 (default)

**Question**: LoRA rank 16 vs 8 (smaller adapter) vs 32 (more capacity)?

**Default**: rank=16 → ~35MB adapter per region.

**Trigger**: Revisit if per-region accuracy on rare species (5–10 images) is poor after Phase 7 evaluation.

**Config key**: `lora.rank` in `config/default.yaml`.

---

## DECISION-7 | OPEN | 2026-02-23 | Text fusion weighted average (default)

**Question**: Habitat text fusion via weighted average (0.8 image + 0.2 text) vs cross-attention module?

**Default**: Weighted average. Zero extra parameters, easy to ablate.

**Trigger**: Switch to cross-attention if text fusion adds measurable Top-1 uplift (>2%) in ablation study.

**Config key**: `model.text_fusion_weight` in `config/default.yaml`.

---

## DECISION-8 | DECIDED | 2026-02-23 | React Native mobile

**Question**: Frontend platform — React+D3 web app vs React Native mobile app?

**Decision**: React Native mobile.

**Rationale**: Matches field botanist use case (offline, field conditions). The `src/viz/` module outputs JSON and is decoupled from frontend. React Native scaffold will be added at Phase 9 once the backend JSON contract is stable.

**Implication**: Strict storage budget. See DECISION-1 for backbone implications.

---

## DECISION-9 | OPEN | 2026-02-23 | 2 re-retrieval rounds (default)

**Question**: 2 rounds of clade-conditioned re-retrieval vs 1 round?

**Default**: 2 rounds.

**Trigger**: Reduce to 1 if end-to-end latency exceeds 3s on mobile CPU equivalent. Measure in Phase 6.

**Config key**: `retrieval.n_rereval_rounds` in `config/default.yaml`.

---

## DECISION-10 | OPEN | 2026-02-23 | Graph edges genus+family only (default)

**Question**: Build phylogenetic retrieval graph with genus+family edges only vs full patristic distance matrix?

**Default**: Genus+family edges. Full patristic matrix is expensive to compute and store.

**Trigger**: Switch to full patristic if genus+family graph is too sparse (< 5 edges per 50-candidate graph on average).

**Config key**: `retrieval.graph_edge_mode` in `config/default.yaml`.

---

## DECISION-11 | DECIDED | 2026-02-23 | NAFlora-1M metadata is JSON from Kaggle

**Question**: NAFlora-1M metadata as CSV (per original stub) vs actual dataset format?

**Decision**: Parse **JSON** metadata. The dataset is distributed via Kaggle (herbarium-2022-fgvc9) as `train_metadata.json` and `test_metadata.json` (COCO-style: images, annotations, categories, genera). We map these to the canonical schema; `latitude`, `longitude`, `state_province` are null for NAFlora (not in the published JSON). Download supports either Kaggle API (optional) or user-placed JSON in the data dir.

**Code**: `parse.parse_naflora_json()` and `parse.parse_naflora_csv()` accepts a path to a .json file or directory containing train/test metadata.

---

## DECISION-12 | OPEN | 2026-02-23 | NAFlora Kaggle metadata vs spec “Retrieved Specimens Panel”

**Spec §3.1** says NAFlora has “Metadata: Collection locality, date, collector, institution.” **Spec §9.1 View 2** says the Retrieved Specimens Panel shows “Species name, collection date, locality, institution, similarity distance.”

**Reality**: The Kaggle competition metadata (train_metadata.json / test_metadata.json) uses a COCO-style schema. It provides:
- **Present**: scientificName, family, genus, species (categories); image_id, file_name; institution_id (annotations) and institutions (collectionCode). Optional “distances” array for pairwise evolutionary distance between genera.
- **Absent in Kaggle JSON**: Collection locality (decimalLatitude, decimalLongitude, stateProvince), collection date (eventDate), collector.

**Implication**: NAFlora is **sufficient** for the core RAG mechanics (hierarchical labels, family/genus subindexes, phylogenetic graph via OpenTree or “distances”, confidence bars). It is **not sufficient** for the full panel as written (date, locality). For regional bundles we use Symbiota/DwCA metadata (which has locality/date) for specimens from those portals; for NAFlora-only specimens we show species, institution, similarity and leave date/locality blank unless a richer NAFlora source becomes available.

---

## DECISION-14 | OPEN | 2026-02-23 | We do not have any DwCA or NAFlora data on disk

**DwCA**: The repo has **no** DwCA dataset. Code can *parse* DwCA (when you provide a path) and *try* to download from Symbiota portal URLs (`download_symbiota_dwca`). You must either run a DwCA export from a Symbiota portal (e.g. CCH2) and point the code at it, or obtain a DwCA from another source.

**NAFlora**: The repo has **no** NAFlora metadata or images. Two formats are supported in code:
- **Kaggle** (herbarium-2022-fgvc9): `train_metadata.json` / `test_metadata.json` — large COCO-style JSON with `images[]`, `annotations[]`, `categories[]` (per-image metadata). You must download via Kaggle (API or manual).
- **GitHub** NAFlora-mini: `h22_miniv1_train.tsv` / `h22_miniv1_val.tsv` — tab-separated, one row per image, columns include genus_id, institution_id, category_id, image_id, file_id, scientificName, family, genus, species, authors. No JSON on GitHub; only TSV. If the only JSON you see is “number of images per species,” that is a different (summary) file and is **not enough** for the pipeline — we need per-image metadata (which image, which species). Use either Kaggle’s full JSON or the GitHub TSV.

---

## DECISION-13 | DECIDED | 2026-02-23 | How we divide by region (geography only on DwCA)

**Question**: How do we get “regional” data when NAFlora Kaggle JSON has no locality/coordinates?

**Decision**: Divide by region **only** using data that has geography:

- **Phase 1 (global baseline)**: Train on **NAFlora-1M** only (all ~1M), capped and stratified. No regional split; NAFlora is one global dataset.
- **Phase 2 (regional LoRA) & Phase 3 (index/bundles)**: Regional dataset = **Symbiota/DwCA only** for that region. Filter DwCA by the region’s bbox (lat/lon) or by `stateProvince` (per DECISION-4). So e.g. “California” = CCH2 (and any other DwCA) records whose coordinates fall inside the California bbox. NAFlora is **not** filtered by region (we don’t have coordinates for it in the Kaggle JSON).

So the table in §3.2 (“California | CCH2 + NAFlora-1M”) is read as: **sources** for the region are CCH2 (regional DwCA) plus NAFlora for taxonomy/global model; the **regional training/index specimens** are the DwCA subset in the bbox. If a NAFlora release with locality ever appears, we can add “NAFlora in California bbox” to the regional pool.
