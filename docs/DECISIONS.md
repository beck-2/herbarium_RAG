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
