# Hyperbolic Herbarium — Technical Specification v0.1

*Offline Regional Plant Identification via Hyperbolic Retrieval, Hierarchical Classification, and Phylogenetic Visualization*

---

## 1. Executive Summary

An offline-first plant identification system for field botanists. Identifies pressed herbarium specimens and field-collected plants by combining a domain-adapted visual encoder with phylogenetically-structured retrieval over regional specimen databases, displayed via a Poincaré disk visualization.

**No generative text calls at inference time.** All explanations come geometrically or from retrieved specimen metadata.

**Primary novelties:**
1. Inference-time graph aggregation over retrieved specimens using phylogenetic edges
2. Iterative clade-conditioned re-retrieval mirroring expert botanical workflow
3. Geometric uncertainty quantification via Poincaré disk spread
4. Regionally-stratified, redistributable training pipeline built on NAFlora-1M

---

## 2. Architecture Overview

### 2.1 Inference Pipeline

```
QUERY
  image (required)  +  habitat text (optional)
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE 1: ENCODING                    │
│  BioCLIP-2 ViT-L/14  (frozen)         │
│    + LoRA adapter (regional, ~35MB)   │
│    + Hyperbolic projection layer      │
│  Output: 512-d Poincaré ball point    │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  STAGE 2: RETRIEVAL  (Round 1)        │
│  FAISS IVF-PQ index (regional)        │
│  Retrieve top-50 candidates           │
│  → Family-level confidence from       │
│    hierarchical classifier heads      │
└───────────────┬───────────────────────┘
                │
                ▼ (if family confidence < threshold)
┌───────────────────────────────────────┐
│  STAGE 2b: ITERATIVE RE-RETRIEVAL     │
│  Re-query family-specific subindex    │
│  (optional round 3 for genus level)   │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  STAGE 3: GRAPH AGGREGATION           │
│  Build small graph over candidates    │
│  Edges = phylogenetic distance        │
│  1-2 rounds message passing           │
│  Output: updated candidate scores     │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  STAGE 4: VISUALIZATION               │
│  Poincaré disk (tree placement)       │
│  Hierarchical confidence bars         │
│  Retrieved specimens panel            │
└───────────────────────────────────────┘
```

### 2.2 Training Pipeline

```
NAFlora-1M (capped + stratified)
  + CCH2 / regional Symbiota DwCA images
  + OpenTree of Life taxonomy
        │
        ▼
DATA PREPARATION
  Taxonomic name resolution (TNRS)
  Geographic filtering by region
  Long-tail capping (150 img/taxon)
  Stratified train/val/test split
        │
        ▼
PHASE 1: GLOBAL BASELINE
  BioCLIP-2 frozen encoder
  Train: hyperbolic projection layer
         hierarchical classifier heads
         taxonomy GNN regularizer
  Dataset: NAFlora-1M capped
        │
        ▼
PHASE 2: REGIONAL LoRA ADAPTERS
  Per-region: train LoRA (rank 16)
  on attention layers of ViT-L/14
  Dataset: regional Symbiota subset
  ~20K–100K images per region
        │
        ▼
PHASE 3: INDEX CONSTRUCTION
  Encode all regional specimens
  Build FAISS IVF-PQ index
  Build family-level subindexes
  Pack regional bundles
```

---

## 3. Dataset

### 3.1 Recommended Dataset: NAFlora-1M + Symbiota Regional Augmentation

**NAFlora-1M** is the primary training dataset.

| Property | Value |
|---|---|
| Total images | 1,050,182 |
| Taxa covered | 15,501 vascular plant species |
| NA species coverage | ≥90% of known North American taxa |
| Image resolution | High-res, color-calibrated with scale bars |
| Labels | Peer-reviewed, hierarchical (family → genus → species) |
| Metadata | Collection locality, date, collector, institution |
| License | Released via DMLR / Kaggle for ML research |
| Long-tail profile | 7–100 images/taxon (capped in competition version) |
| Download | github.com/dpl10/NAFlora-1M |

For regional bundles, pull additional specimens from the corresponding Symbiota portal for regionally-dense coverage of rare endemics underrepresented in NAFlora-1M.

> ⚠️ **Design Decision #5**: CCH2 explicitly prohibits redistribution of raw data. Use NAFlora-1M for public bundles. Use CCH2 for training only. Seek legal clarity before app release.

### 3.2 Regional Splits

| Region | Primary Source | Approx. Specimens | Approx. Taxa |
|---|---|---|---|
| California | CCH2 + NAFlora-1M | 300K–400K | ~6,500 |
| Pacific Northwest | CPNW + NAFlora-1M | 150K–250K | ~5,000 |
| Southwest (AZ/NM) | SEINet + NAFlora-1M | 100K–200K | ~4,500 |
| Southeast | SERNEC + NAFlora-1M | 200K–350K | ~7,000 |
| Intermountain / Rockies | IRHN + NAFlora-1M | 80K–150K | ~4,000 |
| Midwest | Midwest Herbaria + NAFlora-1M | 150K–250K | ~3,500 |
| Mid-Atlantic / Northeast | MAHC + NAFlora-1M | 100K–200K | ~4,000 |

> ⚠️ **Design Decision #4**: Regional boundary definition options: (a) state/province boundaries from DarwinCore `stateProvince` field; (b) EPA Level III Ecoregion polygons; (c) floristic province boundaries. Default: ecological provinces (b). State lines for prototype.

### 3.3 Capping and Balancing

- **Per-taxon cap**: 150 images max. For taxa with >150, select most recent (recency improves label quality).
- **Minimum threshold**: Taxa with <5 images excluded from classifier training, retained in retrieval index.
- **Stratified split**: 70/15/15 by rarity tier (abundant >50, moderate 10–50, rare 5–10), family, and geographic subregion.
- **Phylogenetic test holdout**: Reserve 10% of genera entirely for test-only open-set evaluation.

---

## 4. Model Architecture

### 4.1 Backbone: BioCLIP-2 (Fine-tune, Not From Scratch)

**Strong recommendation: fine-tune BioCLIP-2.** Rationale:
- Already trained on TreeOfLife-200M (214M images, 952K taxa) with hierarchical contrastive learning
- Encodes taxonomic structure into embedding geometry by design
- Training ViT-L/14 from scratch on 1M images would regress from a model that saw 200× more biological imagery
- Demonstrates emergent ecological alignment and intra-species variation preservation

| Parameter | BioCLIP-2 (ViT-L/14) | BioCLIP-1 (ViT-B/16) |
|---|---|---|
| Parameters | ~307M | ~86M |
| Training data | TreeOfLife-200M | TreeOfLife-10M |
| Accuracy gain over CLIP | +30.1% | +17% |
| Model size (fp32) | ~1.2GB | ~340MB |
| Model size (int8) | ~300MB | ~86MB |
| HuggingFace ID | `imageomics/bioclip-2` | `imageomics/bioclip` |
| License | MIT | MIT |

> ⚠️ **Design Decision #1**: BioCLIP-2 (more accurate, larger) vs BioCLIP-1 (smaller, faster). Default: BioCLIP-2. Switch if mobile storage < 500MB total.

### 4.2 LoRA Regional Adapters

| Hyperparameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | ~35MB per adapter for ViT-L/14 |
| LoRA alpha | 32 (2× rank) | Standard scaling |
| Target layers | `q_proj, v_proj, out_proj` in all attention blocks | Max representational coverage |
| Dropout | 0.1 | Regularization for small regional datasets |
| Epochs | 5–10 (early stopping) | |
| Batch size | 64 (accumulate to effective 256) | Fits 40GB A100 |
| Learning rate | 2e-4 with cosine decay | |
| Warmup steps | 200 | |

Adapter size estimate: ~30–45MB per region in fp16.

> ⚠️ **Design Decision #6**: LoRA rank 16 (default) vs 8 (smaller) or 32 (more capacity). Revisit if per-region accuracy on rare species is poor.

### 4.3 Hyperbolic Projection Layer

```python
class HyperbolicProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, curvature=-1.0):
        # Linear projection → normalize to unit ball → scale by tanh
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.c = curvature  # trainable or fixed

    def forward(self, x):
        # Map to Poincaré ball via exponential map at origin
        x = self.linear(x)
        x = F.normalize(x, dim=-1) * 0.9  # keep inside unit ball
        return geoopt.manifolds.PoincareBall(c=abs(self.c)).expmap0(x)
```

Library: `geoopt` (pip install geoopt). Always clip output norm to <0.99.

> ⚠️ **Design Decision #2**: Output dim 512 (default) vs 128/256. Smaller = smaller index, lower retrieval precision for similar species.  
> ⚠️ **Design Decision #3**: Curvature fixed at -1.0 (default) vs learned. Try learned if calibration ECE > 0.08.

### 4.4 Hierarchical Classifier Heads

Three independent linear heads on the Euclidean features (before hyperbolic projection), trained jointly. Fine-to-coarse order following BiLT (AAAI 2025) — species head feeds into genus head.

| Head | Output Dim | Use at Inference |
|---|---|---|
| Family head | ~500 (NA families) | Stage 2 confidence gating |
| Genus head | ~4,000 (NA genera) | Stage 2b re-retrieval trigger |
| Species head | ~15,501 (NA species) | Final confidence bar |

**Loss**: hierarchical cross-entropy with soft label smoothing. Distribute 10–20% probability mass to sibling taxa (same genus), smaller fraction to parent genus/family.

### 4.5 Taxonomy GNN Regularizer (Training Only)

A GNN runs over the taxonomy graph *during training* (not inference) to regularize class prototypes. Nodes = taxa, edges = parent-child in OpenTree. The GNN propagates information across classes so species nearby in the tree have similar embeddings.

**Loss term**: `L_taxonomy = MSE(embed_dist_matrix, patristic_dist_matrix)`

### 4.6 Inference-Time Phylogenetic Graph

```python
def build_retrieval_graph(retrieved_specimens, opentree_distances):
    G = nx.Graph()
    for i, s_i in enumerate(retrieved_specimens):
        for j, s_j in enumerate(retrieved_specimens[i+1:], i+1):
            lca_rank = get_lca_rank(s_i.taxon, s_j.taxon)
            if lca_rank in ('species', 'genus'):
                G.add_edge(i, j, weight=1.0 if lca_rank=='genus' else 0.3)
    return G

def aggregate_scores(query_embed, retrieved, graph, n_rounds=2):
    scores = cosine_similarity(query_embed, [r.embed for r in retrieved])
    for _ in range(n_rounds):
        new_scores = scores.copy()
        for node in graph.nodes:
            neighbors = graph[node]
            neighbor_contribution = sum(
                graph[node][nb]['weight'] * scores[nb] for nb in neighbors
            ) / (len(neighbors) + 1e-6)
            new_scores[node] = 0.7 * scores[node] + 0.3 * neighbor_contribution
        scores = new_scores
    return scores
```

> ⚠️ **Design Decision #10**: Graph edges genus+family only (default) vs full patristic distance matrix.  
> ⚠️ **Design Decision #9**: 2 re-retrieval rounds (default) vs 1. Reduce to 1 if latency > 3s.

---

## 5. Data Pipeline

### 5.1 Directory Structure

```
data/
  raw/
    naflora1m/           # github.com/dpl10/NAFlora-1M
    symbiota/
      cch2/              # California DwCA
      sernec/            # Southeast DwCA
      cpnw/              # Pacific Northwest DwCA
      irhn/              # Intermountain DwCA
  processed/
    regions/
      california/
        specimens.parquet
        train.txt / val.txt / test.txt
      pacific_northwest/ ...
  taxonomy/
    opentree_distances.db   # SQLite pairwise patristic distances
    tnrs_cache.db           # TNRS resolution cache
```

### 5.2 Taxonomy Resolution

Resolve all names to OpenTree OTT IDs via TNRS before training. Cache everything to SQLite. Budget 2–4 hours.

```python
from opentree import OT
response = OT.tnrs_match_names(batch, context_name="Land plants")
```

### 5.3 DwCA Parsing

```python
from dwca.read import DwCAReader
# Fields: occurrenceID, scientificName, decimalLatitude, decimalLongitude,
#         stateProvince, imageURL, reproductiveCondition
```

### 5.4 Geographic Bounding Boxes

```python
REGION_BBOXES = {
    'california':        {'lat': (32.5, 42.0),  'lon': (-124.5, -114.1)},
    'pacific_northwest': {'lat': (42.0, 49.0),  'lon': (-124.6, -116.5)},
    'southwest':         {'lat': (31.3, 37.0),  'lon': (-114.8, -103.0)},
    'southeast':         {'lat': (24.5, 36.5),  'lon': (-91.7,  -75.5)},
    'intermountain':     {'lat': (37.0, 49.0),  'lon': (-116.0, -104.0)},
    'midwest':           {'lat': (36.0, 49.0),  'lon': (-104.0,  -82.0)},
    'mid_atlantic':      {'lat': (36.5, 45.0),  'lon': (-80.5,   -71.0)},
}
```

---

## 6. Training

### 6.1 Compute Estimates

| Phase | Hardware | Time | Cloud Cost (~$2/hr A100) |
|---|---|---|---|
| Data prep | CPU | 4–8h | ~$0 |
| TNRS resolution | CPU + network | 2–4h | ~$0 |
| Phase 1: Global baseline | 1× A100 80GB | 8–16h | $16–32 |
| Phase 2: Regional LoRA (per region) | 1× A100 40GB | 3–6h | $6–12 |
| Phase 2: All 7 regions parallel | 7× A100 40GB | 3–6h wall time | $42–84 total |
| Phase 3: Index construction | 1× A100 40GB | 1–2h/region | $2–4/region |
| Evaluation | 1× A100 40GB | 4–8h | $8–16 |
| **Total (sequential)** | **1× A100** | **~60–100h** | **$120–200** |
| **Total (parallel, 8 GPUs)** | **8× A100** | **~12–20h wall** | **$200–320** |

LoRA VRAM profile: BioCLIP-2 fp16 weights ~600MB + activations ~6–8GB + LoRA states ~2–4GB = **~8–14GB total**. Fits comfortably on 40GB A100. Works on RTX 4090 (24GB) with batch_size=32.

> ⚠️ **Design Decision #1 (budget variant)**: BioCLIP-1 reduces LoRA training time ~3× and VRAM ~4×.

### 6.2 Training Commands

```bash
# Phase 1: Global baseline
python src/train/train_global.py \
    --backbone imageomics/bioclip-2 \
    --dataset data/processed/naflora1m_capped/ \
    --hyperbolic-dim 512 \
    --curvature -1.0 \
    --epochs 15 \
    --batch-size 64 \
    --lr 2e-4 \
    --hierarchical-loss \
    --taxonomy-gnn \
    --output checkpoints/global/

# Phase 2: Regional LoRA
python src/train/train_regional_lora.py \
    --base-checkpoint checkpoints/global/best.pt \
    --region california \
    --lora-rank 16 \
    --lora-alpha 32 \
    --data data/processed/regions/california/ \
    --epochs 8 \
    --output checkpoints/lora/california/
```

### 6.3 Loss Function

Three terms combined with learned weights:

```
L = L_hier + α·L_hyperbolic + β·L_taxonomy

L_hier      = Σ_h λ_h · CrossEntropy(logits_h, labels_h)  [fine-to-coarse]
L_hyperbolic = margin_loss(geodesic_dist(pos), geodesic_dist(neg))
L_taxonomy  = MSE(embed_dist_matrix, patristic_dist_matrix)

Starting values: α=0.1, β=0.05
```

---

## 7. FAISS Index and Regional Bundle

### 7.1 Index Parameters

```python
index = faiss.IndexIVFPQ(
    quantizer, d=512,
    n_clusters=256,    # 256 for <500K vectors
    n_subquantizers=32,
    bits_per_code=8
)
index.nprobe = 32      # cells to visit at query time
# Expected size: ~6MB per 50K vectors
# Expected recall@10: ~95-98%
```

### 7.2 Bundle Format

```
bundles/california/
  manifest.json               # version, taxonomy_version, creation_date, stats
  encoder_base.bin            # int8 quantized BioCLIP-2 (~300MB, shared)
  lora_california.safetensors # LoRA adapter weights (~35MB)
  hyperbolic_proj.pt          # projection layer (~2MB)
  classifier_heads.pt         # hierarchical heads (~8MB)
  faiss_global.bin            # IVF-PQ global index (~50MB)
  faiss_families/             # family sub-indexes (~20MB total)
  specimens.db                # SQLite metadata
  thumbnails/                 # 128×128 JPEGs (~100MB for 100K specimens)
  opentree_subtree.json       # relevant OpenTree subtree

Base model (once):   ~300MB
Per-region bundle:   ~215–320MB
Total (1 region):    ~515–620MB
Total (7 regions):   ~1.8–2.5GB
```

> ⚠️ **Design Decision #8 (mobile)**: For strict mobile storage, consider BioCLIP-1 base (~86MB) or int4 quantization (~150MB).

---

## 8. Retrieval Pipeline

### 8.1 Query Encoding

```python
def encode_query(image, habitat_text=None):
    img_embed = backbone(preprocess(image))           # 768-d ViT features
    if habitat_text:
        text_embed = text_encoder(tokenize(habitat_text))
        combined = 0.8 * img_embed + 0.2 * text_embed  # Design Decision #7
    else:
        combined = img_embed
    poincare_point = hyperbolic_proj(combined)        # 512-d Poincaré point
    family_logits = family_head(combined)
    genus_logits  = genus_head(combined, family_logits)  # BiLT-style conditioning
    species_logits = species_head(combined)
    return { 'euclidean': combined, 'poincare': poincare_point,
             'family_probs': softmax(family_logits), ... }
```

### 8.2 Iterative Re-Retrieval Logic

```python
FAMILY_CONFIDENCE_THRESHOLD = 0.65
GENUS_CONFIDENCE_THRESHOLD  = 0.55

def retrieve(query, bundle):
    # Round 1: global retrieval (top-50)
    candidates = global_faiss_search(query['euclidean'], k=50)
    top_family_prob = query['family_probs'].max()
    
    # Round 2: family-targeted (if confident)
    if top_family_prob > FAMILY_CONFIDENCE_THRESHOLD:
        family_candidates = family_index_search(top_family, k=30)
        candidates = merge_and_dedup(candidates, family_candidates, k=50)
    
    # Graph aggregation
    graph  = build_retrieval_graph(candidates, bundle.opentree_subtree)
    scores = aggregate_scores(query['poincare'], candidates, graph)
    return top_k(zip(candidates, scores), k=10)
```

---

## 9. Visualization

### 9.1 Three UI Views

**View 1 — Poincaré Disk**: Interactive 2D disk showing query position in taxonomic space. Retrieved specimens as colored dots. Clade boundaries as geodesic arcs. Uncertainty = blur radius proportional to spread of retrieved points. Distance from center = how derived/specialized. Angular position = clade membership.

**View 2 — Retrieved Specimens Panel**: Top-10 nearest neighbors as herbarium thumbnails. Species name, collection date, locality, institution, similarity distance (not a percentage).

**View 3 — Hierarchical Confidence Bars**:
```
Family: Onagraceae    ████████████████████░░ 94%
Genus:  Clarkia       ████████████████░░░░░░ 78%
Species: C. gracilis  ██████████░░░░░░░░░░░░ 43%
         C. unguiculata ████████░░░░░░░░░░░░ 38%
         C. cylindrica  ███░░░░░░░░░░░░░░░░░ 11%
```

### 9.2 Open-Set Signal

When uncertainty_score > threshold (query falls in sparse Poincaré region):
```
"Possible undescribed or undigitized taxon.
 Closest known specimens shown.
 Consider collecting a voucher specimen."
```

> ⚠️ **Design Decision #8**: Frontend platform — React+D3 web app vs React Native mobile. Visualization layer outputs JSON, decoupled from frontend choice.

---

## 10. Evaluation Protocol

### 10.1 Metrics

| Metric | Target |
|---|---|
| Retrieval precision@1 | >70% overall, >85% abundant taxa |
| Retrieval precision@5 | >85% overall |
| Family accuracy | >95% |
| Genus accuracy | >85% |
| Species accuracy | >70% |
| Mistake severity (mean LCA height) | <2 (within genus) |
| Calibration ECE | <0.05 |
| Open-set recall (held-out genera flagged) | >60% |
| End-to-end latency (mobile CPU equiv.) | <3 seconds |
| Bundle size per region | <400MB |

### 10.2 Stratified Evaluation

All metrics reported stratified by:
- Rarity tier (abundant / moderate / rare)
- Taxonomic difficulty (same-genus pairs, convergent pairs)
- Geographic subregion
- Phenological stage (where labeled)

**Key scientific evaluation**: improvement in confusion rates between known convergent pairs (Cactaceae vs. Euphorbiaceae succulents, Droseraceae vs. Nepenthaceae carnivores) vs. Euclidean baseline = primary evidence for hyperbolic geometry contribution.

---

## 11. Dependencies

```toml
[project]
dependencies = [
    "torch>=2.2.0",
    "open_clip_torch>=2.24.0",
    "peft>=0.10.0",
    "geoopt>=0.5.0",
    "faiss-gpu>=1.7.4",
    "faiss-cpu>=1.7.4",
    "opentree>=3.0.0",
    "dwca-reader>=0.9.0",
    "networkx>=3.0",
    "umap-learn>=0.5.5",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "Pillow>=10.0",
    "tqdm>=4.65",
    "wandb>=0.16",
    "pytest>=7.0",
]
```

---

## 12. Open Design Decisions

| # | Decision | Default | Alternative | Trigger |
|---|---|---|---|---|
| 1 | Backbone size | BioCLIP-2 ViT-L/14 | BioCLIP-1 ViT-B/16 | Mobile storage <500MB |
| 2 | Hyperbolic dim | 512 | 128 or 256 | Index too large / too slow |
| 3 | Curvature c | Fixed -1.0 | Learned | ECE > 0.08 |
| 4 | Regional boundaries | Ecological provinces | State lines | Data density uneven |
| 5 | CCH2 in index | NAFlora-1M only | CCH2 images | Legal clearance obtained |
| 6 | LoRA rank | 16 | 8 or 32 | Rare species accuracy poor |
| 7 | Text fusion | Weighted average | Cross-attention | Text adds measurable uplift |
| 8 | Frontend platform | React+D3 web | React Native mobile | Deployment context |
| 9 | Re-retrieval rounds | 2 | 1 | Latency >3s |
| 10 | Graph edges | Genus+family | Full patristic matrix | Graph too sparse |

---

## 13. Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| CCH2 redistribution rights | High | NAFlora-1M only in public bundles. CCH2 for training only. |
| LoRA overfits on small regional datasets | Medium | Dropout 0.1, early stopping, max 8 epochs |
| FAISS IVF-PQ recall drops for rare species | Medium | Increase nprobe to 64 at low classifier confidence; exact search fallback for small family sub-indexes |
| Taxonomic name mismatches across sources | Medium | TNRS-resolve all names before training. Flag unresolved for manual review. |
| Hyperbolic projection collapses to boundary | Low | Clip norm to 0.99. Monitor mean norm during training. |
| OpenTree API unavailable | Low | Cache all API results. Rebuild from cache if needed. |
| BioCLIP-2 license change | Low | Pin to specific commit hash. Currently MIT. |

---

## 14. Implementation Phases (for Claude Code)

Build in this order, validate at each step before proceeding:

1. **Phase 0** — Skeleton: repo structure, stub files with TODO comments, pyproject.toml, pytest suite with one import test per module
2. **Phase 1** — Data: NAFlora-1M metadata download (CSV first, not images), DwCA parsing, geographic filtering, capping/splits on 1K-specimen test subset
3. **Phase 2** — Taxonomy: TNRS resolution with SQLite caching, OpenTree patristic distances, build `opentree_distances.db` for California subset
4. **Phase 3** — Model: BioCLIP-2 loading, LoRA injection via `peft`, hyperbolic projection — verify forward pass produces Poincaré point with norm < 1
5. **Phase 4** — Training: smoke test on 10K specimens, 3 epochs, log loss curves
6. **Phase 5** — Index: FAISS IVF-PQ on 10K encodings, verify known specimen retrieval
7. **Phase 6** — Retrieval: full pipeline with graph aggregation, test on 10 query images
8. **Phase 7** — Evaluation: precision@k and mistake severity on validation set
9. **Phase 8** — Scale: full California bundle
10. **Phase 9** — Visualization: Poincaré disk JSON export, confidence bar data

**At each phase**: write tests first, flag unresolved design decisions in `docs/DECISIONS.md`, don't guess on decisions — ask.
