# Hyperbolic Herbarium — Claude Code Working Memory

> Full spec: @docs/SPEC.md  
> Design decisions log: @docs/DECISIONS.md (create this as you make calls)

## What This Project Is

An **offline-first plant identification system** for field botanists. No LLM at inference time — all outputs come from retrieval and geometric computation. The system identifies herbarium specimens by combining a domain-adapted visual encoder with phylogenetically-structured retrieval, displayed via a Poincaré disk visualization.

**Three novel contributions:**
1. Inference-time phylogenetic graph aggregation over retrieved specimens
2. Iterative clade-conditioned re-retrieval (mirrors how botanists actually work)
3. Geometric uncertainty quantification — Poincaré disk spread IS the uncertainty signal

---

## Stack

- **Python 3.11+**, PyTorch ≥2.2
- **Backbone**: `imageomics/bioclip-2` (ViT-L/14, MIT license) — fine-tune via LoRA, DO NOT train from scratch
- **LoRA**: `peft` library, rank=16, targets `q_proj v_proj out_proj` in all attention blocks
- **Hyperbolic geometry**: `geoopt` library, Poincaré ball model
- **Vector search**: `faiss-gpu` (training/indexing), `faiss-cpu` (inference)
- **Taxonomy**: `opentree` Python package for OpenTree of Life API
- **Data**: DarwinCore Archives via `dwca-reader`, parquet via `pyarrow`
- **Frontend**: TBD (see Design Decision #8) — outputs JSON for visualization

---

## Repository Layout

```
hyperbolic_herbarium/
├── CLAUDE.md                  ← you are here
├── docs/
│   ├── SPEC.md                ← full technical specification
│   └── DECISIONS.md           ← log design decisions here as you make them
├── config/
│   ├── default.yaml           ← all hyperparameters
│   ├── regions.yaml           ← region definitions + Symbiota portal URLs
│   └── backbone.yaml          ← bioclip2 vs bioclip1 variants
├── src/
│   ├── data/                  ← download, parse, filter, balance
│   ├── taxonomy/              ← TNRS resolution, OpenTree, GNN regularizer
│   ├── model/                 ← backbone, LoRA, hyperbolic projection, heads
│   ├── train/                 ← Phase 1 global, Phase 2 regional LoRA
│   ├── index/                 ← FAISS IVF-PQ construction, bundle packing
│   ├── retrieval/             ← query encoding, graph aggregation, re-retrieval
│   ├── viz/                   ← Poincaré disk layout export
│   └── eval/                  ← metrics, stratified evaluation, convergence tests
├── scripts/                   ← shell scripts for end-to-end runs
└── notebooks/                 ← exploration and visualization
```

---

## Implementation Phases (check off as completed)

- [x] **Phase 0** — Skeleton: repo structure, stub files, pyproject.toml, basic pytest suite
- [ ] **Phase 1** — Data pipeline: NAFlora-1M download, DwCA parsing, geographic filtering, capping/splits
- [ ] **Phase 2** — Taxonomy: TNRS resolution with SQLite caching, OpenTree patristic distances
- [ ] **Phase 3** — Model: BioCLIP-2 + LoRA injection, hyperbolic projection, verify forward pass (norm < 1)
- [ ] **Phase 4** — Training (global): smoke test on 10K specimens, 3 epochs
- [ ] **Phase 5** — Index: FAISS IVF-PQ on 10K encodings, verify retrieval
- [ ] **Phase 6** — Retrieval: full pipeline with graph aggregation, test on 10 queries
- [ ] **Phase 7** — Evaluation: precision@k, mistake severity, calibration ECE
- [ ] **Phase 8** — Scale: full California bundle
- [ ] **Phase 9** — Visualization: Poincaré disk JSON export, confidence bars

---

## Critical Rules

- **NEVER generate text at inference time** — no LLM calls in the retrieval or visualization path
- **Poincaré ball constraint**: always clip output norm to <0.99 after hyperbolic projection
- **Taxonomy names**: always resolve through TNRS before training — never use raw strings as class labels
- **Long-tail cap**: 150 images maximum per taxon in training data
- **Test before implement**: write pytest stubs before writing the implementation for each module
- **Flag design decisions**: when you hit an unresolved design decision, don't guess — add it to `docs/DECISIONS.md` and ask the human

---

## Key Numbers (for sanity checks)

| Thing | Expected value |
|---|---|
| NAFlora-1M taxa | 15,501 species |
| NAFlora-1M images | ~1.05M |
| BioCLIP-2 size (fp16) | ~600MB |
| BioCLIP-2 size (int8) | ~300MB |
| LoRA adapter size (rank 16) | ~30–45MB |
| FAISS index per 50K vectors (IVF-PQ) | ~6MB |
| Hyperbolic projection output dim | 512 |
| LoRA training VRAM (ViT-L/14, batch 64) | ~8–14GB |
| LoRA training time per region (A100 40GB) | 3–6 hours |

---

## Open Design Decisions

These are UNRESOLVED — check `docs/DECISIONS.md` for any that have been made. Do NOT hardcode assumptions about these:

1. **Backbone size**: BioCLIP-2 ViT-L/14 (default) vs BioCLIP-1 ViT-B/16 (smaller/faster)
2. **Hyperbolic dim**: 512 (default) vs 128/256
3. **Curvature c**: fixed -1.0 (default) vs learned
4. **Regional boundaries**: ecological provinces (default) vs state lines
5. **CCH2 data in index**: legal question — use NAFlora-1M only until cleared
6. **LoRA rank**: 16 (default) vs 8 or 32
7. **Text fusion**: weighted average (default) vs cross-attention module
8. **Frontend platform**: React+D3 web app vs React Native mobile
9. **Re-retrieval rounds**: 2 (default) vs 1
10. **Graph edges**: genus+family (default) vs full patristic matrix
