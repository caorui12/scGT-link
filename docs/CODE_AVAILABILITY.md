# Code availability (revision)

This document matches the Code Availability response for Briefings in Bioinformatics (BIB-26-2493).

## Repository

- Public demo (default branch `main`): https://github.com/caorui12/scGT-Link  
- Full training / ablation code (branch `scGT-link_v2`): https://github.com/caorui12/scGT-Link/tree/scGT-link_v2  

## scGPT checkpoint used to build `scgpt_gene_emb.pt`

Embeddings were extracted from the released **scGPT human** checkpoint (cellxgene census human pretrained weights):

| File | Role |
|------|------|
| `scGPT_human/args.json` | Model config used for embedding extraction (tracked) |
| `scGPT_human/vocab.json` | Gene vocabulary (tracked) |
| `scGPT_human/best_model.pt` | Weight file (download separately; not in git) |

Download the weight file from the official scGPT model zoo:

https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo

Place `best_model.pt` next to the tracked `args.json` / `vocab.json`, then:

```bash
python scripts/precompute_scgpt_gene_emb.py \
  --data_dir /path/to/task \
  --scgpt_model_dir scGPT_human \
  --output /path/to/task/scgpt_gene_emb.pt
```

Only the gene-token embedding table is used (frozen). Genes absent from the vocabulary receive the shared pad embedding.

## Gene-symbol mapping

- **Default (manuscript):** exact match, then case-insensitive match, into the human scGPT vocabulary (covers human panels and same-symbol mouse genes).  
- **HomoloGene sensitivity (Table S2):** remap vocabulary misses with NCBI HomoloGene via MyGene.info:

```bash
python scripts/build_mouse_human_ortholog_map.py ...
python scripts/precompute_mouse_scgpt_ortholog_emb.py ...
```

See script docstrings for arguments. `scgpt_loader.load_scgpt_gene_embedding_matrix(..., ortholog_map=...)` applies the map before pad fallback.

## Locked run configuration

See `configs/locked_v2_beeline.json` for the 200-epoch BEELINE recipe (lr, depth, heads, prior, seeds) and CLI flags for the unconfounded ablation settings in Figure 5 / Table S5.

Each training run writes `results.json` (and related logs) under `--output_dir`, including the resolved hyperparameters used for that run.

## Implementation notes aligned with Methods

- Laplacian PE: DGL `lap_pe(..., k=d_model, padding=True)` with `d_model=768`; unused dimensions are zero-padded when `n_genes` is small.  
- Prior edges store Pearson weight, direction, and binary `tf2t`; SparseMHA includes a learnable `tf2t` term (`edge_tf2t_scale` in `src/model.py`).  
- Checkpoint selection under a fixed hyperparameter setting uses the best **validation AUROC**; reported benchmark scores are on the held-out test split.
