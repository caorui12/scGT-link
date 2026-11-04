# Code availability

Repository: https://github.com/caorui12/scGT-Link

## How to obtain and use the scGPT checkpoint

Paper embeddings (`scgpt_gene_emb.pt`) were built from the released **scGPT human** checkpoint (cellxgene census human pretrained weights) in the official model zoo:

https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo

### Step-by-step

1. **Clone this repo** and install dependencies (`requirements-grn.txt`, matching DGL, and `pip install scgpt` for embedding extraction).

2. **Download weights**  
   From the model zoo, get the human checkpoint file named **`best_model.pt`**.

3. **Place files** under `scGPT_human/` in the repo root:

   | File | In git? | Role |
   |------|---------|------|
   | `scGPT_human/args.json` | yes | Model config used for extraction |
   | `scGPT_human/vocab.json` | yes | Gene vocabulary |
   | `scGPT_human/best_model.pt` | no (download) | Pretrained weights |

4. **Precompute embeddings** for one BEELINE-style task folder (must contain `BL--ExpressionData.csv`):

   ```bash
   python scripts/precompute_scgpt_gene_emb.py \
     --data_dir /path/to/task \
     --scgpt_model_dir scGPT_human \
     --output /path/to/task/scgpt_gene_emb.pt
   ```

5. **Train** (example matching the locked paper recipe; see also `configs/locked_beeline.json`):

   ```bash
   python train_grn_graph.py \
     --data_dir /path/to/task \
     --split_dir /path/to/splits \
     --epochs 200 \
     --lr 3e-4 \
     --prior_graph train_pos_pearson \
     --expr_encoder scgpt \
     --gt_variant directed \
     --gt_num_layers 2 \
     --gt_num_heads 4 \
     --seed 42 \
     --output_dir out_run
   ```

   Training loads `<data_dir>/scgpt_gene_emb.pt` when present. Only the frozen gene-token embedding table is used (no full scGPT generative forward pass at train time).

## Gene-symbol mapping

- **Default:** exact match into the human scGPT vocabulary, then case-insensitive match; unmatched genes use the shared pad embedding.  
- **HomoloGene sensitivity (Table S2):**  

  ```bash
  python scripts/build_mouse_human_ortholog_map.py --help
  python scripts/precompute_mouse_scgpt_ortholog_emb.py --help
  ```

  `scgpt_loader.load_scgpt_gene_embedding_matrix(..., ortholog_map=...)` applies the map before pad fallback.

## Locked configuration and ablations

- Paper hyperparameters and common ablation CLI flags: **`configs/locked_beeline.json`**  
- Each run writes resolved settings into **`results.json`** under `--output_dir`.

## Implementation notes (Methods alignment)

- Laplacian PE: DGL `lap_pe(..., k=d_model, padding=True)` with `d_model=768`; unused dimensions zero-padded when the gene panel is small.  
- Prior edges carry Pearson weight, direction, and binary `tf2t`; SparseMHA uses learnable `edge_tf2t_scale` (`src/model.py`).  
- Checkpoint selection: best **validation AUROC** under fixed hyperparameters; test metrics for final reporting.
