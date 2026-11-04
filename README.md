# scGT-Link

Official implementation of **scGT-Link** (*scGPT-informed Graph Transformer for regulatory **Link** prediction*).

Frozen **scGPT** gene embeddings + **Laplacian PE**, a directed sparse graph transformer on a Pearson-weighted prior built from known **training** interactions, and an MLP link scorer. Expression is used to weight prior edges (`train_pos_pearson`); it is **not** the default node encoder (`--expr_encoder scgpt`). Use `--expr_encoder scgpt_concat` only if you want the older dual-encoder path.

Run all commands from the **repository root**.

---

## Installation

```bash
pip install -r requirements-grn.txt
```

Install a **DGL** build that matches your PyTorch / CUDA: https://www.dgl.ai/pages/start.html  
For embedding extraction you also need the `scgpt` package (`pip install scgpt`) when running `scripts/precompute_scgpt_gene_emb.py`.

---

## scGPT checkpoint (how to use)

We use the released **scGPT human** weights (cellxgene census human) from the official model zoo:

https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo

### 1. Prepare the checkpoint folder

This repo already tracks:

- `scGPT_human/args.json`
- `scGPT_human/vocab.json`

Download **`best_model.pt`** from the model zoo above and place it here:

```text
scGPT_human/
  args.json      (tracked)
  vocab.json     (tracked)
  best_model.pt  (you download; large, not in git)
```

### 2. Build per-task gene embeddings

```bash
python scripts/precompute_scgpt_gene_emb.py \
  --data_dir data/hESC/TFs+500 \
  --scgpt_model_dir scGPT_human \
  --output data/hESC/TFs+500/scgpt_gene_emb.pt
```

- Input expression file: `data_dir/BL--ExpressionData.csv` (gene symbols as row index).  
- Matching: exact symbol, then case-insensitive; missing genes get the shared **pad** embedding.  
- Output `scgpt_gene_emb.pt` is what training loads by default.

Mouse HomoloGene remapping (optional sensitivity):  
`scripts/build_mouse_human_ortholog_map.py` → `scripts/precompute_mouse_scgpt_ortholog_emb.py`  
(see `docs/CODE_AVAILABILITY.md`).

More detail: **`docs/CODE_AVAILABILITY.md`**.

---

## Data layout

Under **`--data_dir`**:

- **`BL--ExpressionData.csv`** — genes × cells; row index = gene symbols.

Under **`--split_dir`**:

- **`Train_set.csv`**, **`Validation_set.csv`**, **`Test_set.csv`** — columns **`TF`**, **`Target`** (integer row indices into the expression matrix), **`Label`** (`0` / `1`).

---

## Training (paper default)

Locked BEELINE recipe is summarized in **`configs/locked_beeline.json`** (200 epochs, lr `3e-4`, `d_model=768`, `L=2`, `H=4`, `prior_graph=train_pos_pearson`, `expr_encoder=scgpt`).

Minimal example (after embeddings exist):

```bash
python train_grn_graph.py \
  --data_dir data/hESC/TFs+500 \
  --split_dir data/Train_validation_test/hESC_500 \
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

Defaults also remap **`scgpt_emb_pt`** to `<data_dir>/scgpt_gene_emb.pt` when present.  
Under a fixed hyperparameter setting, the run keeps the checkpoint with best **validation AUROC**; test metrics are for final reporting only.

Registered datasets (`--dataset`) and batch helpers: see the docstring at the top of `train_grn_graph.py`.

### Ablation settings (Fig. 5 / Table S5)

| Setting | Example flags |
|---------|----------------|
| Gene-permuted scGPT | `--shuffle_scgpt` |
| Constant edge weight | `--prior_graph train_pos_uniform` |
| Shuffled Pearson weights | `--prior_graph train_pos_shuffled_w` |
| Density-matched random prior | `--prior_graph random` |
| Degree-preserving rewire | `--prior_graph degree_rewire` |
| No message passing (keep LapPE) | `--gt_num_layers 0` |
| No LapPE (keep GT) | `--no_lappe` |

Full flag list: `configs/locked_beeline.json`.

### Inference

- `scripts/infer_grn_checkpoint.py` — score Train / Val / Test edges from `best.pt`
- `scripts/infer_grn_predict_network.py` — score a large candidate panel (see script docstring)

---

## Project layout

| Path | Role |
|------|------|
| `train_grn_graph.py` | Training |
| `grn_data.py` | Expression and splits |
| `gold_network_graph.py`, `correlation_graph.py` | Prior graphs |
| `src/model.py` | Sparse `GraphTransformer` (incl. `tf2t`) |
| `src/utils.py` | `LinkPredictor`, metrics |
| `scripts/precompute_scgpt_gene_emb.py` | Build `scgpt_gene_emb.pt` |
| `scGPT_human/` | Checkpoint config + vocab (`best_model.pt` download) |
| `configs/locked_beeline.json` | Paper hyperparameters |
| `docs/CODE_AVAILABILITY.md` | Checkpoint, mapping, configs |

---

## Outputs

Under **`--output_dir`**:

- `best.pt` — best validation AUROC weights  
- `results.json` — test AUROC / AUPRC and run settings  
