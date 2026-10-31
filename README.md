# scGT-Link v2

Official implementation of **scGT-Link v2** (*scGPT-informed Graph Transformer for regulatory **Link** prediction*).

**v2 change:** gene-expression profiles are **not** used as encoder node features (`--expr_encoder scgpt`). Nodes use **scGPT** embeddings + **LapPE**; the expression matrix is still loaded when building Pearson-weighted priors / LapPE graphs (`train_pos_pearson`, `correlation`, etc.). Keep `--expr_encoder scgpt_concat` for the v1 dual-encoder path.

PyTorch pipeline for **gene regulatory network edge prediction**. It builds a **prior graph** from single-cell expression (and optional gold / train-positive edges), encodes genes with **scGPT** (+ LapPE), runs a **sparse graph transformer backbone**, and scores TF–target pairs with an MLP decoder.

Run all commands from the **repository root** (`train_grn_graph.py` adds `src/` to the import path).

See **`VERSION`**: **scGT-Link v2** — removes the expression encoder branch; default `--expr_encoder scgpt`.

---

## Installation

```bash
pip install -r requirements-grn.txt
```

**DGL** must match your **PyTorch / CUDA** build. Follow the official [DGL install guide](https://www.dgl.ai/pages/start.html).

---

## Data layout

Under **`--data_dir`**:

- **`BL--ExpressionData.csv`** — genes × cells; row index must be **gene symbols** aligned with gold edges and embeddings.

Under **`--split_dir`**:

- **`Train_set.csv`**, **`Validation_set.csv`**, **`Test_set.csv`** — edges with columns **`TF`**, **`Target`** (integer row indices into the expression matrix), **`Label`** (`0` / `1`).

Gold edges for the prior graph (**`BL--network.csv`** by default mapping) must use **gene symbols** matching the expression row names.

---

## Training

```bash
python train_grn_graph.py \
  --data_dir data/hESC/TFs+500 \
  --split_dir data/Train_validation_test/hESC_500 \
  --epochs 50 \
  --output_dir out_run
```

Defaults (**scGT-Link v2**): **`prior_graph=train_pos_pearson`**, **`expr_encoder=scgpt`**, **`gt_variant=directed`**, **`gt_num_layers=2`**, **`gt_ffn_act=swiglu`**, and remap **`scgpt_emb_pt`** to `<data_dir>/scgpt_gene_emb.pt`. Use `--gt_ffn_act relu` to reproduce the historical baseline. Precompute embeddings once:

```bash
python scripts/precompute_scgpt_gene_emb.py \
  --data_dir data/hESC/TFs+500 \
  --scgpt_model_dir scGPT_human \
  --output data/hESC/TFs+500/scgpt_gene_emb.pt
```

**scGPT checkpoint:** use the released scGPT **human** model (`args.json` + `vocab.json` are tracked under `scGPT_human/`; download `best_model.pt` from the [scGPT model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo)). Details, HomoloGene remapping, and locked BEELINE / ablation configs are in **`docs/CODE_AVAILABILITY.md`** and **`configs/locked_v2_beeline.json`**.

Registered local datasets (**`--dataset`**) and **`--train_all`** are described in the docstring at the top of **`train_grn_graph.py`**.

### Revision ablations (Fig. 5 / Table S5)

Unconfounded single-factor settings used in the revision (examples; full recipe in `configs/locked_v2_beeline.json`):

| Setting | Example flags |
|---------|----------------|
| Gene-permuted scGPT | `--shuffle_scgpt` |
| Constant edge weight | `--prior_graph train_pos_uniform` |
| Shuffled Pearson weights | `--prior_graph train_pos_shuffled_w` |
| Density-matched random prior | `--prior_graph random` |
| Degree-preserving rewire | `--prior_graph degree_rewire` |

Mouse HomoloGene remapping scripts: `scripts/build_mouse_human_ortholog_map.py`, `scripts/precompute_mouse_scgpt_ortholog_emb.py`.

### Inference

- **`scripts/infer_grn_checkpoint.py`** — load **`best.pt`** and score edges from the split CSVs (Train / Val / Test).
- **`scripts/infer_grn_predict_network.py`** — same checkpoint + prior rebuild, but scores a **large candidate edge set** (TF × full expression panel, or directed rows from **`BL--network.csv`**). See the script docstring for **`--candidate_mode`**, **`--exclude_gold_rows`**, and **`--top_k_per_tf`**.

---

## Project layout

| Path | Role |
|------|------|
| `train_grn_graph.py` | Training, validation checkpointing, test metrics |
| `grn_data.py` | Expression and edge-split loading |
| `grn_model.py` | Gene encoders (`GeneExpressionTransformer`, `ScGPTExprConcatEncoder`, …) |
| `gold_network_graph.py`, `correlation_graph.py` | Prior-graph construction |
| `src/model.py` | Sparse `GraphTransformer` |
| `src/utils.py` | `LinkPredictor`, metrics |
| `scripts/infer_grn_checkpoint.py`, `scripts/infer_grn_predict_network.py` | Checkpoint inference on splits vs. full candidate panels |
| `scripts/precompute_scgpt_gene_emb.py` | Build `scgpt_gene_emb.pt` from the scGPT human checkpoint |
| `scripts/build_mouse_human_ortholog_map.py` | HomoloGene mouse→human map (Table S2 sensitivity) |
| `configs/locked_v2_beeline.json`, `docs/CODE_AVAILABILITY.md` | Locked recipe, checkpoint ID, ablation flags |
| `scripts/split_grn_train_val_test.py` | Build Train / Val / Test CSVs |
| `benchmarks/statistical_baselines/` | PCC / MI / GRNBoost2 baselines for comparison |

---

## Outputs

Inside **`--output_dir`**:

- **`best.pt`** — checkpoint with best validation AUROC (`seq_model`, `gt_model`, `predictor`).
- **`results.json`** — test AUROC / AUPRC and training history.

Hyperparameter sweeps use **`scripts/tune_grn_hparams.py`** (see script header).
