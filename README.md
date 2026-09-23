# scGT-Link

Reproduction demo for **scGT-Link v2**: an scGPT-informed directed graph transformer for supervised gene regulatory network (GRN) link prediction.

This repository contains a **minimal, self-contained** codebase and **one** BEELINE demo dataset (STRING / hESC / TFs+500). Ablation branches and multi-dataset launchers are omitted on purpose.

## What this is

1. Prior graph = train-positive edges only, Pearson-weighted (`train_pos_pearson`)
2. Node features = frozen scGPT gene embeddings + Laplacian PE
3. Directed sparse Graph Transformer (L=2, H=4, hidden=128)
4. Concat MLP link head + weighted BCE

Precomputed `scgpt_gene_emb.pt` is included. You do **not** need the full scGPT checkpoint or the `scgpt` Python package to train.

## Layout

```
.
  train.py                 # entry point
  requirements.txt
  run_smoke.sh             # 2-epoch CPU smoke test
  scgt/                    # slim library (v2 path only)
  data/STRING_hESC_TFs500/
    BL--ExpressionData.csv
    scgpt_gene_emb.pt
    Train_validation_test/{Train,Validation,Test}_set.csv
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a DGL build that matches your PyTorch / CUDA. See https://www.dgl.ai/pages/start.html

CPU example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
pip install numpy pandas scikit-learn
```

## Train

```bash
python train.py
```

Defaults: `--epochs 200 --lr 3e-4 --d_model 768 --gt_num_layers 2 --gt_hidden_dim 128 --gt_num_heads 4 --seed 42`

Outputs:

- `out/STRING_hESC_TFs500/best.pt`
- `out/STRING_hESC_TFs500/results.json` (test AUROC / AUPRC)

GPU is used automatically when available. Force CPU with `--cpu`.



