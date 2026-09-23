# scGT-Link

Reproduction demo for **scGT-Link v2**: an scGPT-informed directed graph transformer for supervised gene regulatory network (GRN) link prediction.

This repository contains a **minimal, self-contained** codebase and **one** BEELINE demo dataset (STRING / hESC / TFs+500). Ablation branches and multi-dataset launchers are omitted on purpose.

## What this is

1. Prior graph = train-positive edges only, Pearson-weighted (`train_pos_pearson`)
2. Node features = frozen scGPT gene embeddings + Laplacian PE
3. Directed sparse Graph Transformer (L=2, H=4, hidden=128)
4. Concat MLP link head + weighted BCE

Precomputed scGPT embeddings are included under `scGPT/`. You do **not** need the full scGPT checkpoint or the `scgpt` Python package to train.

## Layout

```
.
  README.md
  requirements.txt
  data/STRING_hESC_TFs500/          # expression + train/val/test splits
  scGPT/STRING_hESC_TFs500/         # precomputed scGPT gene embeddings
    scgpt_gene_emb.pt
  src/                              # model + training scripts
    main.py
    model.py
    encoder.py
    link.py
    prior.py
    input_data.py
  checkpoints/STRING_hESC_TFs500/   # pretrained demo weights
    best.pt
    results.json
```

## Pretrained checkpoint

Trained demo weights are in `checkpoints/STRING_hESC_TFs500/`:

- `best.pt`: best validation AUROC after 200 epochs (seed 42)
- `results.json`: test AUROC ≈ 0.951, AUPRC ≈ 0.413 (CPU run)

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

## How to run

Same style as GT-GRN:

```bash
cd src/
python main.py
```

Defaults: `--epochs 200 --lr 3e-4 --d_model 768 --gt_num_layers 2 --gt_hidden_dim 128 --gt_num_heads 4 --seed 42`

Outputs (repo root):

- `out/STRING_hESC_TFs500/best.pt`
- `out/STRING_hESC_TFs500/results.json`

GPU is used automatically when available. Force CPU with `--cpu`.

