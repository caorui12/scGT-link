# scGT-Link (demo)

Minimal reproduction demo for **scGT-Link**: scGPT-informed directed graph transformer for supervised GRN link prediction.

This **`main`** branch ships one BEELINE demo task (STRING / hESC / TFs+500) with precomputed embeddings so you can train without downloading the full scGPT weight file.

**Full training code, scGPT checkpoint instructions, HomoloGene scripts, and locked BEELINE configs** are in the same repository on branch **`release`** (formerly the training tree). If you only need the paper recipe:

https://github.com/caorui12/scGT-link/tree/release

See `docs/CODE_AVAILABILITY.md` on that branch for step-by-step scGPT human checkpoint setup.

## What this demo is

1. Prior graph = known training edges, Pearson-weighted (`train_pos_pearson`)
2. Node features = frozen scGPT gene embeddings + Laplacian PE
3. Directed sparse Graph Transformer (L=2, H=4)
4. Concat MLP link head + weighted BCE

Precomputed embeddings are under `scGPT/`. You do **not** need `best_model.pt` or the `scgpt` package to run this demo.

## scGPT checkpoint (when regenerating embeddings)

Demo embeddings were built from the released **scGPT human** checkpoint (cellxgene census) in the [scGPT model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) (`args.json`, `vocab.json`, `best_model.pt`). Full how-to: `docs/CODE_AVAILABILITY.md` on the `release` branch.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install matching torch + DGL, then:
cd src/
python main.py
```

Defaults: `--epochs 200 --lr 3e-4 --d_model 768 --gt_num_layers 2 --gt_hidden_dim 128 --gt_num_heads 4 --seed 42`
