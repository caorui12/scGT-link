# Graph Transformer GRN

基因调控网络（GRN）**边预测**：由表达矩阵构造 **Pearson 相关图**，再用 **GeneExpressionTransformer**（表达 + 图 Laplacian PE）与 **GraphTransformer**（可选基因符号语义向量）得到节点表示，最后 **LinkPredictor** 对训练/验证/测试边打分。

所有命令均在**仓库根目录**执行（`train_grn_graph.py` 会自动把 `src/` 加入 `PYTHONPATH`）。

---

## 环境

```bash
pip install -r requirements-grn.txt
```

- **DGL** 需与 **PyTorch / CUDA** 版本匹配；不要混用 `pip install --user` 与 conda 里的 DGL（脚本会尽量屏蔽 user site，也可用 `GT_GRN_ALLOW_USER_SITE=1` 关闭该行为）。Linux 上请按 [DGL 安装说明](https://www.dgl.ai/pages/start.html) 使用对应的 `data.dgl.ai` wheel。
- 预计算基因符号向量时需：`pip install transformers`（版本见 `requirements-grn.txt` 注释）。

---

## 数据目录约定

**`--data_dir`**（例如 `data/hESC/TFs+500/`）下，训练脚本**必须**存在：

- **`BL--ExpressionData.csv`**：行为基因、列为细胞；**行索引 = 基因名**（与图中节点顺序、预计算符号向量顺序一致）。

同目录下其他文件（如 `TF.csv`、`Label.csv`）可用于数据分析或构建划分；**当前 `train_grn_graph.py` 只读取上述表达矩阵**。

**`--split_dir`**（例如 `data/Train_validation_test/hESC_500/`）内需有：

- `Train_set.csv`、`Validation_set.csv`、`Test_set.csv`（列含 TF、Target、Label 等，见 `grn_data.load_edge_split`）。

---

## 主入口：`train_grn_graph.py`

训练并按 **验证集 AUROC** 保存 `best.pt`，最后在测试集上评估并写入 `results.json`。

```bash
python train_grn_graph.py \
  --data_dir data/hESC/TFs+500 \
  --split_dir data/Train_validation_test/hESC_500 \
  --epochs 100 \
  --seed 42 \
  --output_dir out_grn_gt
```

### 常用参数

| 参数 | 含义 |
|------|------|
| `--data_dir` | 含 `BL--ExpressionData.csv` 的数据目录 |
| `--split_dir` | 训练/验证/测试边 CSV 所在目录 |
| `--output_dir` | 输出目录（`best.pt`、`results.json`） |
| `--gene_bert_emb` | 预计算基因符号 `.pt`；**留空或不传** = 语义分支全零（消融基线） |
| `--epochs` | 训练轮数 |
| `--lr` | 学习率（默认 `1e-4`） |
| `--d_model` | 表达编码与 GT 输入维、lap PE 维（默认 128） |
| `--top_k` | 相关图每节点保留的 Top-K 邻居（默认 20） |
| `--no_cuda` | 强制 CPU |
| `--test_prob_threshold` | 测试集上 P/R/F1/acc 的二分类阈值（默认 0.5） |

带 **GENA / BioBERT 等符号向量** 时示例：

```bash
python train_grn_graph.py \
  --data_dir data/hESC/TFs+500 \
  --split_dir data/Train_validation_test/hESC_500 \
  --epochs 100 --seed 42 \
  --gene_bert_emb data/hESC/TFs+500/gene_symbols_gena.pt \
  --output_dir out_grn_gt_hESC_500_gena
```

---

## 脚本：`scripts/encode_gene_symbols.py`

离线把 **基因名**（与表达矩阵**行顺序一致**）编码为 `(n_genes, hidden)`，供 `--gene_bert_emb` 加载。

```bash
# BioBERT（默认后端）
python scripts/encode_gene_symbols.py --backend biobert \
  --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv \
  --output data/hESC/TFs+500/gene_symbols_biobert.pt

# GENA-LM（DNA 预训练；符号文本为探索性用法）
python scripts/encode_gene_symbols.py --backend gena-lm \
  --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv \
  --output data/hESC/TFs+500/gene_symbols_gena.pt

# 任意 HuggingFace 模型
python scripts/encode_gene_symbols.py --backend hf \
  --model_name <org/model> \
  --trust_remote_code \
  --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv \
  --output data/hESC/TFs+500/gene_symbols_custom.pt
```

输出：

- `*.pt`：`embeddings`、`gene_names`、`bert_dim`、`model_name`、`encoder_backend` 等。
- `*.manifest.json`：同上不含大矩阵，便于检查版本与基因顺序哈希。

可选：`--text_template`（默认 `{symbol}`）、`--max_length`、`--batch_size`、`--device`。

### 兼容入口：`scripts/encode_gene_names_biobert.py`

等价于 `encode_gene_symbols.py --backend biobert ...`，仅保留旧命令习惯；stderr 会提示迁移。

---

## 脚本：`scripts/inspect_gene_symbol_tokenization.py`

检查 tokenizer：**UNK 比例、是否被 `max_length` 截断**等，与 `encode_gene_symbols.py` 使用同一套 `--backend` / `--model_name` 逻辑。

```bash
python scripts/inspect_gene_symbol_tokenization.py --backend biobert \
  --expression_csv data/hESC/TFs+500/BL--ExpressionData.csv

python scripts/inspect_gene_symbol_tokenization.py \
  --manifest_json data/hESC/TFs+500/gene_symbols_gena.manifest.json \
  --list_unk
```

`--manifest_json` 时会尽量读取 manifest 中的 `model_name`、`trust_remote_code`、`max_length`。

---

## 代码结构（简要）

| 路径 | 作用 |
|------|------|
| `train_grn_graph.py` | 训练、验证选优、测试评估 |
| `grn_data.py` | 表达矩阵与边划分加载；`load_gene_bert_embeddings` 读 `.pt` |
| `correlation_graph.py` | Pearson 相关 → DGL 图 |
| `grn_model.py` | `GeneExpressionTransformer` |
| `src/model.py` | `GraphTransformer`（融合 `semantic_feat`） |
| `src/utils.py` | `LinkPredictor`、`evaluate` |

---

## 输出

在 `--output_dir` 下：

- **`best.pt`**：验证 AUROC 最优 checkpoint（含 `seq_model` / `gt_model` / `predictor` / 可选 `bert_proj`）。
- **`results.json`**：测试 AUROC/AUPRC 及按 epoch 的 `history` 等。

恢复训练或单独评测时，需与训练时相同的 `--gene_bert_emb`（若 checkpoint 含 `bert_proj`）。
