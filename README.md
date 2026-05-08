# Graph Transformer GRN

基因调控网络（GRN）**边预测**：由表达矩阵构造先验图（如 Pearson 相关或参考网络），再用 **GeneExpressionTransformer**（表达 + 图 Laplacian PE）或 **scGPT** 编码与 **GraphTransformer** 得到节点表示，最后 **LinkPredictor** 对训练/验证/测试边打分。

所有命令均在**仓库根目录**执行（`train_grn_graph.py` 会自动把 `src/` 加入 `PYTHONPATH`）。

---

## 环境

```bash
pip install -r requirements-grn.txt
```

- **DGL** 需与 **PyTorch / CUDA** 版本匹配；不要混用 `pip install --user` 与 conda 里的 DGL（脚本会尽量屏蔽 user site，也可用 `GT_GRN_ALLOW_USER_SITE=1` 关闭该行为）。Linux 上请按 [DGL 安装说明](https://www.dgl.ai/pages/start.html) 使用对应的 `data.dgl.ai` wheel。
---

## 数据目录约定

**`--data_dir`**（例如 `data/hESC/TFs+500/`）下，训练脚本**必须**存在：

- **`BL--ExpressionData.csv`**：行为基因、列为细胞；**行索引 = 基因名**（与图中节点顺序、预计算符号向量顺序一致）。

同目录下其他文件（如 `TF.csv`、`Label.csv`）可用于数据分析或构建划分；**当前 `train_grn_graph.py` 只读取上述表达矩阵**。

**`--split_dir`**（例如 `data/Train_validation_test/hESC_500/`）内需有：

- `Train_set.csv`、`Validation_set.csv`、`Test_set.csv`（列含 TF、Target、Label 等，见 `grn_data.load_edge_split`）。

**参考网络**（``--gold_network_file``，用于 ``prior_graph=gold`` / ``gold_pearson``）：边表两列端点须为**基因符号**（与表达矩阵行名一致，大小写不敏感），不支持用整数行号代替。

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
| `--expr_encoder` | `transformer` / `scgpt` / `scgpt_concat`（scGPT 需 `--scgpt_emb_pt` 或 `--scgpt_model_dir`） |
| `--epochs` | 训练轮数 |
| `--lr` | 学习率（默认 `1e-4`） |
| `--d_model` | 表达编码与 GT 输入维、lap PE 维（默认 128） |
| `--top_k` | 相关图每节点保留的 Top-K 邻居（默认 20） |
| `--no_cuda` | 强制 CPU |
| `--test_prob_threshold` | 测试集上 P/R/F1/acc 的二分类阈值（默认 0.5） |

带 **scGPT** 预计算基因嵌入时（见 `scripts/precompute_scgpt_gene_emb.py`）：

```bash
python train_grn_graph.py \
  --data_dir data/hESC/TFs+500 \
  --split_dir data/Train_validation_test/hESC_500 \
  --expr_encoder scgpt_concat \
  --scgpt_emb_pt data/hESC/TFs+500/scgpt_gene_emb.pt \
  --epochs 100 --seed 42 \
  --output_dir out_grn_scgpt
```

---

## 代码结构（简要）

| 路径 | 作用 |
|------|------|
| `train_grn_graph.py` | 训练、验证选优、测试评估 |
| `grn_data.py` | 表达矩阵与边划分加载 |
| `correlation_graph.py` | Pearson 相关 → DGL 图 |
| `grn_model.py` | `GeneExpressionTransformer` |
| `src/model.py` | `GraphTransformer` |
| `src/utils.py` | `LinkPredictor`、`evaluate` |
| `scripts/precompute_scgpt_gene_emb.py` | 从 scGPT checkpoint 导出每基因向量 → `--scgpt_emb_pt` |
| `scripts/tune_grn_hparams.py` | 超参网格：多次子进程调用 `train_grn_graph.py`，汇总 `summary.json` |
| `scripts/tune_fixed_scgpt_gold_pearson.example.json` | 与 `--fixed_json` 配合的示例（scGPT + gold_pearson） |

**微调 / 扫参**（见脚本顶部说明）：

```bash
python scripts/tune_grn_hparams.py --epochs 80 --seed 42 \\
  --fixed_json scripts/tune_fixed_scgpt_gold_pearson.example.json \\
  --output_root out_tune_run
```

微调脚本**默认只使用 hESC_500** 路径；与其它数据请同时传入 ``--data_dir`` 与 ``--split_dir``。默认网格用**显式合法的 (d_model, nhead) 对**与 ``lr``、``num_layers``、``gt_*`` 组合，**不含** ``top_k``。自定义请 ``--grid_json``（仍会过滤 ``d_model%nhead`` 等非法组合）。

---

## 输出

在 `--output_dir` 下：

- **`best.pt`**：验证 AUROC 最优 checkpoint（含 `seq_model` / `gt_model` / `predictor`）。
- **`results.json`**：测试 AUROC/AUPRC 及按 epoch 的 `history` 等。
