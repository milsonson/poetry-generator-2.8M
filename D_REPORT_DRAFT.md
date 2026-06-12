# D 部分：超参数实验与统一集成报告草稿

## 1. 集成说明

本项目将 A/B/C 三部分整合到 `model.py` 中：

- A：`CausalSelfAttention.__init__` 中定义 `w_q/w_k/w_v/w_o`，并在 `forward` 中把输入 `(B,T,d_model)` 投影为 Q/K/V，再 reshape 为 `(B,n_head,T,d_head)`。
- B：`CausalSelfAttention.forward` 中计算 `Q @ K^T / sqrt(d_head)`，使用上三角 mask 屏蔽未来位置，softmax 后与 V 相乘，最后合并多头并经过 `w_o`。
- C：`FeedForward` 使用 `Linear(d_model,d_ff) -> GELU -> Dropout -> Linear(d_ff,d_model)`，并在 `generate.py` 中通过 `temperature` 控制采样随机性。
- D：补齐 `prepare_data.py / train.py / generate.py / run_experiments.py`，统一跑通数据准备、训练、生成和超参数对比。

## 2. 超参数实验设计

实验采用控制变量法。baseline 固定为：

| 参数 | 取值 |
|---|---:|
| n_layer | 2 |
| n_head | 4 |
| d_model | 96 |
| d_ff | 384 |
| dropout | 0.1 |
| learning_rate | 3e-4 |

在此基础上分别改变 `n_head`、`d_ff`、`dropout`、`learning_rate`：

```bash
python run_experiments.py --max-iters 300 --batch-size 32 --block-size 128
```

脚本会输出：

- `experiments/hyperparam_results.csv`
- `experiments/hyperparam_results.json`
- `experiments/hyperparam_results.md`

报告中推荐直接粘贴 `hyperparam_results.md` 的表格。

## 3. 结果记录位置

正式训练和实验后，把下面文件放进最终 zip：

- `loss_curve.png`：主模型训练 loss 曲线。
- `transformer_poetry.pth`：主模型权重。
- `experiments/hyperparam_results.csv`：D 部分超参对比表。
- `experiments/hyperparam_results.md`：可直接放进报告的结果表和结论。

## 4. 正式超参数实验结果

说明：下表是早期用于控制变量分析的快速超参数实验，使用 10000 首诗、`max_iters=300`、`batch_size=32`、`block_size=128`。最终提交用的主模型已经按老师原始规模重新取 `100000` 条同源数据，清洗后保留 `97559` 首，筛选并标注近体诗 `94737` 首；最终主模型仍使用 `4 layer / d_model=128 / d_ff=512` 的紧凑结构，`best_iter=7600`，`best_val_loss=4.3808`。

| experiment | changed param | params(M) | train loss | val loss | 结论 |
|---|---:|---:|---:|---:|---|
| baseline | baseline | 1.353 | 6.3676 | 6.3731 | 作为基准 |
| fewer_heads | n_head=2 | 1.353 | 6.3735 | 6.3784 | 与 baseline 接近，减少头数没有明显收益 |
| wider_ffn | d_ff=768 | 1.501 | 6.3587 | 6.3744 | 参数更多，训练 loss 略低，但验证 loss 未优于 baseline |
| higher_dropout | dropout=0.2 | 1.353 | 6.4063 | 6.4069 | dropout 偏大后拟合变慢 |
| higher_lr | lr=1e-3 | 1.353 | 5.7998 | 5.9063 | 当前验证 loss 最低 |

本轮控制变量实验中，`lr=1e-3` 的验证集 loss 最低，说明在 300 step 的训练预算下，更大的学习率能更快降低 loss。`n_head=2` 与 baseline 差异很小，说明在当前小模型设置下，减少头数不会造成明显退化，但也没有带来收益；`d_ff=768` 增加了参数量，训练 loss 略低，但验证 loss 没有低于 baseline，说明单纯增大 FFN 宽度在短训练内不一定提升泛化效果；`dropout=0.2` 的训练和验证 loss 都更高，说明正则化偏强会拖慢字符级小模型的拟合。

## 5. 生成样例与 temperature 观察

生成样例保存在 `generation_samples.txt`，当前文件已更新为 100000 来源数据重训后的 checkpoint 样例。temperature 的一般观察仍为：

- `temperature=0.7, start=春`：文本较保守，出现较多常见诗歌意象，如寒烟、松竹、夕阳，整体更像训练语料中的常见表达。
- `temperature=1.0, start=月`：多样性提高，句式和用字更丰富，但局部语义开始不稳定。
- `temperature=1.3, start=山`：随机性明显增强，出现更多少见搭配和不连贯表达，创造性更高但质量下降。

因此，temperature 越低，模型越倾向选择高概率字符，生成结果更保守、更稳；temperature 越高，采样分布更平坦，生成更有变化，但也更容易出现语义跳跃和不通顺。

## 6. 结论写法模板

超参数实验显示，在相同数据划分、batch size、block size、随机种子和训练步数下，不同超参数会影响收敛速度与验证集 loss。若增大 `d_ff` 后验证集 loss 下降，说明更宽的 FFN 提升了模型表达能力；若 dropout 过高导致训练和验证 loss 都偏高，说明正则化过强会降低小模型拟合能力；若 learning rate 过高使 loss 波动明显，说明训练不够稳定。最终选择验证集 loss 较低且生成文本更通顺的一组参数作为主模型配置。

## 7. 给队友的交付说明

我的 D 部分交付：

- 完整集成后的项目目录 `pj9_transformer_poetry/`。
- 超参数实验脚本 `run_experiments.py`。
- 超参数实验结果 `experiments/hyperparam_results.csv` 和 `experiments/hyperparam_results.md`。
- 正式训练产物 `transformer_poetry.pth` 和 `loss_curve.png`。
- 温度生成样例 `generation_samples.txt`。
- 报告可用文字草稿 `D_REPORT_DRAFT.md`。

队友需要把 A/B/C 的说明、loss 曲线截图、temperature 生成结果和我的超参表格合并进最终 PDF。
