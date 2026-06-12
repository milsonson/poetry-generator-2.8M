# 给队友的 D 部分交接

我负责的是 D：超参数实验、A/B/C 代码统一集成、README/报告材料整理。

## 我已经完成的内容

- 已把 A/B/C 的代码整合进 `model.py`，包括 `CausalSelfAttention`、`FeedForward`、`TransformerBlock` 和完整 `PoetryTransformer`。
- 已补齐完整运行流程：
  - `prepare_data.py`
  - `train.py`
  - `generate.py`
  - `run_experiments.py`
- 已跑通一组正式超参数实验，结果在：
  - `experiments/hyperparam_results.csv`
  - `experiments/hyperparam_results.md`
  - `experiments/hyperparam_results.json`
- 已用 100000 条同源原始数据重建清洗/体裁标注数据，并跑完当前正式主训练：
  - `transformer_poetry.pth`
  - `loss_curve.png`
  - 原始抽样: 100000
  - 清洗后: 97559
  - 体裁标注后: 94737
  - best_iter: 7600
  - best_val_loss: 4.3808
- 已生成当前 checkpoint 样例：
  - `generation_samples.txt`
- 已写好 D 部分报告草稿：
  - `D_REPORT_DRAFT.md`

## 可以直接给报告的人粘贴的结论

本轮控制变量实验中，`lr=1e-3` 的验证集 loss 最低，验证 loss 为 5.9063，说明在 300 step 的训练预算下，更大的学习率能更快降低 loss。`n_head=2` 与 baseline 差异很小，说明在当前小模型设置下，减少头数不会造成明显退化，但也没有带来收益；`d_ff=768` 增加了参数量，训练 loss 略低，但验证 loss 没有低于 baseline；`dropout=0.2` 的训练和验证 loss 都更高，说明正则化偏强会拖慢字符级小模型的拟合。

## 需要队友合并进最终 PDF 的材料

- A/B 的 causal attention 原理说明和代码截图。
- C 的 FeedForward 说明、temperature 生成对比结果。
- 我的 D 部分超参表格：`experiments/hyperparam_results.md`。
- 主训练的 loss 曲线：`loss_curve.png`。
- 2-3 组生成样例：`generation_samples.txt`。

## 推荐正式重跑命令

正式训练已经在服务器上跑完，通常不用重跑。复现实验命令如下：

```bash
python prepare_data.py --sample-size 100000 --output-dir data_100k_raw
python clean_poetry_data.py --input data_100k_raw/poetry.txt --output data_100k_form/poetry_cleaned.txt --stats data_100k_form/poetry_cleaned_stats.json
python prepare_form_data.py --input data_100k_form/poetry_cleaned.txt --output data_100k_form/poetry_form_labeled.txt --stats data_100k_form/form_stats.json
python prepare_data.py --local-txt data_100k_form/poetry_form_labeled.txt --sample-size 0 --output-dir data_100k_form
python train.py --data-dir data_100k_form --output-dir model_100k_form --device cuda
python run_experiments.py --max-iters 300 --batch-size 32 --block-size 128 --device cuda
```

如果只需要生成更多样例：

```bash
python generate.py --start 春 --temperature 0.7 --max-new-tokens 80
python generate.py --start 月 --temperature 1.0 --max-new-tokens 80
python generate.py --start 山 --temperature 1.3 --max-new-tokens 80
```
