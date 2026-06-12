# Hyperparameter Experiment Results

| experiment | changed param | params(M) | train loss | val loss | conclusion |
|---|---:|---:|---:|---:|---|
| baseline | baseline | 1.353 | 6.3676 | 6.3731 |  |
| fewer_heads | n_head=2 | 1.353 | 6.3735 | 6.3784 |  |
| wider_ffn | d_ff=768 | 1.501 | 6.3587 | 6.3744 |  |
| higher_dropout | dropout=0.2 | 1.353 | 6.4063 | 6.4069 |  |
| higher_lr | lr=1e-3 | 1.353 | 5.7998 | 5.9063 | best validation loss |

## Brief Analysis

In this run, `higher_lr` reached the lowest validation loss (5.9063). Because all experiments keep the same data split, batch size, block size, seed, and training iterations, the comparison isolates the listed hyperparameter change.

Report wording in Chinese:

在本轮控制变量实验中，验证集 loss 最低的是 `higher_lr` (5.9063)。实验保持数据划分、batch size、block size、随机种子和训练迭代数一致，只改变表中的单个超参数，因此可以比较该超参数对收敛速度和验证集表现的影响。
