---
title: Poetry Generator 2.8M
sdk: docker
app_port: 7860
pinned: false
---

# PJ9 中文诗歌生成项目

## 在线体验

- 真实生成器：https://milsonson-poetry-generator-2-8m.hf.space
- Hugging Face Space：https://huggingface.co/spaces/milsonson/poetry-generator-2-8m
- GitHub 仓库：https://github.com/milsonson/poetry-generator-2.8M

线上版本运行的是 `web_app.py`，会加载仓库中的 `transformer_poetry.pth` 和 `vocab.json` 实时生成诗句；不是 GitHub Pages 静态样例页。

本项目实现了一个面向中文古诗生成的字符级 GPT 风格 Causal Transformer。它不是 BERT，也不是 encoder-decoder 翻译模型，而是一个自回归语言模型：给定前面的字符，预测下一个字符，并在生成阶段按诗体、重复惩罚、温度、自适应采样和候选重排来控制输出。

当前项目重点不是追求大模型规模，而是在一个较小模型上把训练流程、数据清洗、体裁控制、生成策略和弱排序评估做完整，方便课程展示、实验报告和本地交互。

## 当前状态

当前主模型文件为 `transformer_poetry.pth`，前端默认加载该 checkpoint。

当前工作目录：

```text
/home/milsonson/pj9_transformer_poetry_D_ready/pj9_transformer_poetry
```

| 项目 | 当前值 |
|---|---:|
| 架构 | decoder-only Causal Transformer |
| token 粒度 | 字符级 |
| vocab size | `7758` |
| block size | `128` |
| Transformer 层数 | `4` |
| attention heads | `4` |
| hidden size | `128` |
| FFN hidden size | `512` |
| dropout | `0.1` |
| 参数量 | `2,795,776`，约 `2.80M` |
| 当前训练格式 | `poem_sequences` |
| 最佳迭代 | `7600` |
| best val loss | `4.3808` |
| 训练设备 | CUDA |

说明：当前主模型仍保持 `4 layer / d_model 128 / d_ff 512` 的紧凑 Transformer 结构。参数量从旧版约 `2.25M` 增到约 `2.80M`，主要原因是训练数据扩大后词表从 `5642` 增到 `7758`，不是因为加深或加宽了 Transformer。旧版主数据和 checkpoint 已备份到 `backups/before_100k_retrain_20260612_192908/`。

### 当前文件一致性核对

下面这些数字来自当前项目根目录里的实际文件，而不是手写估计：

| 文件 | 关键内容 | 当前值 |
|---|---|---:|
| `poetry_cleaned_stats.json` | 原始抽样条数 | `100000` |
| `poetry_cleaned_stats.json` | 清洗后保留 | `97559` |
| `form_stats.json` | 体裁标注后样本 | `94737` |
| `form_stats.json` | 七言绝句 | `37674` |
| `form_stats.json` | 七言律诗 | `24859` |
| `form_stats.json` | 五言律诗 | `24931` |
| `form_stats.json` | 五言绝句 | `7273` |
| `vocab.json` | `sample_size` | `94737` |
| `vocab.json` | `vocab_size` | `7758` |
| `vocab.json` | `data_format` | `poem_sequences` |
| `transformer_poetry.pth` | `best_iter` | `7600` |
| `transformer_poetry.pth` | `best_val_loss` | `4.3808` |
| `transformer_poetry.pth` | 参数量 | `2795776` |

如果后续重新训练，只要这些文件被替换，就应该同步更新本节和“当前状态”表。

## Web GUI 与公网部署

仓库包含两个网页入口：

- `static/index.html`：本地完整 GUI，配合 `web_app.py` 调用 PyTorch checkpoint 实时生成。
- `docs/index.html`：GitHub Pages 静态展示页，只能展示界面和样例，不能运行 PyTorch 模型。

公网真实生成版本已部署到 Hugging Face Spaces：

```text
https://milsonson-poetry-generator-2-8m.hf.space
```

如果要重新部署或迁移到其他平台，仓库已经包含 Docker 入口，适合部署到 Hugging Face Spaces、Render、Railway、Fly.io 或自己的服务器。详见 `DEPLOY.md`。

本地运行完整生成器：

```bash
python web_app.py --host 127.0.0.1 --port 7860
```

然后打开：

```text
http://127.0.0.1:7860
```

Docker 运行完整生成器：

```bash
docker build -t poetry-generator-2-8m .
docker run --rm -p 7860:7860 poetry-generator-2-8m
```

## 主要优化

### 1. 从原始随机滑窗改为按诗训练

早期训练方式是把所有诗拼成一个很长的字符序列，然后随机切 `block_size=128` 的窗口：

```text
整段文本 -> 随机位置 i -> x = data[i:i+128], y = data[i+1:i+129]
```

这种方式适合通用语言模型，但对诗歌生成不理想，因为训练样本经常从诗中间开始，模型不稳定看到：

```text
体裁 token -> 第一联 -> 第二联 -> 韵脚 -> 结尾
```

现在改成了诗级样本：

```text
prepare_data.py:
每首诗 -> 一个 tensor
末尾追加 <EOS>
batch 内用 <PAD> 补齐

train.py:
x = 诗的前 n-1 个 token
y = 诗的后 n-1 个 token
padding 的 y 位置写成 -100
loss 使用 ignore_index=-100 忽略 padding
```

这意味着每条训练样本都从诗的开头开始，包括体裁 token，例如：

```text
㊁春风...
```

训练脚本仍保留旧的 sliding token 兼容路径，但只要存在 `train_sequences.pt` 和 `val_sequences.pt`，就优先使用诗级训练。

### 2. 数据清洗

新增 `clean_poetry_data.py`，用于清洗原始诗歌文本。

清洗内容包括：

- 统一中文标点；
- 去掉明显非诗内容、注释、网页残留、异常符号；
- 过滤过短或异常过长文本；
- 去重；
- 保留诗内逗号、句号等结构信号；
- 输出清洗统计。

当前清洗结果：

```text
输入诗数: 100000
保留诗数: 97559
移除诗数: 2441
```

相关文件：

- `poetry_cleaned.txt`
- `poetry_cleaned_stats.json`

### 3. 体裁条件训练

当前数据可以稳定支持四种近体诗格式：

| 体裁 | 句长模式 | 样本数 |
|---|---|---:|
| 五言绝句 | `(5,5,5,5)` | `7273` |
| 七言绝句 | `(7,7,7,7)` | `37674` |
| 五言律诗 | `(5 x 8)` | `24931` |
| 七言律诗 | `(7 x 8)` | `24859` |

`prepare_form_data.py` 会自动识别句长结构，把可识别诗歌标注为体裁条件样本：

```text
五言绝句 -> ㊀
七言绝句 -> ㊁
五言律诗 -> ㊂
七言律诗 -> ㊃
```

训练文本示意：

```text
㊁方外主人名道林，怕将水月净身心。居然对我说无我，寂历山深将夜深。
```

这样做的原因是当前 tokenizer 是字符级的。使用单字符体裁 token 可以避免引入复杂 tokenizer，也能让模型稳定看到体裁条件。

当前体裁标注统计：

```text
清洗诗数: 97559
成功标注: 94737
丢弃其他格式: 2822
```

相关文件：

- `generation_forms.py`
- `prepare_form_data.py`
- `poetry_form_labeled.txt`
- `form_stats.json`

### 4. 结构化生成

仅靠体裁 token 不能保证模型每次都严格生成五言/七言，所以生成阶段又加了一层结构约束。

结构化生成做了这些事：

- 支持 `五言绝句`、`七言绝句`、`五言律诗`、`七言律诗`；
- 根据体裁模板控制句数字数；
- 句内只允许采样中文内容字符；
- 到句末自动插入 `，` 或 `。`；
- 插入的标点会喂回模型上下文，而不是只在最后格式化显示。

最后一点很重要。早期结构化生成只是强行连续吐汉字，模型在第 5 或第 7 个字后本来想预测标点，却被迫继续预测汉字，容易偏离训练分布。现在会把句尾标点放回上下文，让模型看到更接近训练时的输入。

### 5. 解码阶段重复惩罚

项目没有直接修改训练 loss 去惩罚重复，因为古诗中本来允许合理叠字，例如“悠悠”“萧萧”。直接在训练 loss 中硬惩罚重复会误伤正常表达。

当前做法是在生成阶段进行 repetition penalty：

- 只作用于最近 `repetition_window` 个 token；
- 对已经出现过的 token 降低 logits；
- 默认值从 `1.1` 提高到 `1.5`，用于压住当前模型常见的重复字和重复短片段；
- Web 前端和命令行都可以调整重复惩罚强度和窗口。

相关参数：

```bash
--repetition-penalty 1.5
--repetition-window 64
```

### 6. 自适应温度

生成时加入了 adaptive temperature。它根据当前 logits 分布的不确定性动态调整温度：

- 如果模型分布过于尖锐，适当升温，增加变化；
- 如果模型分布过于发散，适当降温，减少乱字；
- 最终温度限制在设定上下界内。

相关参数：

```bash
--adaptive-temperature
--target-entropy 0.55
--temperature-strength 0.65
--min-temperature 0.55
--max-temperature 1.35
```

当前默认生成设置更偏稳：

```text
form = 七言绝句
temperature = 0.6
top_k = 40
repetition_penalty = 1.5
adaptive_temperature = true
```

### 7. 50 选 1 候选重排

单次采样不稳定，尤其是小模型容易偶然生成怪句。因此 Web API 和 `generate_candidates.py` 支持一次生成多个候选，再用弱 scorer 选一个相对更好的。

当前默认：

```text
candidates = 50
```

注意这里的 scorer 不是“诗歌质量裁判”，只是弱排序器。它用于把明显更差的候选往后排，例如：

- 格式错误；
- 韵脚明显分散；
- 句尾平仄收束明显不顺；
- 重复字或重复片段过多；
- 极低频怪字；
- 虚字过密；
- 和主题完全脱节。

具体流程是：

1. `generate_candidates.py` 或 Web API 先按同一组生成参数采样多首候选诗。
2. 每首候选交给 `poem_scorer.score_poem()`，得到一个 `PoemScore`。
3. `PoemScore.parts` 保存各分项，例如 `rhyme/tone/theme/style/repetition/rare_chars/weak_syntax`。
4. `PoemScore.reasons` 保存加分理由，例如“格式通过”“近似押韵”“主题字出现”。
5. `PoemScore.warnings` 保存风险提示，例如“韵脚分散”“重复2字片段”“极低频字”。
6. `select_best_candidate()` 按 `(是否被拒绝, -rank_score)` 排序，选择未被格式拒绝且总分最高的候选。

这个 scorer 不参与训练，也不会反向传播梯度。它只在生成后做 rerank，因此不会改变模型参数。

CLI 示例：

```bash
python generate_candidates.py --form 七言绝句 --start 月 --theme 月 --candidates 50
```

JSON 输出：

```bash
python generate_candidates.py --form 七言绝句 --start 月 --candidates 50 --json
```

### 8. 押韵识别和“同声”处理

押韵 scorer 经过多轮调整。现在使用 `pypinyin` 获取普通话拼音韵母，并做保守归并。

例子：

```text
山 / 间 / 关 -> an
间 / 还 -> an / ai，不算押韵
新 / 心 -> 同韵但同声，只给弱分并提示
```

对应逻辑在 `poem_scorer.py`：

- 绝句检查第 2、4 句韵脚；
- 律诗检查第 2、4、6、8 句韵脚；
- 同韵不同声给较高押韵分；
- 同声韵脚降权；
- 重复韵脚字降权；
- 韵脚分散给负分。

这仍然不是严格平水韵。它只是一个可解释、可测试、不会太死板的近似规则。

### 9. 平仄和句尾声调弱排序

生成结果之前经常出现“最后一句或中间某一句最后一个字读起来别扭”的问题。这里的别扭主要不是语义问题，而是音韵问题：某一句的句尾声调和平仄位置不协调，尤其是韵脚句用仄声收尾时，会显得收束不稳。

当前处理方式是给 scorer 增加 `tone` 分项，只在 50 个候选之间做 rerank，不在采样时硬 mask token。这样可以利用“50 选 1”显著改善输出，同时避免把模型逼得过死。

核心原则：

- 不构建大词表，不维护“哪些字好/哪些字坏”的名单；
- 不按字面语义判断句尾是否好，只看声调和平仄搭配；
- 声调来自 `pypinyin` 对生成结果中实际汉字的动态读取；
- 如果当前环境没有 `pypinyin`，声调读不到就跳过 `tone` 评分，不手工猜；
- 平仄评分只影响排序，不会导致候选被 `rejected`；
- 分数限制在 `-8` 到 `+4`，避免平仄项压倒格式、押韵、重复和怪字等更硬的信号。

代码入口：

```text
pinyin_tone(char)
tone_class(char)
expected_line_final_tone(form, line_index)
tail_tone_score(line, form, line_index)
tone_cadence_bonus(text, form)
```

其中 `tone_class()` 使用宽松的普通话近似：

```text
1 声、2 声 -> 平
3 声、4 声 -> 仄
轻声或无法识别 -> 不参与评分
```

句位规则是宽松近体诗习惯，而不是严格格律模板：

```text
绝句: 第 2、4 句是韵脚句，句尾偏好平声
律诗: 第 2、4、6、8 句是韵脚句，句尾偏好平声
其他非韵脚句: 句尾轻微偏好仄声
```

每一句还会看末尾几个字，而不是只看最后一个字：

- 最后一个字：根据该句是否为韵脚句，判断平/仄是否合适；
- 倒数第二个字：轻微鼓励和最后一个字形成起伏，避免末二字平仄黏住；
- 末三字：如果三个已知声调全是平或全是仄，会轻微扣分；
- 每一联的两句句尾：轻微偏好“前句仄收、后句平收”的收束关系。

例子：

```text
春风吹柳色，明月照江山。
白鸟归云外，清霜入故关。
```

如果第 2、4 句韵脚 `山/关` 能读成平声，`tone` 分项会加分；如果换成第 4 声韵脚，即使押韵分项认为韵母接近，`tone` 也会扣分并给出类似 `韵脚偏仄` 的 warning。

这项设计故意保持宽松，因为当前模型不是严格格律模型，训练数据也不是按平水韵、对仗和平仄模板精标的。更实用的策略是：结构化生成保证句长和标点，押韵 scorer 保证偶句韵脚大致接近，`tone` scorer 再把声调收束明显差的候选排后。

### 10. 弱 scorer 详细机制

`poem_scorer.py` 的核心入口是：

```text
score_poem(text, form, theme="", poet="")
select_best_candidate(candidates, form, theme="", poet="")
```

`score_poem()` 会先做格式检查。如果用户选择的是五绝、七绝、五律或七律，格式必须完全匹配 `generation_forms.FORM_TEMPLATES`：

```text
五言绝句: (5,5,5,5)
七言绝句: (7,7,7,7)
五言律诗: (5,5,5,5,5,5,5,5)
七言律诗: (7,7,7,7,7,7,7,7)
```

如果句数字数不符合，直接返回：

```text
rejected = true
rank_score = -999
parts = {"format": -999}
```

这样格式错误的候选不会因为押韵或主题词碰巧命中而排到前面。

格式通过后，scorer 会计算这些分项：

| 分项 | 函数 | 分数范围/规则 | 作用 |
|---|---|---|---|
| 格式 | `format_check()` | 不合格直接 `-999` | 保证选择的候选符合目标诗体句长 |
| 押韵 | `rhyme_bonus()` | `+6 / +2 / +1 / -1 / -2` | 鼓励偶句韵脚接近，惩罚韵脚分散 |
| 平仄 | `tone_cadence_bonus()` | `-8` 到 `+4` | 按句位、尾字、末二字、末三字和联句收束做宽松声调排序 |
| 重复 | `repetition_penalty()` | 最多扣 `18` | 惩罚连续重复字、重复二字/三字片段和坏模式 |
| 低频字 | `rare_char_penalty()` | 最多扣 `18` | 使用当前训练语料字频，惩罚极低频怪字 |
| 弱句法 | `weak_syntax_penalty()` | 最多扣 `12` | 惩罚虚字过密、数字过密、缺少景物/动作支点的句子 |
| 主题 | `theme_bonus()` | 最多加 `4` | 起始主题字或主题意象分布越明显越靠前 |
| 诗人风格 | `style_bonus()` | 最多加 `3` | 对李白、杜甫、李贺的人工词场做轻量命中加分 |

押韵分项的细节：

- 绝句只看第 2、4 句句尾；
- 律诗看第 2、4、6、8 句句尾；
- 所有韵脚都能归入同一近似韵母组，给 `+6`；
- 多数韵脚同组但不完全一致，给 `+2`；
- 同韵但韵脚字重复，只给 `+1`；
- 韵脚无法可靠识别，给 `-1`；
- 韵脚明显分散，给 `-2`。

为了避免普通话近似押韵过度乐观，scorer 还会处理两个特殊情况：

- `新/心` 这类同声韵脚虽然同韵，但只给较低押韵分，并在 warnings 中提示“同声韵脚”；
- `还` 是常见多音字，当前用 `RHYME_GROUP_OVERRIDES` 保守处理为 `ai`，避免 `间/还` 被误判成干净押韵。

平仄分项的细节：

- `pinyin_tone()` 只动态读取单字声调，不维护手工声调大表；
- `expected_line_final_tone()` 根据诗体和第几句给出句尾期望；
- `tail_tone_score()` 评价单句尾部的最后一个字、末二字、末三字；
- `tone_cadence_bonus()` 汇总全诗，并额外检查每一联的两个句尾是否有起伏；
- 如果某个字读不到声调，该字不会参与平仄扣分；
- 该分项不会设置 `rejected=True`，只给 `parts["tone"]`、`reasons` 和 `warnings`。

重复分项的细节：

- 连续相同字会扣分，例如非安全叠字的 `情情`；
- 重复二字片段每多一次扣 `2`；
- 重复三字片段每多一次扣 `5`；
- `BAD_PATTERNS` 中的弱句模式额外扣分，例如“诗家诗句”“何言是说”“即是贫”；
- 扣分上限是 `18`，避免某一项完全压倒其他信号。

低频字分项的细节：

- `load_char_counts()` 会从 `poetry_form_labeled.txt`、`poetry_cleaned.txt` 或 `poetry.txt` 读取当前训练语料字频；
- 训练语料中出现 `0-1` 次的字按“极低频字”处理，每个扣 `6`；
- 出现 `2-3` 次的字按“低频字”处理，每个扣 `2`；
- 低频字扣分上限是 `18`。

弱句法分项不是语法模型，只是少量启发式规则：

- 一句里虚字过密，例如 `不/无/有/何/谁/莫/更/未/自/相/为/与/是/如/可/应/将/但/却/又` 过多，会扣分；
- 数字字过密，例如一二三四五六七八九十百千万过多，会扣分；
- 一句里既没有常见景物字，也没有常见动作字，会被认为“句法支点弱”。

主题分项目前支持这些主题词场：

```text
月、春、山、水、酒、风
```

如果主题字本身出现在诗中，加 `2`；如果至少两句命中主题词场，再加 `2`。Web API 默认把 `theme` 设成用户输入的起始字，所以输入“月”时，含有月、夜、霜、露、云、天、清、影、桂、寒、秋等意象的候选更容易靠前。

诗人风格分项目前是很轻量的人工词场，不等于严格风格判别：

```text
李白: 月、酒、天、云、长安、青、君、仙、剑、山、水、舟
杜甫: 江、客、故园、白发、兵乱、老、村、国、风尘
李贺: 金、玉、冷、鬼、血、宫、秋、露、马、龙、寒
```

命中这些词场会有小幅加分，最多 `3`。它的作用是弱排序，不是保证生成“真正像某位诗人”。

最终总分是各分项相加：

```text
rank_score = rhyme + tone + theme + style + repetition + rare_chars + weak_syntax
```

因为这个分数只是排序用，所以报告里不要把它写成“诗歌质量分”。更准确的说法是：它是一个可解释的弱规则 reranker，用来从多个候选中筛掉格式错、韵脚散、句尾声调别扭、重复重、怪字多、主题弱的样本。

相关测试覆盖：

- `山/间/关` 应归为同韵；
- `间/还` 不应归为同韵；
- `新/心` 这种同声韵脚不能拿到完整押韵分；
- `间/还` 不应压过干净押韵候选。
- 缺少 `pypinyin` 时，`pinyin_tone()` 不回退到手工声调表；
- `tone` 分项会根据第几句、最后一个字、末二字和末三字做宽松排序；
- 仄声韵脚候选不会被拒绝，但在 50 选 1 中会排到更顺的平声韵脚候选后面。

### 11. 诗人风格微调

项目还尝试了不同诗人的风格微调。当前保留三个版本：

- 李白
- 杜甫
- 李贺

对应 checkpoint：

```text
poet_style_models/李白/transformer_poetry.pth
poet_style_models/杜甫/transformer_poetry.pth
poet_style_models/李贺/transformer_poetry.pth
```

风格数据来自本地 `chinese-poetry` JSON 数据，使用 `build_poet_style_data.py` 按作者抽取、清洗、转简体，并尽量保持和主模型词表兼容。

当前微调摘要：

| 诗人 | 支持样本 | train | val | best val loss |
|---|---:|---:|---:|---:|
| 李白 | `372` | `327` | `45` | `4.7419` |
| 杜甫 | `709` | `624` | `85` | `4.8037` |
| 李贺 | `92` | `81` | `11` | `5.1340` |

注意：这些风格模型数据量很小，适合展示“倾向”，不能保证稳定复现真实诗人水平。

### 12. 本地 Web 前端

新增了本地 Web UI：

- `web_app.py`
- `static/index.html`

页面支持：

- 起始文字；
- 体裁选择；
- 温度；
- 续写字数；
- 生成结果展示。

Web 前端保留常用生成参数，方便调试重复和采样稳定性；已经移除了保存样例和训练曲线展示，避免页面被演示材料占据。默认参数是：

```text
top_k = 40
repetition_penalty = 1.5
repetition_window = 64
adaptive_temperature = true
candidates = 50
```

也就是说，网页上点一次“生成”，后端默认会采样 50 个候选，然后用 `poem_scorer.py` 里的格式、押韵、平仄、重复、低频字、弱句法和主题分项做弱排序，返回当前分数最高的一首。自适应温度默认开启。

默认端口是 `7860`：

```bash
python web_app.py --host 127.0.0.1 --port 7860
```

如果 `7860` 被占用，可以换端口。本次调试时 `7860` 和 `7861` 已被其他 Python 服务占用，所以使用过 `7862`：

```bash
python web_app.py --host 127.0.0.1 --port 7862
```

## 项目结构

```text
.
├── model.py                       # Transformer 模型、采样辅助函数
├── train.py                       # 训练脚本，优先使用诗级序列
├── prepare_data.py                # 构造 vocab、train/val 数据、诗级 tensor
├── clean_poetry_data.py           # 数据清洗
├── prepare_form_data.py           # 自动识别五绝/七绝/五律/七律并加体裁 token
├── generation_forms.py            # 体裁 token、句长模板、结构化格式化
├── generate.py                    # CLI 生成，支持结构化生成
├── generate_candidates.py         # 多候选生成和弱排序
├── poem_scorer.py                 # 弱 scorer：格式、押韵、平仄、重复、低频字等
├── web_app.py                     # 本地 Web API
├── static/index.html              # 前端页面
├── build_poet_style_data.py       # 从作者数据构建风格语料
├── transformer_poetry.pth         # 当前主模型
├── vocab.json                     # 当前词表
├── train_sequences.pt             # 诗级训练序列
├── val_sequences.pt               # 诗级验证序列
├── loss_curve.png                 # 当前训练曲线
├── generation_samples.txt         # 当前展示样例
├── data_100k_raw/                 # 从 Hugging Face 抽取的 100000 条原始同源数据
├── data_100k_form/                # 100000 来源数据清洗、体裁标注和训练 tensor
├── model_100k_form/               # 100000 来源数据训练得到的 checkpoint 和 loss 曲线
├── poet_style_models/             # 李白、杜甫、李贺风格 checkpoint
├── backups/                       # 历次模型备份
└── tests/                         # 回归测试
```

## 安装环境

推荐使用项目虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前本机调试时使用的 Python 环境是：

```text
/home/milsonson/pj9_transformer_poetry_D_ready/pj9_transformer_poetry/.venv/bin/python
```

如果使用这个环境，可以把下面命令里的 `python` 替换成该绝对路径。

## 从零复现主流程

### 1. 按老师原始规模抽取数据

```bash
mkdir -p data_100k_raw data_100k_form

python prepare_data.py \
  --sample-size 100000 \
  --output-dir data_100k_raw
```

这一步从 `Lifan-Z/Chinese-poetries-txt` 抽取最多 `100000` 条原始诗词文本。`data_100k_raw/poetry.txt` 是后续清洗的输入。

### 2. 清洗原始数据

```bash
python clean_poetry_data.py \
  --input data_100k_raw/poetry.txt \
  --output data_100k_form/poetry_cleaned.txt \
  --stats data_100k_form/poetry_cleaned_stats.json
```

### 3. 自动标注体裁

```bash
python prepare_form_data.py \
  --input data_100k_form/poetry_cleaned.txt \
  --output data_100k_form/poetry_form_labeled.txt \
  --stats data_100k_form/form_stats.json
```

### 4. 构建训练数据

```bash
python prepare_data.py \
  --local-txt data_100k_form/poetry_form_labeled.txt \
  --sample-size 0 \
  --output-dir data_100k_form
```

这一步会生成：

```text
vocab.json
train_data.pt
val_data.pt
train_sequences.pt
val_sequences.pt
poetry.txt
```

`train_data.pt` / `val_data.pt` 是旧滑窗兼容数据；当前训练优先使用 `train_sequences.pt` / `val_sequences.pt`。

### 5. 训练主模型

```bash
python train.py \
  --data-dir data_100k_form \
  --output-dir model_100k_form \
  --device cuda
```

当前默认训练配置：

```text
max_iters = 8000
batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
d_model = 128
d_ff = 512
learning_rate = 6e-4
weight_decay = 0.01
eval_interval = 200
eval_iters = 30
early_stop_patience = 5
grad_clip = 1.0
```

训练脚本会：

- 从随机初始化开始训练；
- 优先使用整首诗序列；
- 定期评估 train/val loss；
- 保存验证集最优 checkpoint，而不是最后一步；
- early stopping 防止过拟合；
- 输出 `transformer_poetry.pth` 和 `loss_curve.png`。

训练完成后，把 `model_100k_form/transformer_poetry.pth`、`model_100k_form/loss_curve.png` 和 `data_100k_form/` 中的词表/数据文件复制到项目根目录，即可作为默认模型使用。当前仓库根目录已经是这个扩大数据版本。

### 6. CLI 生成

```bash
python generate.py --form 七言绝句 --start 月
```

更完整的例子：

```bash
python generate.py \
  --form 七言绝句 \
  --start 月 \
  --temperature 0.6 \
  --top-k 40 \
  --repetition-penalty 1.5 \
  --repetition-window 64 \
  --adaptive-temperature
```

### 7. 50 选 1 生成

```bash
python generate_candidates.py \
  --form 七言绝句 \
  --start 月 \
  --theme 月 \
  --candidates 50
```

### 8. 启动 Web 页面

```bash
python web_app.py --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860/
```

如果端口被占用，可以换成 `7862`：

```bash
python web_app.py --host 127.0.0.1 --port 7862
```

## 训练到底是怎么做的

当前训练不是“每步从一整段大文本中随机切窗口”。现在是：

```text
每个 batch 随机抽若干首完整诗
每首诗都从开头 token 开始
输入 x 是前 n-1 个 token
目标 y 是后 n-1 个 token
短诗 padding 到同一长度
padding 位置不参与 loss
```

这对本项目更合适，因为模型能稳定看到：

```text
体裁 token -> 起句 -> 承句 -> 转句 -> 结句 -> <EOS>
```

重新运行 `train.py` 时仍然是从头训练，不会自动续训旧 checkpoint。如果要续训，需要额外实现加载旧 `model_state_dict` 和 optimizer state 的逻辑。

## 为什么生成质量仍有限

当前优化主要解决了格式、训练样本对齐、重复控制、候选筛选和弱押韵判断，但模型仍然有明显上限。

主要原因：

1. 模型是字符级，从零训练，不理解词和语义结构。
2. 当前虽然已经按老师原始规模从 100000 条同源数据中清洗筛出 94737 首近体诗，但这些诗仍然是混合题材、混合时代、混合作者的数据。
3. 语料本身比较杂，混有题赠、应酬、官场、佛道、题跋等内容，不是严格精选诗集。
4. 当前 best val loss 约为 `4.3808`，比旧版 10000 条数据训练明显降低，但字符级 next-token prediction 仍然有较高不确定性。
5. 结构化生成只保证句长和标点，不保证真正平仄、古韵、对仗和起承转合。
6. 当前押韵 scorer 是生成后弱排序，不是强制格律生成。

因此当前模型可以做到：

- 五言/七言格式稳定；
- 绝句/律诗句数稳定；
- 可以本地快速生成；
- 可以 50 选 1 规避明显坏样本；
- 可以识别一部分押韵问题。

但不能保证：

- 语义总是通顺；
- 每首都押古韵；
- 平仄严格正确；
- 律诗对仗合格；
- 诗人风格稳定可信。

## 关于词牌名

当前主数据集基本是近体诗，不是宋词数据集。我们用几个常见词牌的句长模板检查过，几乎没有精确匹配。

所以当前项目不声称支持真正词牌名训练。若要支持词牌，需要额外引入带词牌 metadata 的宋词数据，例如：

```text
水调歌头
如梦令
浣溪沙
卜算子
念奴娇
```

并且要把词牌名作为条件 token，同时在生成阶段加入对应句长模板。

## 重要实验记录

本节保留的是项目迭代过程中的历史实验记录。前几项不是当前默认模型，只用于说明我们做过哪些改动，以及为什么最后选择当前 100000 来源的 compact 主模型。

### 原始模型

最初 checkpoint 约 `2.30M` 参数：

```text
vocab_size = 5817
n_layer = 4
n_head = 4
d_model = 128
d_ff = 512
final val loss ≈ 5.1396
```

### 清洗后重训

清洗后词表减小，训练分布变化：

```text
清洗后诗数 = 9971
vocab_size ≈ 5688
final val loss ≈ 5.2051
```

### 体裁条件训练

加入体裁 token 后：

```text
成功标注 = 9690
vocab_size ≈ 5641
final val loss ≈ 5.0910
```

### 强模型尝试

曾尝试：

```text
n_layer = 6
n_head = 8
d_model = 256
d_ff = 1024
params ≈ 7.66M
best val loss ≈ 5.0218
```

但后来因为模型规模偏大、用户希望回到原始大小附近，主模型回退到紧凑 Transformer 结构。

### 诗级训练

改成整首诗训练后：

```text
data_format = poem_sequences
train poems = 8721
val poems = 969
```

这一步改善的是训练样本结构和体裁 token 稳定性，不保证单看 loss 一定低于滑窗。

### 当前 100000 来源 compact 主模型

当前最终主模型：

```text
params = 2,795,776
vocab_size = 7758
data_format = poem_sequences
train poems = 85263
val poems = 9474
best_iter = 7600
best_val_loss = 4.3808
```

## 测试和验证

运行全部测试：

```bash
pytest tests
```

如果当前环境没有 `pytest`，可使用虚拟环境：

```bash
.venv/bin/pytest tests
```

编译检查：

```bash
python -m py_compile \
  clean_poetry_data.py \
  generation_forms.py \
  prepare_form_data.py \
  prepare_data.py \
  model.py \
  train.py \
  generate.py \
  generate_candidates.py \
  poem_scorer.py \
  web_app.py
```

Web API 状态检查：

```bash
curl -s http://127.0.0.1:7860/api/status
```

生成接口：

```bash
curl -s -X POST http://127.0.0.1:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"start":"春","form":"七言绝句","candidates":50}'
```

## 后续改进方向

优先级较高：

1. 更精选的数据集，而不是盲目扩大杂数据。
2. 把 scorer 继续作为弱排序器优化，尤其是押韵、重复和坏搭配。
3. 生成多个候选后做更细的 rerank，而不是强行写死规则。
4. 引入更可靠的词级信息，例如分词、常见二字词搭配统计。
5. 如果要强调格律，可做“软平仄约束”，不要一开始硬 mask。

暂不建议：

1. 继续盲目加训练轮数，当前已经容易过拟合。
2. 直接把重复惩罚写进训练 loss，容易误伤正常叠字。
3. 声称支持严格平水韵或词牌名，当前数据和规则还不够。
4. 用过强的硬规则控制韵脚，否则会牺牲语义自然度。

## 结论

这个项目现在已经从一个基础字符级诗歌 LM，扩展成了一个较完整的本地诗歌生成系统：

- 有数据清洗；
- 有体裁条件训练；
- 有按整首诗的训练样本；
- 有结构化生成；
- 有重复惩罚和自适应温度；
- 有 50 选 1 候选重排；
- 有更严格但不死板的押韵弱评分；
- 有李白、杜甫、李贺风格微调 checkpoint；
- 有可交互 Web 页面。

当前最稳的使用方式是：选择 `七言绝句`，使用默认低温和 `50` 候选，让 scorer 做弱排序。这样生成结果格式最稳定，也最能避开明显重复和错韵样本。

## 原始项目要求对照

原始老师给的项目目录在：

```text
/home/milsonson/下载/PJ9_Transformer/PJ9_Transformer
```

当前完成版项目目录在：

```text
/home/milsonson/pj9_transformer_poetry_D_ready/pj9_transformer_poetry
```

原始要求来自 `实验指导.md`，核心任务是补全小型 decoder-only Transformer 的因果自注意力和前馈网络，并完成训练、生成、loss 曲线、temperature 对比和超参数对比。当前项目没有使用 `nn.Transformer` 一键封装，而是在 `model.py` 中手写实现。

| 原始要求 | 当前文件 | 当前实现 |
|---|---|---|
| 补全 `CausalSelfAttention.forward` | `model.py` | `CausalSelfAttention` 中手写 Q/K/V 线性投影、多头 reshape、`Q @ K^T / sqrt(d_head)`、上三角 causal mask、softmax、dropout、`att @ V`、合并多头和 `w_o` 输出。 |
| 补全 `FeedForward.__init__` 和 `FeedForward.forward` | `model.py` | `FeedForward` 使用 `Linear(d_model,d_ff) -> GELU -> Dropout -> Linear(d_ff,d_model)`，输入输出形状保持 `(B,T,d_model)`。 |
| 构建字符级 GPT/decoder-only LM | `model.py` | `PoetryTransformer` 使用 token embedding、position embedding、多层 `TransformerBlock`、最终 `LayerNorm` 和 `lm_head`，任务仍是 next-token prediction。 |
| 训练脚本 | `train.py` | 加载 `vocab.json`、`train_data.pt/val_data.pt` 或优先加载 `train_sequences.pt/val_sequences.pt`，训练模型，评估 train/val loss，保存最佳 checkpoint。 |
| 数据准备 | `prepare_data.py` | 默认数据源仍是 `Lifan-Z/Chinese-poetries-txt`；也支持 `--local-txt` 使用本地清洗和体裁标注后的数据。 |
| 验证集 loss 和 loss 曲线 | `train.py`, `loss_curve.png` | 训练时定期估计 train/val loss，最终保存 `loss_curve.png`；当前主模型记录在 checkpoint 和本文档中。 |
| 生成脚本和 temperature 采样 | `generate.py` | 加载 `transformer_poetry.pth`，支持 `--temperature`、`--top-k`、重复惩罚、自适应温度和体裁结构化生成。 |
| 至少一条生成示例 | `generation_samples.txt` | 保存了不同起始字和诗体的生成样例，可直接放入报告。 |
| temperature 对比实验 | `generation_samples.txt`, `D_REPORT_DRAFT.md` | 报告草稿中写了 temperature 观察；如果严格按“每个 temperature 至少 3 次”验收，建议额外补一份 3 档 temperature、每档 3 条的生成记录。 |
| 至少 3 组超参对比，且涉及至少两类超参 | `run_experiments.py`, `experiments/hyperparam_results.*` | 已比较 baseline、`n_head=2`、`d_ff=768`、`dropout=0.2`、`lr=1e-3`，涉及注意力头数、FFN 宽度、dropout 和学习率。 |
| 报告说明因果掩码、缩放点积、FFN 作用 | `D_REPORT_DRAFT.md` | 已给出 A/B/C/D 集成说明和可放入报告的文字草稿；最终报告还需要合并队友 A/B/C 的说明。 |

当前训练数据和老师原始数据的关系：

- 老师原始脚本直接从 `Lifan-Z/Chinese-poetries-txt` 随机抽样最多 `100000` 条，构造长 token 序列并用滑窗训练。
- 当前项目的数据来源仍是 `Lifan-Z/Chinese-poetries-txt`，但为了让生成结果更符合五绝、七绝、五律、七律，额外做了清洗、近体诗筛选和体裁 token 标注。
- 当前主训练集为 `94737` 首体裁标注诗，`vocab.json` 中记录 `sample_size = 94737`、`data_format = poem_sequences`。
- 当前 `train_data.pt/val_data.pt` 仍保留旧式长序列兼容数据，但 `train.py` 会优先使用 `train_sequences.pt/val_sequences.pt` 按整首诗训练。

当前项目相比原始要求额外增加了：

- `clean_poetry_data.py`：清洗异常文本、统一标点、去重和过滤非诗内容。
- `prepare_form_data.py` / `generation_forms.py`：识别五绝、七绝、五律、七律，并加入单字符体裁 token。
- `generate_candidates.py` / `poem_scorer.py`：一次生成多个候选，再按格式、押韵、平仄、重复、低频字、主题词等弱规则重排。
- `web_app.py` / `static/index.html`：提供本地 Web 交互页面。
- `tests/`：覆盖数据清洗、体裁识别、结构化生成、重复惩罚、自适应温度和训练默认参数等关键逻辑。

需要注意的提交兼容点：

1. 原始资料包的生成脚本叫 `predict.py`，当前项目主生成脚本叫 `generate.py`。人工批改时说明即可；如果老师按文件名自动运行，建议添加一个兼容的 `predict.py` 入口。
2. 当前项目为了诗体控制没有直接使用老师原始的 10 万条长序列数据，而是使用同源数据的清洗筛选版本。报告中应明确写成“数据来源相同，训练样本经过清洗和近体诗筛选”。
