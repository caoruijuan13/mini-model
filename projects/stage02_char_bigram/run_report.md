# 02 字符 bigram 运行结果报告

## 1. 报告范围

本报告记录一次训练、评估和模型保存运行，不执行文本生成。设计说明见 [design.md](design.md)，自动化测试见 [test_report.md](test_report.md)。

## 2. 运行方式

```bash
PYTHONPATH=projects/stage02_char_bigram python3 projects/stage02_char_bigram/train.py
```

推理模型产物保存到：

```text
projects/stage02_char_bigram/artifacts/char_bigram.npz
```

## 3. 训练前：语料和切分

本次运行使用固定的 `DEFAULT_CORPUS`。该语料由同一段英文说明文本重复组成，长度为 `1512` 个字符。

| 数据集 | 比例 | 字符数 | 相邻转移数 |
|---|---:|---:|---:|
| 训练集 | 80% | 1209 | 1208 |
| 验证集 | 10% | 151 | 150 |
| 测试集 | 10% | 152 | 151 |

词表大小为 `26`，包含语料中出现的字母、空格和标点。模型平滑系数为 `1e-3`。

## 4. 训练过程

训练阶段只使用训练集的 1208 个相邻字符转移：

1. 将字符转换为 ID；
2. 统计 `current → following` 的计数；
3. 加入平滑计数；
4. 对每一行归一化为条件概率；
5. 在训练、验证和测试文本上计算指标；
6. 保存词表和概率矩阵作为推理模型。

这里没有 epoch、梯度或学习率，因为概率表由计数直接计算。

## 5. 运行结果

| 指标 | 数值 |
|---|---:|
| 训练集 loss | 1.736987 |
| 验证集 perplexity | 5.840164 |
| 测试集 perplexity | 5.714840 |

## 6. Sample 输出

使用训练阶段保存的推理模型，从起始字符 `s` 开始生成 100 个字符：

```text
smon es tre, cimexpry bl aldured. bl. are untin mesel e. thelenged. auctesmary maiod urilaibule. ake
```

采样配置：

| 项目 | 配置 |
|---|---|
| 起始字符 | `s` |
| 生成长度 | `100` |
| 随机种子 | `7` |
| 模型来源 | `artifacts/char_bigram.npz` |

输出中存在部分合理的字符片段，但整体不连贯。这是预期现象：当前模型只根据前一个字符预测下一个字符，不具备更长上下文和句法建模能力。Sample 仅用于确认加载、采样和随机复现流程，不作为语言质量指标。

## 7. 结果解释

- 训练集 loss 是所有训练相邻字符对负对数概率的平均值。
- 验证集和测试集使用训练集得到的同一张概率表，没有参与计数。
- 验证集和测试集困惑度接近，说明在这段重复语料上没有明显的评估崩溃。
- 困惑度仍然较高，符合只使用一个字符上下文的极简模型特征。

## 8. 结果限制

- 语料规模小且高度重复，结果不能代表真实语言建模能力。
- 顺序切分的数据片段来自重复语料，评估难度有限。
- 只报告 loss 和困惑度，没有衡量语义、语法或文本质量。
- 模型没有未知字符处理，也没有更长上下文、神经网络参数或批量训练。

## 9. 原始采样与改进采样对比

原有运行报告中的 sample 来自 `sample()`，该方法保留作为全量概率采样基线。使用同一个已保存模型、相同起始字符和随机种子，调用 `sample_new(temperature=0.8, top_k=5)` 得到：

```text
s bectimodain, e be bls tsery erels anged bl mondalangure aule merere an, me be the tind ectind meri
```

新策略过滤了低概率候选字符，输出中的字符片段更集中；但模型仍只有一个字符的上下文，因此整体连贯性提升有限。这个对比反映的是采样策略差异，不是模型训练效果差异。

## 10. 真实语料对照

真实语料使用美国《独立宣言》公共领域英文节选。该实验独立构建模型，不覆盖前面的 `DEFAULT_CORPUS` 基线结果。

| 项目 | 数值 |
|---|---:|
| 总字符数 | 652 |
| 训练集字符数 | 521 |
| 验证集字符数 | 65 |
| 测试集字符数 | 66 |
| 词表大小 | 40 |
| 训练集 loss | 1.918266 |
| 验证集 perplexity | 36.390926 |
| 测试集 perplexity | 36.970515 |

使用相同的起始字符 `W`、长度 `100` 和随机种子 `7`，原始 `sample()` 输出：

```text
Wess at tome anghtopuigh Gomopond at, Cr athser ioven d Thewenme f e uitese tsthalthathecof theat d 
```

改进后的 `sample_new(temperature=0.8, top_k=5)` 输出：

```text
We toved tharnd t therntir o thesstharsernene ts t atindo athe tonght athe it thto to thengo of-eded
```

真实语料的词表更大、字符组合更多，而且没有重复短句强化固定转移，因此验证和测试 perplexity 明显高于教学基线。`sample_new()` 减少了低概率字符跳转，但 bigram 仍只依赖前一个字符，所以无法形成稳定的长距离语义。两组结果只用于同一模型结构下的语料和采样对照。
