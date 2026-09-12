# 03 字符 trigram 运行报告

## 1. 报告定位

本报告只记录真实语料 trigram 结果。原始 bigram 的指标和 sample 保存在项目 02 的 [运行报告](../02_char_bigram/run_report.md)，没有被覆盖。

## 2. 运行方式

训练并保存升级模型：

```bash
PYTHONPATH=projects/03_char_trigram python3 projects/03_char_trigram/train.py
```

加载模型并使用新采样策略：

```bash
PYTHONPATH=projects/03_char_trigram python3 projects/03_char_trigram/generate.py
```

模型产物：

```text
projects/03_char_trigram/artifacts/char_trigram.npz
```

## 3. 数据与配置

| 项目 | 数值 |
|---|---:|
| 语料 | `REAL_CORPUS` |
| 总字符数 | 652 |
| 训练集字符数 | 521 |
| 验证集字符数 | 65 |
| 测试集字符数 | 66 |
| 训练 trigram 数 | 519 |
| 词表大小 | 40 |
| smoothing | 0.1 |
| temperature | 0.8 |
| top-k | 5 |

## 4. 平滑选择

只根据验证集 perplexity 选择平滑值：

| smoothing | 验证 perplexity |
|---:|---:|
| 0.001 | 65.378980 |
| 0.003 | 44.064979 |
| 0.01 | 30.058172 |
| 0.03 | 23.511058 |
| 0.1 | **21.679572** |
| 0.3 | 24.027651 |
| 1.0 | 29.471893 |

因此固定 `smoothing=0.1`，然后才查看测试集结果。

## 5. 最终指标

| 指标 | 数值 |
|---|---:|
| 训练集 loss | 1.489565 |
| 验证集 perplexity | 21.679572 |
| 测试集 perplexity | 17.423710 |

同一真实语料上的 bigram 对照为：训练 loss `1.918266`、验证 perplexity `36.390926`、测试 perplexity `36.970515`。调整平滑后的 trigram 在这次切分上的验证和测试 perplexity 均更低。

## 6. Sample

使用起始字符 `We`、长度 100、seed 7、temperature 0.8、top-k 5：

```text
We theseat alish it areate Rights,dowers al, thent, Goverights thatividess. That inew GovertzzThater
```

与真实语料 bigram 的新采样结果相比，trigram 出现了更多接近原语料的局部片段，例如 `Rights`、`Gover` 和 `That`。输出仍然不是完整自然语言，这是短语料和两字符上下文的能力边界。

## 7. 结论与限制

- 这次升级同时改变了上下文长度和经过验证集选择的平滑值，因此提升不能只归因于一个因素。
- backoff 避免未见上下文直接落入均匀分布，但生成仍可能进入低频字符路径。
- 单个公共领域节选不是大规模真实语言语料，结果只作为教学升级证据。
- 项目 02 的 bigram 报告和模型产物继续保留，便于复现前一阶段结果。
