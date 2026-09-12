# 03 字符 trigram 设计报告

## 1. 升级目标

原始 bigram 使用一个字符预测下一个字符：

```text
P(c_t | c_{t-1})
```

升级后的 trigram 使用前两个字符：

```text
P(c_t | c_{t-2}, c_{t-1})
```

项目 03 使用 `REAL_CORPUS`，并直接采用 `temperature + top-k` 的 `sample_new()` 采样策略。项目 02 的 bigram 代码、模型文件和报告独立保留，用作前一阶段基线。

## 2. 数据与训练样本

真实语料是美国《独立宣言》的公共领域英文节选，共 652 个字符。长度为 `n` 的文本产生 `n-2` 个 trigram 样本，例如：

```text
文本：We hold
上下文：We  e   h  ho  ol
目标：   空格 h  o   l   d
```

数据仍按顺序切分为 `80%/10%/10%`，只使用训练集统计计数。

## 3. 三维概率矩阵

词表大小为 `V` 时：

```text
counts.shape = (V, V, V)
probs.shape  = (V, V, V)
```

`probs[first, second, following]` 表示：

```text
P(following | first, second)
```

沿最后一个维度归一化：

```python
probs = counts / counts.sum(axis=2, keepdims=True)
```

## 4. Loss 与 perplexity

每个位置的损失为：

\[
L_t=-\log P(c_t|c_{t-2},c_{t-1})
\]

整段文本对所有 `n-2` 个预测取平均，perplexity 仍为 `exp(loss)`。

## 5. 平滑参数选择

trigram 的上下文数量是 `V²`，真实短语料中的许多上下文只出现一次或没有出现，因此比 bigram 稀疏。升级版仅根据验证集比较候选平滑值，并选择 `0.1`；测试集不参与选择。

## 6. Bigram backoff

如果生成过程中遇到训练集没有出现过的两字符上下文，trigram 行只有均匀平滑概率。直接 top-k 会在并列字符中任意选择，容易产生噪声。

升级版同时统计 bigram 后备分布：

```text
见过上下文   → P(next | first, second)
未见过上下文 → P(next | second)
```

模型文件保存 `probs`、`backoff_probs` 和 `context_seen`，生成时不需要训练计数。

## 7. 新采样策略

`sample_new()` 默认使用：

```text
start="We"
temperature=0.8
top_k=5
```

每一步先选择 trigram 或 backoff 概率，再通过 temperature 调整分布，只保留 top-k 候选并重新归一化。

## 8. 文件边界

- `model.py`：trigram、backoff、loss、采样和持久化
- `train.py`：使用真实语料训练、评估和保存
- `generate.py`：加载模型并生成
- `tests/test_char_trigram.py`：升级版独立测试
- `run_report.md`：实际运行结果
- `test_report.md`：测试说明

## 9. 边界

- 两字符上下文仍然无法表达句法和长距离语义。
- 当前真实语料只有 652 个字符，trigram 稀疏问题明显。
- backoff 能处理未见上下文，但不等同于神经网络的泛化能力。
- 生成连贯度只能人工观察；loss 和 perplexity 衡量的是真实下一字符概率。
