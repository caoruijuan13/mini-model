# 05 Token Embedding 与 MLP 设计报告

## 1. 目标

阶段 05 使用阶段 04 的固定 Tokenizer，以两个 token ID 为上下文，通过可训练 embedding 和一层非线性隐藏层预测下一个 token。

本阶段直接实现：

```text
Embedding → Linear → tanh → Linear → logits → softmax
```

不单独实现无隐藏层模型。

## 2. 数据样本

原始文本先切分为 train/valid/test，Tokenizer 只使用 train 建立 vocabulary。每个数据集单独编码，再构造固定宽度上下文。

例如文本 token ID 为 `[3, 4]`、`context_size=2`：

```text
[BOS, BOS] → 3
[BOS, 3]   → 4
[3, 4]     → EOS
```

所以：

```python
contexts.shape == (3, 2)
targets.shape == (3,)
```

## 3. 固定基线与形状

初始配置：

```text
vocab_size    = tokenizer.vocab_size
context_size  = 2 (定义为C)
embedding_dim = 8 (定义为D)
hidden_dim    = 32 (定义为H)
```

参数和中间值：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `embedding` | `(V, D)` | 每个 token ID 的可学习输入向量 |
| `context_ids` | `(B, C)` | batch 中的上下文 ID |
| `embedded` | `(B, C, D)` | embedding 查表结果 |
| `features` | `(B, C*D)` | 拼接后的上下文向量 |
| `W1` | `(C*D, H)` | 输入到隐藏层权重 |
| `b1` | `(H,)` | 隐藏层偏置 |
| `hidden` | `(B, H)` | `tanh` 后的隐藏状态 |
| `W2` | `(H, V)` | 隐藏层到 vocabulary 权重 |
| `b2` | `(V,)` | 每个 token ID 的输出偏置 |
| `logits` | `(B, V)` | 每个候选 token ID 的分数 |
| `probs` | `(B, V)` | 下一 token 条件概率 |

其中 `V` 是 vocabulary size，`B` 是 batch size，`C=2`，`D=8`，`H=32`。

## 4. 前向传播

```python
embedded = embedding[context_ids]
features = embedded.reshape(batch_size, context_size * embedding_dim)
z1 = features @ W1 + b1
hidden = np.tanh(z1)
logits = hidden @ W2 + b2
```

稳定 softmax：

```python
shifted = logits - logits.max(axis=1, keepdims=True)
exp_values = np.exp(shifted)
probs = exp_values / exp_values.sum(axis=1, keepdims=True)
```

## 5. Loss

对每个样本只取正确 target ID 的概率：

```text
loss = mean(-log P(target_id | context_ids))
```

测试和实现都应使用 batch 平均值，避免 loss 随样本数线性增长。

## 6. 反向传播

softmax 与交叉熵组合后的 logits 梯度：

```python
grad_logits = probs.copy()
grad_logits[np.arange(batch_size), targets] -= 1
grad_logits /= batch_size
```

随后依次计算：

```text
grad_W2、grad_b2
→ grad_hidden
→ tanh 导数
→ grad_W1、grad_b1
→ grad_features
→ grad_embedding
```

同一个 token ID 可能在一个 batch 中出现多次，embedding 梯度必须累加：

```python
np.add.at(grad_embedding, context_ids, grad_context)
```

## 7. 生成

模型输出 logits，生成模块负责 temperature、top-k 和随机采样：

```text
[BOS, BOS]
→ MLP logits
→ 采样 next_id
→ 滚动保留最后两个 ID
→ 遇到 EOS 停止
→ Tokenizer.decode
```

模型不持有字符字符串，Tokenizer 不参与神经网络计算。

## 8. 训练边界

本阶段只使用固定随机种子和全批量梯度下降。验证集用于选择最佳参数，测试集在配置固定后评估一次。单次生成样例只用于检查运行路径，不作为模型质量证明。

参数初始化只创建一个固定 seed 的随机生成器。Embedding 使用中心为 0 的小随机值，两个线性层根据输入维度缩放随机权重，偏置初始化为 0；这样可以避免所有参数整体偏正而让 `tanh` 过早进入饱和区域。

训练 loss 和验证 loss 必须分开观察：训练 loss 继续下降而验证 loss 上升表示过拟合，不表示梯度停止。模型选择以最低验证 loss 为准，保存前恢复对应参数。当前短语料下约 743 次参数更新达到最佳验证结果，详细曲线与数据稀疏性见 [运行报告](run_report.md)。

暂不包含 mini-batch、Adam、checkpoint 恢复、attention、GPU 或自动求导。
