# mini-model 项目知识库

本文把阶段 01–10 中出现的概念按知识体系重新归类。它关注概念之间的关系、公式、代码中的具体含义、验证方法和能力边界，而不是重复各阶段的开发记录。

## 1. 项目知识地图

```text
原始数据
  ↓ 切分与样本构造
文本 ──Tokenizer──> token ──词表映射──> token ID
                                                ↓ 查表
数值特征 ────────────────────────────────> embedding
                                                ↓
                 计数模型 / MLP / Transformer 前向传播
                                                ↓
                                             logits
                              ┌─────────────────┴─────────────────┐
                              ↓                                   ↓
                    训练：loss 与梯度                    推理：采样与逐步生成
                              ↓                                   ↓
                     优化器更新模型参数                  新 token、批量、流式输出
                              ↓                                   ↓
                 验证集选参与 early stopping          延迟、吞吐、内存与接口契约
                              └─────────────────┬─────────────────┘
                                                ↓
                          推理文件 / checkpoint / 实验报告
```

| 知识类别 | 核心问题 | 主要阶段 |
| --- | --- | --- |
| 监督学习基础 | 模型如何从带标签数据学习分类边界？ | 01 |
| 语言模型基础 | 如何根据已有 token 预测下一个 token？ | 02–03 |
| 文本表示 | 文本如何稳定地变成模型可处理的 ID？ | 04、09 |
| 神经网络 | embedding、线性层、激活函数和反向传播如何协作？ | 05 |
| 训练工程 | mini-batch、Adam、early stopping 和恢复训练如何实现？ | 06 |
| Transformer | attention 如何混合上下文，同时阻止未来信息泄漏？ | 07–08 |
| 推理与评估 | 如何生成、评估、加载和测量一个冻结模型？ | 10 |

## 2. 数据、任务与学习范式

### 2.1 输入、目标和样本

模型训练需要明确一次计算的输入和目标。

- 逻辑回归的输入是数值特征 `x`，目标是人工提供的二分类标签 `y∈{0,1}`。
- 语言模型的输入是已有 token，目标是原文本中的下一个 token。
- 语言模型不需要人工逐 token 标注，因为目标可以由原序列右移一位自动构造。

例如 token 序列为：

```text
[BOS, A, B, C, EOS]
```

上下文长度为 2 时，可以构造：

```text
[BOS, BOS] -> A
[BOS, A]   -> B
[A, B]     -> C
[B, C]     -> EOS
```

因此，本项目中的语言模型属于**自监督学习**：监督目标来自输入数据本身。阶段 01 的标签来自数据生成过程，属于监督学习。

### 2.2 训练集、验证集和测试集

三个数据集合承担不同职责：

| 数据集 | 可以做什么 | 不应该做什么 |
| --- | --- | --- |
| 训练集 | 计算梯度、更新参数、学习词表或 BPE merge | 报告最终泛化结论 |
| 验证集 | 选择超参数、early stopping、选出最佳 step | 继续更新模型参数 |
| 测试集 | 模型和超参数冻结后进行一次最终评估 | 参与训练、选参或反复调试 |

项目中的文本任务使用顺序 80/10/10 切分。每个文本切分独立添加 BOS/EOS，不能让训练窗口跨入验证集，也不能让验证窗口跨入测试集。

### 2.3 数据泄漏

如果测试数据参与词表学习、平滑参数选择、early stopping 或模型更新，测试结果就不再代表独立数据上的表现。常见泄漏包括：

- 用完整语料建立词表，再声称验证集含有真正的未知字符；
- 根据测试 perplexity 选择 smoothing、模型大小或训练轮数；
- 构造滑动窗口时跨越 train/validation/test 边界；
- 看到测试结果不好后继续修改模型，再把同一结果称为最终测试。

## 3. Token、Token ID 与 Tokenizer

### 3.1 字符编码和 tokenization 的区别

ASCII、Unicode 和 UTF-8 解决字符如何表示或编码为字节的问题。Tokenizer 解决模型把文本拆成哪些离散单元的问题。这两个层次有关，但不相同。

```text
文本 "hello"
  ↓ tokenizer 的切分规则
token ["h", "e", "l", "l", "o"]       # 字符 tokenizer
或    ["he", "llo"]                     # 某个 subword tokenizer
  ↓ 词表查找
token ID [12, 9, 16, 16, 19]
```

token ID 只是词表位置，是离散索引。数字大小本身没有连续语义：ID 12 不表示比 ID 9“更大”或“更相似”。

### 3.2 词表和双向映射

Tokenizer 至少需要两种映射：

- `token_to_id`：编码时把 token 转成整数 ID；
- `id_to_token`：解码时把整数 ID 还原成 token。

词表顺序必须稳定。只保存 token 集合而不保存顺序，会导致同一 ID 在加载后指向不同 token，使已经训练好的 embedding 和输出层失去意义。

共享字符 tokenizer 位于 [`projects/tokenization/char_tokenizer.py`](projects/tokenization/char_tokenizer.py)。它只使用训练文本建立普通字符词表，并通过确定性排序保证相同输入得到相同 ID。

### 3.3 特殊 token

本项目显式配置三种特殊 token：

| token | 含义 | 典型用途 |
| --- | --- | --- |
| `<BOS>` | beginning of sequence | 表示序列开始，为空提示提供初始上下文 |
| `<EOS>` | end of sequence | 训练结束目标；生成时表示正常结束 |
| `<UNK>` | unknown | 表示训练词表中不存在的输入单元 |

特殊 token 不是所有 tokenizer 都必须使用，但一旦模型训练时使用，它们的字符串、ID 和顺序就成为模型契约的一部分。

未知字符策略需要显式定义：

- `use_unk`：映射到 `<UNK>`，编码可以继续，但原字符信息丢失；
- `error`：立即报错，适合要求完全可逆的输入边界。

### 3.4 encode、decode 与可逆范围

已知文本通常满足：

```text
decode(encode(text)) == text
```

出现未知字符并映射成 `<UNK>` 后，这个关系不再成立，因为多个不同字符都变成了同一个 ID。跳过特殊 token 解码也会主动丢弃信息。因此，可逆性必须附带输入范围和解码选项。

### 3.5 字符 token 与 subword token

字符 tokenizer 的特点：

- 基础词表较小；
- 任意已知字符都能独立编码；
- 同一文本通常产生较长序列；
- 不直接表达常见字符组合。

subword tokenizer 把常见相邻单元合并成更长 token：

- 序列通常缩短；
- 词表增大；
- 常见片段可以作为一个单元；
- 词表增大会扩大 embedding 表和输出层；
- 对不匹配训练分布的文本，不保证 token 数一定减少。

### 3.6 本项目的最小 BPE

阶段 09 的实现位于 [`projects/stage09_subword_tokenizer/bpe.py`](projects/stage09_subword_tokenizer/bpe.py)，流程为：

1. 从训练文本字符和特殊 token 建立基础词表；
2. 把训练文本表示为基础 token ID 序列；
3. 统计所有相邻 pair，包括重叠出现次数；
4. 选择频率最高且达到最小频率的 pair；
5. 频率相同时，使用较小的 `(left_id, right_id)` 保证确定性；
6. 从左到右、非重叠地合并该 pair；
7. 新 token 的 ID 为基础词表长度加 merge 序号；
8. 重复到最大 merge 数或没有合格 pair；
9. 编码新文本时严格按训练得到的 merge 顺序执行。

merge 顺序不能丢失，因为后学到的 token 可能依赖前面生成的 token。BPE 的“训练”是离散统计过程，不需要梯度或自动求导。

## 4. 概率、logits、softmax 与采样

### 4.1 概率分布

对给定上下文，下一 token 的所有候选概率必须满足：

```text
p_i >= 0
Σ_i p_i = 1
```

阶段 02–03 的计数模型直接保存条件概率表。神经网络通常先输出 logits，再通过 softmax 转成概率。

### 4.2 logits

logits 是尚未归一化的实数分数：

```text
logits = [2.1, -0.3, 0.7]
```

它们可以为负数，也不需要和为 1。较大的 logit 表示模型相对更偏好对应 token。

### 4.3 softmax

softmax 把 logits 转成概率：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

直接计算较大的 `exp(z)` 可能溢出。利用 softmax 的平移不变性，实际实现先减去最大值：

```text
shifted = logits - max(logits)
probs = exp(shifted) / sum(exp(shifted))
```

这不会改变概率，只改善数值稳定性。

### 4.4 temperature

temperature 在 softmax 前缩放 logits：

```text
p = softmax(logits / temperature)
```

- `temperature < 1`：分布更尖锐，高分 token 更容易被选中；
- `temperature = 1`：保持原分布；
- `temperature > 1`：分布更平坦，随机性增加；
- temperature 必须为有限正数。

temperature 不会重新训练模型，只改变当前推理时的采样分布。

### 4.5 top-k

top-k 只保留 logit 最高的 k 个候选，把其余候选设为负无穷，再重新归一化。

- `k=1` 等价于只允许最高分 token，可视为确定性的贪心选择；
- k 越大，候选范围越广；
- top-k 过滤发生在采样前。

完整采样顺序是：

```text
原始 logits
  → 除以 temperature
  → top-k 过滤
  → softmax
  → rng.choice / torch.multinomial 抽样
```

`rng.choice` 或 `torch.multinomial` 是执行随机抽样的工具，和 temperature/top-k 不属于同一层次。

### 4.6 随机种子与可复现性

随机种子控制伪随机数生成器的初始状态。相同模型、输入、采样参数、执行顺序和随机数实现通常会产生相同结果。

项目中几种 `seed + i` 的含义不同：

- 阶段 06 用 `seed + epoch` 为每个 epoch 重建训练样本排列；
- 阶段 07–08 的一次生成创建一个 RNG，并在各生成 step 连续消费它；
- 阶段 10 的顺序批量接口用 `seed + prompt_index` 为每个请求建立独立、可复现的生成序列。

固定 seed 不等于模型“没有随机性”，只表示随机过程可以复现。

## 5. Loss、交叉熵与 Perplexity

### 5.1 负对数似然

假设正确目标 token 是 `y`，模型给它的概率是 `p_y`，单个目标的负对数似然为：

```text
NLL = -log(p_y)
```

正确 token 概率越大，loss 越小：

- `p_y = 1` 时，loss 为 0；
- `p_y` 接近 0 时，loss 很大；
- loss 衡量模型分配给真实目标的概率，不等同于 0/1 准确率。

### 5.2 交叉熵

当目标是 one-hot 类别时，交叉熵等价于正确类别的负对数似然。一个 batch 或序列的平均交叉熵为：

```text
loss = -(1/N) Σ_n log p(target_n | context_n)
```

PyTorch 的 `cross_entropy(logits, target)` 直接接收 logits，内部组合了 `log_softmax` 和 NLL。先手动 softmax 再传给它会改变含义并降低数值稳定性。

Transformer 对 `(B,T,V)` logits 计算 loss 时，把 batch 和时间位置展平，每个位置对应一个目标 token。

### 5.3 Perplexity

如果 loss 是使用自然对数、按目标 token 平均的交叉熵：

```text
perplexity = exp(loss)
```

直观上，perplexity 可以理解为模型在每一步面对的“等效候选数量”。越低通常表示模型给真实序列分配了更高概率。

使用 perplexity 时必须保持口径一致：

- tokenizer 不同，token 单位不同，数值不能直接横向比较；
- 是否包含 BOS/EOS 会改变目标集合；
- 是每个 token 计一次，还是重叠窗口重复计数，会改变权重；
- 各文本应按目标 token 数加权，不能简单平均不同长度文本的 perplexity；
- perplexity 较低不自动意味着生成文本在事实性、安全性或整体连贯性上更好。

阶段 10 的 [`evaluate_perplexity()`](projects/stage10_inference_serving/runtime.py) 对每段文本独立添加 BOS/EOS，使用滚动上下文，并让每个目标 token 只贡献一次 NLL。

### 5.4 数值稳定性

常见保护方式包括：

- softmax 前减去最大 logit；
- 直接使用 log-sum-exp 计算交叉熵；
- 拒绝 NaN 和无穷 logits、loss、梯度；
- 不先计算极小概率再取 `log`；
- 明确检查张量形状和目标 ID 范围。

## 6. 从逻辑回归到语言模型

### 6.1 逻辑回归

阶段 01 建立最小训练闭环：

```text
z = x @ w + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

二分类交叉熵为：

```text
L = -mean(y log p + (1-y) log(1-p))
```

模型学习权重 `w` 和偏置 `b`，预测时通常用阈值把概率转换成类别。逻辑回归是线性决策边界；sigmoid 只把线性分数映射到 `[0,1]`，不会让输入空间中的决策边界变成任意曲线。

这一阶段引入了贯穿整个项目的流程：

```text
初始化参数 → forward → loss → gradient → update
          → validation → 保存最佳参数 → test
```

### 6.2 Bigram

字符 bigram 使用当前字符预测下一个字符：

```text
P(x_t | x_{t-1})
```

训练不是梯度下降，而是统计转移次数：

```text
count[current, next] += 1
prob[current, :] = count[current, :] / sum(count[current, :])
```

它能学习局部相邻规律，但无法区分拥有相同最后一个字符的不同长上下文。

### 6.3 平滑smoothing

训练中未出现的转移若概率为 0，验证或测试时一旦遇到它：

```text
-log(0) = +∞
```

加法平滑先给每个计数加入一个小正数，再归一化。它避免绝对零概率，但也会把部分概率分给未见事件。平滑强度应由验证集选择，不能根据测试集选择。

### 6.4 Trigram 与稀疏性

trigram 使用前两个字符：

```text
P(x_t | x_{t-2}, x_{t-1})
```

上下文更长，表达能力增加；可能上下文数量也从 `V` 增长到 `V²`，短语料更容易稀疏。

阶段 03 对训练中未见的两字符上下文回退到最后一个字符的 bigram 分布：

```text
if (a,b) seen:
    use P(next | a,b)
else:
    use P(next | b)
```

backoff 体现了一个普遍思想：详细模型证据不足时，退回数据更充足的简单模型。

## 7. Embedding 与 MLP

### 7.1 Embedding

embedding 矩阵形状为：

```text
(vocab_size, embedding_dim)
```

token ID 用作行索引：

```text
vector = embedding[token_id]
```

token ID 本身不训练；embedding 矩阵是模型参数，会根据 loss 梯度更新。同一个 token ID 在模型内始终查到同一行参数。

可以把 token ID 类比为抽屉编号，把 embedding 向量类比为抽屉里的内容。训练改变内容，不改变抽屉编号与 token 的对应关系。

### 7.2 固定窗口 MLP

阶段 05 把多个上下文 embedding 拼接：

```text
context IDs:       (B, C)
embedding lookup:  (B, C, E)
flatten:           (B, C*E)
linear + tanh:     (B, H)
output linear:     (B, V)
```

其中：

- `B`：batch size；
- `C`：固定上下文长度；
- `E`：embedding dimension；
- `H`：hidden dimension；
- `V`：vocabulary size。

### 7.3 为什么需要非线性激活

两个没有激活函数的线性层仍等价于一个线性层：

```text
(xW1+b1)W2+b2 = x(W1W2) + (b1W2+b2)
```

中间加入 `tanh`、ReLU 等非线性后，网络才能表达更复杂的函数。这个结论适用于连续线性层，不表示所有深度模型都只是线性层；Transformer 还包含 attention、归一化和残差结构。

### 7.4 手写反向传播

反向传播使用链式法则，从 loss 沿计算图反向计算每个参数的梯度：

```text
loss
  → output logits
  → W2, b2
  → hidden activation
  → W1, b1
  → flattened embeddings
  → embedding table
```

softmax 与交叉熵组合后，对单个样本的 logits 梯度为：

```text
grad_logits = probs
grad_logits[target] -= 1
```

batch mean loss 还需除以 batch size。

embedding 反向传播需要特别处理重复 ID。同一 token 在一个 batch 中出现多次时，对应行的梯度必须累加。NumPy 实现使用 `np.add.at`，不能用普通赋值覆盖。

### 7.5 梯度检查

有限差分用参数的微小扰动近似梯度：

```text
dL/dθ ≈ [L(θ+ε) - L(θ-ε)] / (2ε)
```

它计算慢，但适合在小模型和少量参数上验证手写 backward。梯度检查通过只能说明所检查点附近的实现与数值近似一致，不能证明训练配置合理或模型能泛化。

## 8. 训练循环与优化器

### 8.1 一个训练 step

典型训练 step 为：

```text
读取一个 mini-batch
  → forward
  → 计算 loss
  → 清空上一 step 的梯度
  → backward
  → optimizer 更新参数
  → step += 1
```

PyTorch 中对应：

```python
optimizer.zero_grad(set_to_none=True)

logits = model(inputs)             # forward
loss = criterion(logits, targets)  # 计算 loss
loss.backward()                    # 计算梯度
optimizer.step()                   # 更新参数
```

梯度默认会累加，所以忘记清空梯度会把多个 step 的梯度意外相加。

### 8.2 batch、step 和 epoch

- **样本**：一个输入与目标；
- **batch**：一次参数更新共同处理的一组样本；
- **step**：通常指一次 optimizer update；
- **epoch**：训练集中的每个样本大致被使用一次。

阶段 06 的 `DeterministicBatcher` 根据全局 step 计算：

```text
epoch, batch_index = divmod(step, batches_per_epoch)
```

然后用 `seed + epoch` 生成整个 epoch 的确定性排列，再按 `batch_index` 切片。这允许恢复训练后重新得到相同 batch，不是每个 batch 都重新使用 `seed + batch_index` 打乱。

### 8.3 梯度下降与 SGD

最基本的更新为：

```text
θ <- θ - learning_rate * gradient
```

全批量梯度下降每步使用整个训练集。mini-batch SGD 每步只使用一部分样本，计算更便宜，但梯度带有采样噪声。

学习率过大可能震荡或发散，过小则收敛缓慢。一次 loss 下降不能证明学习率在完整训练中稳定。

### 8.4 Adam

Adam 为每个参数维护梯度的一阶矩和二阶矩：

```text
m_t = β1*m_{t-1} + (1-β1)*g_t
v_t = β2*v_{t-1} + (1-β2)*g_t²
```

由于初始值为 0，需要偏差修正：

```text
m_hat = m_t / (1-β1^t)
v_hat = v_t / (1-β2^t)
θ <- θ - lr * m_hat / (sqrt(v_hat)+ε)
```

Adam 根据历史梯度为各参数调整更新尺度。它不保证非凸神经网络找到全局最优解，也不能代替数据切分、验证和数值检查。

### 8.5 Early stopping 与最佳参数恢复

early stopping 一般在验证 loss 连续若干次没有达到最小改善量后停止：

```text
if valid_loss improves:
    save best parameters
    stale_count = 0
else:
    stale_count += 1

if stale_count >= patience:
    stop
```

训练停止时的最后一组参数不一定最好，因此返回或保存模型前需要恢复最佳验证参数。测试集只在模型选择完成后评估。

### 8.6 训练可复现性

可复现训练至少需要固定：

- 参数初始化 seed；
- 数据切分；
- batch 顺序；
- 采样和其他随机操作；
- 超参数；
- checkpoint 中的 step、优化器状态和必要随机状态；
- 软件与硬件中可能影响确定性的实现。

相同 seed 只是必要条件之一。代码路径、库版本或并行计算方式变化仍可能改变结果。

## 9. Transformer

### 9.1 输入和输出契约

Transformer 章节统一使用以下维度符号：

| 符号 | 英文 | 含义 | 当前默认值 |
| --- | --- | --- | --- |
| `B` | batch size | 一次 forward 同时处理的序列数量 | 由输入决定 |
| `T` | sequence length | 当前输入序列的 token 数 | `1 <= T <= 8` |
| `T_max` | block size | 模型支持的最大上下文长度 | `8` |
| `V` | vocabulary size | tokenizer 词表大小，也是 logits 最后一维 | `40` |
| `D` | model dimension | 每个 token 在模型内部的特征维度 | `16` |
| `H` | number of heads | attention head 数量 | `4` |
| `d_head` | head dimension | 每个 head 的维度，`D/H` | `4` |
| `F` | feed-forward dimension | FFN 中间隐藏维度 | `32` |
| `L` | number of layers | Transformer block 数量 | `2` |

这里的 `V` 表示 vocabulary size。attention 公式中的 Value 也常写成大写 `V`，两者只是符号相同：前者是整数维度，后者是输入投影得到的张量。下文用 `Value` 表示 attention 的 value 张量。

字符 Transformer 的核心接口为：

```text
token IDs: (B, T)
    ↓ model.forward
logits:    (B, T, V)
```

每个位置的 `(V,)` logits 预测该位置之后的下一个 token。训练时可以一次计算所有位置；生成时只使用最后一个位置的 logits。

### 9.2 完整前向形状

阶段 08 的一次前向传播可以按形状展开为：

```text
token_ids                                      (B, T)
  ↓ token embedding + position embedding
x                                              (B, T, D)
  ↓ qkv linear
combined qkv                                   (B, T, 3D)
  ↓ chunk
query / key / value                            各 (B, T, D)
  ↓ split H heads and transpose
Q / K / Value                                  各 (B, H, T, d_head)
  ↓ Q @ K.transpose(-2, -1)
attention scores                               (B, H, T, T)
  ↓ causal mask + softmax
attention weights                              (B, H, T, T)
  ↓ weights @ Value
per-head context                               (B, H, T, d_head)
  ↓ transpose + concatenate heads
merged context                                 (B, T, D)
  ↓ attention output projection + residual
attention sublayer output                      (B, T, D)
  ↓ FFN first linear
FFN hidden                                     (B, T, F)
  ↓ activation + second linear + residual
Transformer block output                       (B, T, D)
  ↓ repeat L blocks + final LayerNorm
normalized hidden                              (B, T, D)
  ↓ vocabulary projection
logits                                         (B, T, V)
```

除 attention scores/weights 和 FFN 中间层外，残差主干始终保持 `(B,T,D)`，这样子层输出才能与输入逐元素相加。

### 9.3 Token embedding 与位置 embedding

attention 本身不包含序列先后概念，因此输入由两部分相加：

```text
x[b,t] = token_embedding[token_id[b,t]] + position_embedding[t]
```

各张量形状为：

```text
token_ids                         (B, T)
token_embedding.weight            (V, D)
token_embedding(token_ids)        (B, T, D)
position IDs                      (T,)
position_embedding.weight         (T_max, D)
position_embedding(position IDs)  (T, D)
相加后的 x                        (B, T, D)
```

位置 embedding 的 `(T,D)` 会在 batch 轴上广播，为每条序列使用同一组位置向量。

- token embedding 表示“这个 token 是什么”；
- position embedding 表示“它位于序列什么位置”。

本项目使用长度为 `block_size` 的可学习位置 embedding，因此一次 forward 的序列长度不能超过 `T_max=block_size`。

### 9.4 Query、Key 和 Value

输入向量分别经过三个投影：

```text
Q = X W_Q
K = X W_K
Value = X W_V
```

用独立投影矩阵表示时：

```text
X                         (B, T, D)
W_Q / W_K / W_V           各 (D, D)
Q / K / Value             各 (B, T, D)
```

阶段 08 为减少三次线性层调用，使用一个 `D → 3D` 的 `qkv` 线性层，再沿最后一维切成三个 `(B,T,D)` 张量。切分后 reshape 并交换轴：

```text
(B, T, D)
  → (B, T, H, d_head)
  → (B, H, T, d_head)
```

可以把它们理解为：

- Query：当前位置想寻找什么；
- Key：每个位置提供什么匹配特征；
- Value：匹配后实际聚合什么信息。

这是帮助理解的类比，计算本身仍是可训练矩阵投影。

### 9.5 Scaled dot-product attention

单个 head 的计算为：

```text
scores  = Q K^T / sqrt(d_head)
weights = softmax(scores + mask)
output  = weights Value
```

加入 batch 和 head 轴后的形状运算是：

```text
Q                                      (B, H, T, d_head)
K.transpose(-2,-1)                     (B, H, d_head, T)
Q @ K.transpose(-2,-1)                 (B, H, T, T)
mask                                   (T, T)，广播到 B 和 H
weights                                (B, H, T, T)
Value                                  (B, H, T, d_head)
weights @ Value                        (B, H, T, d_head)
```

scores 最后的两个 `T` 含义不同：倒数第二维是 query 位置，最后一维是被查询的 key 位置。因此 `scores[b,h,i,j]` 表示 batch `b`、head `h` 中，位置 `i` 对位置 `j` 的注意力分数。

除以 `sqrt(d_head)` 是为了防止维度较大时点积幅值过大，使 softmax 过度饱和、梯度变小。

attention weights 在最后一个轴归一化。每个 query 位置得到一组对可见 key 位置的权重，输出是对应 Value 的加权和。

### 9.6 因果遮罩

自回归模型训练位置 `i` 时不能读取未来位置 `j>i`。加法遮罩为：

```text
mask[i,j] = 0      if j <= i
mask[i,j] = -inf   if j > i
```

mask 的基础形状是 `(T,T)`。同一个因果结构会广播到所有 `B` 条序列和所有 `H` 个 head，不会改变 scores 的 `(B,H,T,T)` 形状。

mask 加在 attention scores 上，不加在 embedding 上。经过 softmax 后，未来位置权重变为 0。对角线保留，因此一个位置可以看到当前位置及之前的位置。

因果 mask 让训练时的并行计算保持与逐步生成相同的信息边界。它防止的是未来 token 泄漏，不负责处理 padding；本项目没有实现独立 padding mask。

### 9.7 Multi-head attention

模型维度 `D` 被拆成 `H` 个 head：

```text
d_head = D / H
Q/K/Value: (B,T,D) → (B,H,T,d_head)
```

每个 head 独立计算 attention，结果再按以下过程拼接：

```text
head context                 (B, H, T, d_head)
transpose                    (B, T, H, d_head)
reshape，H*d_head=D          (B, T, D)
output projection            (B, T, D)
```

多头允许不同投影子空间学习不同的匹配方式，但不保证每个 head 自动形成可人工命名的语义。

`D` 必须能被 `H` 整除。

### 9.8 残差连接

残差结构把子层输出加回原输入：

```text
y = x + sublayer(x)
```

本项目中相加两侧均为 `(B,T,D)`。它保留直接的信息和梯度路径，让深层网络更容易优化。若 attention 输出仍是 `(B,H,T,d_head)`，必须先合并 heads 并投影回 `(B,T,D)`，才能进行残差相加。

### 9.9 Layer Normalization

LayerNorm 对每个 token 位置的特征维度进行归一化，再应用可训练缩放和平移：

```text
normalized = (x - mean) / sqrt(variance + epsilon)
output = gamma * normalized + beta
```

输入和输出均为 `(B,T,D)`。LayerNorm 对每个 `(b,t)` 的最后一个 `D` 维向量独立计算均值和方差；`gamma`、`beta` 的形状都是 `(D,)`，广播到 B 和 T。它与 BatchNorm 的统计轴和运行方式不同，不依赖整个 batch 的运行均值。

### 9.10 Feed-forward network

Transformer block 中的 MLP 对每个位置独立应用：

```text
FFN(x) = Linear2(activation(Linear1(x)))
```

形状变化为：

```text
x                         (B, T, D)
Linear1: D → F            (B, T, F)
activation                (B, T, F)
Linear2: F → D            (B, T, D)
```

attention 在 token 位置之间混合信息；FFN 在每个位置内部变换特征。两者职责不同。

### 9.11 Pre-LN Transformer block

阶段 07–08 使用 Pre-LN：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

完整 block 中的 shape 保持关系为：

```text
x                                             (B, T, D)
LayerNorm(x)                                  (B, T, D)
Attention(LayerNorm(x))                       (B, T, D)
x + Attention(...)                            (B, T, D)
LayerNorm(...)                                (B, T, D)
FFN 隐藏层                                    (B, T, F)
FFN 输出                                      (B, T, D)
第二次残差后的 block 输出                     (B, T, D)
```

经过 `L` 个 block 不会改变外部形状，`L` 表示重复多少次结构，不是新增张量轴。

多层 block 后再做 final LayerNorm 和词表投影：

```text
normalized hidden                            (B, T, D)
vocabulary projection weight                 (V, D)  # PyTorch Linear 存储形式
logits                                       (B, T, V)
targets                                      (B, T)
flattened logits                             (B*T, V)
flattened targets                            (B*T,)
cross entropy                                标量 ()
```

数学公式常把投影权重写成 `(D,V)` 并计算 `x @ W`；PyTorch `nn.Linear(D,V)` 内部把权重存为 `(V,D)`，计算等价于 `x @ weight.T + bias`。

### 9.12 block size 与滚动上下文

`block_size` 是模型一次能接收的最大上下文 token 数。生成序列超过它时，模型只接收最近的窗口：

```text
context = generated_ids[-block_size:]
```

生成时通常 `B=1`，而 `T` 会从 prompt 长度开始增长，直到达到 `T_max`。超过后，输入形状保持 `(1,T_max)`，窗口内容向前滚动：

```text
未达到上限：context shape = (1, current_length)
达到上限后：context shape = (1, T_max)
单步 logits：             (1, T, V)
用于采样的最后位置：      logits[0,-1,:]，形状 (V,)
```

完整已生成序列仍保留用于最终输出，但较早 token 不再直接出现在模型输入中。

## 10. 手写 NumPy 与 PyTorch 的分工

### 10.1 手写实现的价值

阶段 05 和 07 手写 forward/backward，使以下细节显式可见：

- 中间张量及形状；
- 链式法则；
- 重复 embedding ID 的梯度累加；
- attention softmax 的反向传播；
- 每个参数如何获得梯度；
- 数值梯度如何验证解析梯度。

### 10.2 PyTorch 自动求导

阶段 08 中，开发者仍需定义：

- 模型结构和 `forward()`；
- 输入、目标和 loss；
- optimizer、学习率和训练循环；
- 何时切换 `train()` / `eval()`；
- 保存加载与评估边界。

PyTorch 根据 forward 中构建的计算图，在 `loss.backward()` 时计算梯度。自动求导减少手写导数工作，但不会自动判断任务定义、数据泄漏、模型结构或指标口径是否正确。

### 10.3 `model(context)`、`forward()` 和 `loss()`

PyTorch 的 `nn.Module.__call__` 包装了 hook 等框架逻辑，然后调用模型定义的 `forward()`。因此：

```python
logits = model(context)
```

最终执行的是 `forward(context)`，返回 logits。

生成时没有已知 target，只需要 logits 进行采样，因此不调用 `model.loss()`。训练和固定文本评估有 target，才用 logits 与 target 计算交叉熵。

### 10.4 train、eval 与 inference mode

- `model.train()`：切换到训练模式；
- `model.eval()`：切换到评估模式；
- `torch.inference_mode()`：关闭 autograd 记录，减少推理开销。

`eval()` 不等于关闭梯度，它主要影响 dropout、BatchNorm 等具有训练/推理差异的模块。本项目模型当前没有 dropout，但运行时仍保存并恢复调用前的 training 状态，保持接口行为完整。

## 11. 自回归生成

### 11.1 逐步生成算法

```text
prompt
  → tokenizer.encode(add_bos=True)
  → 取最后 block_size 个 ID 作为 context
  → model(context)
  → 取 logits[0,-1,:] (最后一个位置的 logits)
  → temperature / top-k / softmax
  → 采样 next_id
  → 追加到 generated_ids
  → 如果是 EOS 则停止，否则重复
```

为什么取最后一个位置：对于输入 `[BOS, A, B]`，各位置 logits 分别对应在各自上下文边界后的预测。当前要生成的是 `B` 后面的 token，所以使用最后一个位置。

下一步必须把新 token 加入上下文。若已从 `AB` 生成 `C`，下一次预测应基于 `ABC`，而不是再次基于 `AB`。

### 11.2 BOS 是否加入 prompt

文本提示通常不含显式特殊 token，因此运行时编码时加入 BOS：

```text
"ab" → [BOS, a, b]
```

一次请求只在开头加入一次 BOS。每生成一个新 token 时不再添加 BOS。

### 11.3 EOS 与终止原因

生成可能因为两种原因结束：

- 采样到 EOS：`finish_reason="eos"`；
- 达到 `max_new_tokens`：`finish_reason="length"`。

EOS 用于控制停止，不作为普通生成文本显示。prompt token 数不计入 `max_new_tokens`。

### 11.4 原始输入、生成文本与 token 数

阶段 10 的 `GenerationResult` 分开记录：

- `input_text`：调用者提供的原始提示；
- `generated_text`：只包含新增文本；
- `finish_reason`：EOS 或长度限制；
- `generated_token_count`：实际新增的非 EOS token 数。

不能总从显示文本长度反推 token 数。一个 `<UNK>` 是一个 token，但字符串有五个字符；一个 subword token 也可能解码为多个字符。

### 11.5 顺序批量生成

当前 `generate_batch()` 是语义批量接口：按顺序对每个 prompt 调用单条 `generate()`，并使用 `seed + prompt_index`。它保证输入输出顺序和单条调用语义，但不是把多个变长请求 padding 后组成一次张量 forward，因此不能代表真正的 GPU batch 性能。

### 11.6 流式生成与 `yield`

普通函数执行到 `return` 后一次返回结果。函数体中出现 `yield` 后，它成为生成器函数：

- 调用函数只创建迭代器，不立即执行完整生成；
- 每次迭代运行到下一个 `yield`；
- `yield fragment` 把当前片段交给调用者并暂停；
- 下一次迭代从暂停位置继续；
- `return` 或函数结束触发 `StopIteration`。

`stream_generate()` 每得到一个普通 token 就解码并 `yield`，遇到 EOS 直接结束。同配置和 seed 下：

```text
"".join(stream_generate(prompt)) == generate(prompt).generated_text
```

流式接口降低首片段等待时间，并不减少模型逐 token forward 的总计算量。

## 12. 模型评估

### 12.1 质量评估与性能评估

两类评估回答不同问题：

| 类别 | 典型指标 | 回答的问题 |
| --- | --- | --- |
| 质量评估 | loss、perplexity、准确率、生成样例 | 模型预测是否有效？ |
| 性能评估 | 加载时间、延迟、吞吐、内存 | 模型运行代价是多少？ |

准确率必须定义具体任务。例如逻辑回归使用分类 accuracy；语言模型可以定义 next-token top-1 accuracy，但本项目主要使用交叉熵和 perplexity。

### 12.2 生成样例的作用

固定 prompt、seed 和采样参数的生成样例适合：

- 检查保存加载前后行为；
- 检查流式与非流式一致性；
- 观察模型是否只输出特殊 token 或明显异常文本；
- 展示当前模型能产生什么。

单个样例不能证明总体质量提升。样例变化可能来自随机 seed 或采样参数，而不是模型本身改善。

### 12.3 加载、首请求和预热

阶段 10 分开记录：

1. **加载耗时**：读取推理文件、构造模型和 tokenizer；
2. **未预热首请求**：加载后第一次生成；
3. **预热**：正式测量前运行若干次，不进入统计；
4. **预热后延迟**：对固定工作负载重复测量。

预热循环不是冷启动测量。当前加载测量也没有清理操作系统文件缓存，所以它是当前进程和机器状态下的加载时间，不是严格的硬件全冷启动。

### 12.4 延迟、吞吐和分位数

- 延迟：完成一次请求或一次固定 batch 花费的时间；
- 吞吐：单位时间生成的 token 数；
- P50：一半测量值不超过它；
- P95：约 95% 测量值不超过它；
- 标准差：观察多次测量的离散程度。

报告必须同时给出 prompt 数、输入 token 数、最大生成长度、预热次数、正式重复次数、设备和软件环境。否则数字无法解释或复现。

### 12.5 内存口径

模型参数字节数可以精确计算：

```text
Σ parameter.numel() * parameter.element_size()
```

进程生命周期峰值 RSS 包含 Python、PyTorch、模型、评估和此前分配的内存，不等于模型独占峰值。报告内存时必须说明测量范围。

### 12.6 当前性能结论的边界

阶段 10 报告只描述当前字符级小模型、CPU、固定顺序批量和本机环境。它不能外推到：

- 大语言模型；
- GPU kernel 性能；
- 真正的张量 batch；
- 多用户并发；
- 网络协议与序列化开销；
- 生产服务的 P95/P99 稳定性。

## 13. 保存、加载与版本化

### 13.1 四种产物

| 产物 | 主要内容 | 主要用途 |
| --- | --- | --- |
| Tokenizer 文件 | 格式、版本、词表、特殊 token、merge 规则 | 稳定文本与 ID 映射 |
| 推理模型 | 格式、版本、模型 config、state_dict、tokenizer 契约 | 不依赖训练代码执行推理 |
| 训练 checkpoint | 当前参数、最佳参数、optimizer 状态、step、训练配置 | 中断后连续恢复训练 |
| 实验报告 | 配置、loss、perplexity、best step、耗时、样例 | 记录证据，不参与推理计算 |

这四者不能混为一谈。尤其是：

- config 描述结构，如层数和维度；
- state_dict 保存训练得到的参数；
- optimizer state 不是模型前向参数，但恢复训练需要；
- best step 和 loss 是实验元数据，不是恢复 forward 的必要数据。

### 13.2 为什么模型必须绑定 tokenizer

模型的 embedding 第 `i` 行和输出层第 `i` 个 logit 都对应训练时词表的第 `i` 个 token。即使两个 tokenizer 的词表大小相同，只要顺序不同，模型含义就完全错位。

因此阶段 08 的自包含推理文件同时保存：

- `format`；
- `version`；
- `config`；
- `state_dict`；
- tokenizer 的有序 tokens 和特殊 token 定义。

### 13.3 format 与 version

- `format` 标识产物类型，防止把 tokenizer、checkpoint 或其他 `.pt` 文件误当成模型；
- `version` 标识该格式的结构版本，供未来升级时拒绝或迁移不兼容文件。

格式和版本检查属于理解该文件格式的模型加载器。运行时调用加载器并传播清楚的兼容错误，无需重复解析底层字段。

### 13.4 推理不依赖训练状态

冻结模型推理只需：

```text
模型结构 + 最终参数 + 完全匹配的 tokenizer
```

它不需要训练语料、梯度、optimizer、当前 step 或 Adam 动量。评估 perplexity 需要显式提供评估文本，但仍不执行参数更新。

## 14. 测试与可信证据

### 14.1 单元测试验证什么

项目测试覆盖的典型契约包括：

- 数值：概率归一化、有限 loss、已知交叉熵；
- 形状：embedding、attention、logits 和梯度维度；
- 边界：空输入、非法 ID、错误 dtype、长度上限；
- 因果性：修改未来 token 不影响过去位置输出；
- 梯度：解析梯度与有限差分一致；
- 训练：小数据可以过拟合、loss 能下降；
- 数据：切分不重叠、词表只用训练集；
- 持久化：保存加载前后 logits 或编码结果一致；
- 可恢复性：中断恢复与连续训练逐元素一致；
- 生成：滚动上下文、EOS、长度限制、固定 seed；
- 运行时：单条、批量、流式、perplexity 和 benchmark 语义。

### 14.2 测试不能证明什么

测试通过不等于：

- 训练数据足够代表真实世界；
- 模型具有大模型级语言能力；
- 生成事实正确或安全；
- 所有未测试输入都正确；
- 当前 CPU 性能可以代表生产环境；
- 更复杂模型一定优于简单基线。

### 14.3 证据层次

一个较完整的结论应区分：

1. **数学定义**：公式是否正确；
2. **实现契约**：输入、输出、形状和错误是否明确；
3. **自动化测试**：关键行为是否有可重复检查；
4. **实际运行**：固定配置下得到什么数值；
5. **能力边界**：这些证据不能支持哪些外推。

## 15. 容易混淆的概念

| 容易混淆 | 正确关系 |
| --- | --- |
| 字符编码与 tokenizer | 字符编码定义字符/字节表示；tokenizer 定义模型单元及其 ID |
| token ID 与 embedding | ID 是离散索引；embedding 是可训练连续向量 |
| logits 与 probability | logits 未归一化；softmax 后才是概率 |
| temperature/top-k 与 choice | 前两者调整候选分布；choice/multinomial 执行抽样 |
| forward 与 loss | forward 产生预测；loss 还需要 target 衡量预测误差 |
| `model(x)` 与 `model.loss()` | 前者经 `__call__` 调用 forward；后者组合 forward 与目标损失 |
| backward 与 optimizer | backward 计算梯度；optimizer 根据梯度更新参数 |
| batch step 与生成 step | batch step 更新一次参数；生成 step 追加一个 token，不更新参数 |
| epoch seed 与请求 seed | 前者控制训练排列；后者控制一次生成的采样序列 |
| causal mask 与 embedding mask | causal mask 加到 attention scores，阻止读取未来位置 |
| validation 与 test | validation 用于选择；test 用于冻结后的最终评估 |
| early stopping 与立即停止 | 通常等待 patience 次未改善，并恢复最佳参数 |
| 推理模型与 checkpoint | 推理模型服务 forward；checkpoint 还要恢复训练过程 |
| perplexity 与生成质量 | perplexity 衡量目标 token 概率；样例观察另一部分行为 |
| batch API 与张量 batch | 顺序处理多个请求也可形成 API，但不等于一次并行 forward |
| 流式输出与并行生成 | streaming 提前交付片段；自回归 token 依然逐步计算 |
| 参数内存与进程 RSS | 前者只计算模型参数；后者包含整个进程 |

## 16. 阶段与代码导航

| 阶段 | 核心概念 | 主要入口 |
| --- | --- | --- |
| 01 | 逻辑回归、sigmoid、二元交叉熵、梯度下降 | [`projects/stage01_logistic_regression`](projects/stage01_logistic_regression/) |
| 02 | bigram、计数概率、平滑、采样 | [`projects/stage02_char_bigram/model.py`](projects/stage02_char_bigram/model.py) |
| 03 | trigram、上下文稀疏、bigram backoff | [`projects/stage03_char_trigram/model.py`](projects/stage03_char_trigram/model.py) |
| 04 | 字符词表、token ID、特殊 token、encode/decode | [`projects/tokenization/char_tokenizer.py`](projects/tokenization/char_tokenizer.py) |
| 05 | embedding、MLP、手写 backward、梯度检查 | [`projects/stage05_token_mlp/model.py`](projects/stage05_token_mlp/model.py) |
| 06 | mini-batch、SGD、Adam、early stopping、checkpoint | [`projects/stage06_training_engineering`](projects/stage06_training_engineering/) |
| 07 | causal attention、Transformer、手写梯度、生成 | [`projects/stage07_char_transformer`](projects/stage07_char_transformer/) |
| 08 | `nn.Module`、autograd、PyTorch optimizer、推理 bundle | [`projects/stage08_pytorch_transformer`](projects/stage08_pytorch_transformer/) |
| 09 | BPE pair 统计、merge、subword、版本化 tokenizer | [`projects/stage09_subword_tokenizer/bpe.py`](projects/stage09_subword_tokenizer/bpe.py) |
| 10 | perplexity、批量、流式、加载、benchmark、CLI | [`projects/stage10_inference_serving`](projects/stage10_inference_serving/) |

## 17. 术语速查

| 术语 | 简明定义 |
| --- | --- |
| feature | 提供给模型的数值输入属性 |
| label / target | 模型需要预测的真实答案 |
| vocabulary | tokenizer 可以表示的全部 token 及固定顺序 |
| token | tokenizer 定义的离散文本单元 |
| token ID | token 在词表中的整数索引 |
| embedding | token ID 查表得到的可训练连续向量 |
| parameter | 由训练更新、参与模型计算的数值 |
| hyperparameter | 人工配置而非直接由梯度学习的值，如学习率、层数 |
| logit | softmax 前的未归一化分数 |
| probability | 非负且总和为 1 的归一化置信分布 |
| loss | 衡量当前预测与目标差异的训练标量 |
| gradient | loss 对参数的局部变化率 |
| backpropagation | 用链式法则高效计算各参数梯度 |
| optimizer | 根据梯度及内部状态更新参数的算法 |
| batch | 一次共同计算 loss 和梯度的样本集合 |
| step | 一次优化器参数更新；在生成中也可指一次 token 追加，需看上下文 |
| epoch | 训练样本被完整遍历一次 |
| context | 当前预测允许使用的已有 token |
| autoregressive | 把此前 token 作为上下文，逐步预测下一 token |
| attention | 根据 query-key 匹配权重聚合 value 的机制 |
| causal | 当前位置不能读取未来 token 的信息约束 |
| residual | 把子层结果与原输入相加的连接 |
| checkpoint | 为恢复训练保存的完整训练状态 |
| inference artifact | 为冻结模型推理保存的最小兼容状态 |
| latency | 完成一次规定工作负载的时间 |
| throughput | 单位时间完成的样本数或生成 token 数 |
| RSS | 操作系统观测到的进程驻留内存规模 |

## 18. 自测问题

完成本项目后，应能够不依赖代码回答以下问题：

1. token ID 为什么不能直接当作有大小语义的数值特征？ 
token ID 是词表位置，离散索引

2. tokenizer 词表顺序变化为什么会让同一模型失效？
词表顺序变化不仅影响解码，还会让 embedding 第 i 行和 logits 第 i 项对应错误 token。

3. 一个 `<UNK>` token 为什么不能用显示字符串长度统计 token 数？
token 数量由 tokenizer 的词表和编码结果决定，不是由解码后的字符串长度决定
特殊token，解码后长度为5，但token id = 0，token数为1
且subword也有同样问题，一个id对应的token解码后长度>=1

4. bigram、trigram、固定窗口 MLP 和 Transformer 的上下文能力分别是什么？
bigram：前1个token，查二维概率表
trigram：前2个token，查三维概率表
fixed window MLP：前C个token，C=窗口大小，embedding拼接后送入MLP
transformer：前T_max个token，T_max表示允许的最大上下文长度，attention动态聚合所有可见位置

5. 平滑和 backoff 分别解决什么问题？
平滑：当概率为0时，给一个很小的概率，避免0概率，项目中给计数加入 smoothing，再归一化
backoff：当trigram没有见过当前这一对2-token上下文时，退回bigram，即详细模型证据不足时，退回数据更充足的简单模型

6. logits 为什么可以为负，softmax 为什么先减最大值？
logits由forward计算获得，是未归一化的实数分数，可以为负，因为后面要进行softmax
softmax先减最大值，是因为浮点数的范围有限，太大可能溢出，而且softmax=exp(x)/sum(exp(x))利用的是x之间的相对大小关系，所以减最大值不影响总结果

7. temperature、top-k 和随机抽样分别发生在哪一步？
temperature先对logits执行logits/temperature
然后选择 top-k 个最大 logit，其他位置设置为-inf
然后softmax后进行随机抽样，rng.choice(numpy) / torch.multinomial抽取一个token id

8. 交叉熵为什么是正确目标概率的负对数？
交叉熵计算的是分配到正确目标上的概率，概率越高，则loss越小。
one-hot 目标下，交叉熵求和后只剩正确类别的 -ln(p_target)；负号使最大化正确目标概率等价于最小化loss

9. 什么条件下 `perplexity = exp(loss)`？
loss是使用自然对数、按目标token平均的交叉熵

10. embedding 的 ID、向量和梯度分别是什么？
embedding是个矩阵，所有 token 向量组成的可训练参数
embedding的ID是行索引，即token id
向量是embedding_matrix[token_id]，即矩阵中的一行
梯度是loss对embedding矩阵的梯度

11. 为什么连续线性层之间需要非线性才能提高函数表达能力？
两个连续线性层如果中间没有非线性：(xW1+b1)W2+b2 = x(W1W2)+b1W2+b2，整体仍等价于一个线性层。加入非线性（激活函数）后，组合不再能合并为一个线性变换，因此能够表达更复杂的非线性关系。

12. `loss.backward()` 和 `optimizer.step()` 分别做什么？
loss.backward() 计算loss对每个参数的梯度，并保存在参数的.grad中，执行完参数不变
optimizer.step() 读取这些.grad，将其更新到参数中，参数改变

13. 当前项目为什么使用 `seed + epoch`，而不是每个 batch 使用 `seed + step`？
因为当前项目epoch定义为一个epoch遍历一次训练集，期望每个epoch内，每个训练样本恰好使用一次。所以随机打乱的对象是整个epoch中的样本顺序，而不是每个 step 使用独立 seed 重新生成一份排列。使用seed+step, 每个step独立生成batch后，无法自然保证一个epoch内所有样本无重复、无遗漏。
```text
每个 step：
  计算 epoch 和 batch_index
  检查当前 epoch 的排列是否已经缓存
  ├─ 已缓存：直接复用
  └─ 未缓存：用 seed+epoch 生成一次并缓存
  根据 batch_index 从缓存排列中切出 batch
```


14. early stopping 为什么还需要恢复最佳参数？
early stopping的定义为x次后没有再优化，但最佳参数不一定是early stopping时的参数，所以需要恢复。

15. attention 中 Q、K、V 的形状和作用是什么？
Q、K、Value 投影后形状都是 (B,T,D)。拆分多头并转置后，形状都是 (B,H,T,d_head)。
Query：当前位置希望匹配什么特征；
Key：每个可见位置提供什么匹配特征；
Value：匹配成功后，需要从该位置聚合什么信息。
Q @ K^T 得到 (B,H,T,T) 的 attention scores，softmax 后的权重与 Value 相乘，得到 (B,H,T,d_head)。

16. causal mask 为什么加在 attention scores 上？对角线是否遮罩？
attention scores 是一个矩阵，定义的是每个位置的受关注分数，分数越大越关注
causal mask定义对角线右上角设置为-inf，其他为0，加在attention scores上，是为了设置其右上角的值为-inf，即让模型只能关注当前位置及以前的位置，不能关注未来的位置
对角线上不遮罩

17. 多头 attention 为什么要求 `model_dim % num_heads == 0`？
因为每个head的QKV特征维度为 model_dim / num_heads，model_dim % num_heads == 0可以保证其为整数，需要整除，才能把 D 个特征均匀 reshape 成 H 个 head。

18. 训练时为什么能并行预测所有位置，生成时却仍要逐 token 运行？
训练时完整输入和所有目标 token 都已知，因果 mask 保证每个位置只能读取允许的历史信息，因此所有位置可以在一次 forward 中并行计算。
生成时下一个 token 尚未确定，必须先生成它，才能组成下一步输入。

19. `model(context)` 为什么用于生成，而 `model.loss(context,target)` 用于训练？
`model(context)`执行的是model.forward，返回logits，用于生成；
`model.loss(context,target)`执行model.forward，返回loss = cross_entropy(logits,target)，用于训练;

20. prompt 是 `ab` 时，第一次生成的 context 为什么通常是 `[BOS,a,b]`？
prompt 本身不包含序列开始标记，因此 encode(add_bos=True)在开头加入一次 BOS。后续生成不会重复添加 BOS，但原来的 BOS 仍保留在上下文中，直到被滚动窗口截掉

21. 已生成 `abc` 后，下一次 forward 的输入为什么应包含 `c`？
因为生成时，需要根据已有的context生成下一个token，所以需要将已有的context作为输入。

22. 顺序 `generate_batch()` 与真正张量 batch 有什么区别？
当前项目的 generate_batch() 是接口层面的批量，本质上仍然是依次执行多个单条请求。
真正的张量 batch 是把多条输入组成一个带 B 维的 tensor，在一次模型 forward 中并行计算。

23. `yield` 如何让调用者在完整生成结束前收到片段？
包含 yield 的函数是生成器函数。调用它返回生成器对象；每次 next() 运行到下一个 yield，产出一个值并暂停，下次从暂停位置继续。

24. 推理文件、训练 checkpoint 和实验报告各自必须保存什么？
推理文件：用于冻结模型以便以后使用，保存以后执行推理所需的模型状态，包括format，version，config，state_dict，tokenizer等。
训练 checkpoint：用于恢复训练，保存训练过程中的相关模型参数和运行时状态，除了模型状态外，还包括运行时状态，如optimizer状态，当前step，训练状态，最佳参数等
实验报告：记录模型配置、训练配置、训练历史、train/validation loss、best step、冻结后的 test loss/perplexity、训练耗时和固定生成样例。

25. 为什么预热循环不属于冷启动时间？
冷启动时间是模型加载时间，预热循环时模型已经加载完成，属于运行阶段

26. 为什么进程峰值 RSS 不能称为模型独占内存？
进程峰值 RSS 包含：
- Python 解释器；
- PyTorch 运行时；
- 模型参数；
- 中间 tensor；
- tokenizer；
- allocator 缓存；
- 评估过程；
- 进程此前产生的内存分配。
并且当前记录的是进程生命周期峰值，不是模型加载前后内存差。

27. BPE 为什么必须保存有序 merge，而不只保存最终词表？
BPE 训练按顺序学习 merge，后面的 merge 可能依赖前面创建的新 token。最终词表只保存有哪些 token，不能唯一恢复合并顺序。因此必须保存有序 merge，并在 encode() 时按相同顺序执行，才能得到与训练时一致且确定的 token ID 序列。

28. token 数减少为什么不自动证明模型更好或端到端速度更快？
token 数减少会缩短序列；
但词表增大会扩大 embedding 和输出层；
新 tokenizer 需要与语言模型共同重新训练；
tokenizer 编码本身有计算成本；
实际速度取决于硬件、矩阵形状和实现；
token 数变化没有提供 loss 或生成质量证据；
某类文本 token 数减少，不保证其他文本也减少。

29. 自动化测试通过后，还需要哪些实际运行证据和能力边界？
自动化测试通过，只表示目前设计的测试样例通过，在这些覆盖的输入和条件下，代码行为符合预期，并不能证明真实环境下的正确性，还需要实际运行证据和能力边界。
实际运行证据：数据证据、配置证据、环境证据、运行报告、固定生成样例、性能报告、保存和加载的正确性、运行与恢复的稳定性
能力边界：测试用例覆盖能力，还包括：
- 短语料不能代表外部泛化；
- 当前小模型不能代表大语言模型；
- 字符 tokenizer 和最小 BPE 不是生产 tokenizer；
- perplexity 不能代替生成质量和事实正确性；
- CPU benchmark 不能代表 GPU；
- 顺序 batch 不能代表张量 batch；
- 进程 RSS 不是模型独占内存；
- 当前 CLI 不是并发网络服务；
- 没有覆盖分布式训练、量化、KV cache、连续批处理等能力。

## 19. 项目能力边界

本项目完成的是可验证的教学闭环：从概率、梯度和 tokenizer，逐步进入 Transformer、自动求导、推理评估和性能测量。当前证据支持理解这些机制及其小规模实现。

当前项目没有证明：

- 模型具有通用语言理解或可靠事实生成能力；
- 短语料上的 loss 可以代表外部语料表现；
- 字符 tokenizer 或当前 BPE 适合生产模型；
- 单机 CPU benchmark 可以代表 GPU 或在线服务；
- 当前实现覆盖分布式训练、混合精度、KV cache、连续批处理、量化、模型并行或容错服务；
- 教学实现经过生产安全、稳定性和兼容性验证。

理解这些边界也是知识体系的一部分：模型结构正确、测试通过、单次指标良好和生产可用，是四个不同层次的结论。