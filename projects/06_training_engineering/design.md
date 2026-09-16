# 06 训练工程基础设计

## 1. 阶段边界

阶段 06 继续使用字符 Tokenizer、长度为 2 的上下文和 `Embedding → Linear → tanh → Linear` 模型。新增内容全部属于训练系统：数据顺序、优化器、验证、恢复和观测。

```text
训练数据 → 确定性 mini-batch → loss/gradients → optimizer → 当前参数
                                      ↓
验证数据 → early stopping → 最佳参数
                                      ↓
checkpoint / 推理模型 / 实验报告
```

## 2. Mini-batch 与可恢复数据顺序

训练样本数为 `N`，batch size 为 `B`，每轮 batch 数是 `ceil(N/B)`；这个值在 batcher 初始化时计算一次。全局 optimizer step 唯一确定：

```text
epoch       = step // batches_per_epoch
batch_index = step % batches_per_epoch
permutation = RNG(seed + epoch).permutation(N)
```

进入一个新 epoch 时生成一次 permutation，并在该 epoch 的所有 batch 之间缓存复用；只有 epoch 改变时才重新生成。缓存只是性能优化，不属于训练状态。因此 checkpoint 不必序列化一个隐含的数据迭代器；恢复的 `step` 足以用 `seed + epoch` 重建当前 epoch 的相同 permutation 和下一批样本。最后一个 batch 可以小于 `B`，但一轮内每个样本恰好出现一次。

## 3. SGD

SGD 直接用当前 batch 梯度更新：

```text
theta_t = theta_(t-1) - learning_rate * gradient_t
```

它的训练状态只有 `step_count`。mini-batch 梯度带有噪声，所以单个 batch loss 不保证单调下降。

## 4. Adam

对每个参数分别维护一阶矩和二阶矩：

```text
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
m_hat = m_t / (1-beta1^t)
v_hat = v_t / (1-beta2^t)
theta_t = theta_(t-1) - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

- `m`：梯度方向的指数移动平均；
- `v`：平方梯度的指数移动平均；
- `t`：更新次数，参与初期偏差修正；
- `epsilon`：避免分母为零。

只恢复模型参数而不恢复 `m`、`v`、`t`，下一步更新就会不同，因此不能称为连续恢复。

## 5. Early stopping

每 25 个 optimizer step 计算一次完整验证 loss。验证 loss 创新低时复制最佳参数并清零耐心计数；连续 16 次没有改善就停止。训练过程中不读取测试数据。完成模型选择后，最佳参数才用于一次最终测试评估。

`best_step` 指最佳验证参数所在的更新次数，`completed_step` 指实际停止时的更新次数，两者不同是正常现象。

## 6. Checkpoint 恢复契约

checkpoint 同时保存：

- 模型配置与当前参数；
- optimizer 类型、`step_count`，以及 Adam 的 `m` 和 `v`；
- 全局 step、最佳验证 loss、最佳 step、耐心计数和最佳参数；
- 完整训练配置、历史观测和累计耗时；
- checkpoint 格式版本。

恢复时拒绝不同的模型配置、训练配置或 optimizer。自动化测试要求连续训练 12 步与第 5 步中断后恢复到 12 步的参数、最佳参数和 Adam 状态逐元素相等。

## 7. 产物分离

| 产物 | 面向对象 | 包含内容 | 不包含内容 |
| --- | --- | --- | --- |
| training checkpoint | 训练恢复 | 当前/最佳参数、optimizer、trainer 状态 | 文本报告 |
| inference model | 推理 | 最佳模型配置和参数 | optimizer、训练进度 |
| experiment report | 人和实验比较 | 配置、loss、perplexity、梯度范数、耗时 | 可执行参数状态 |

## 8. 能力边界

本阶段仍是 NumPy、CPU、单进程、字符级短语料实验。耗时只用于当前环境内观察；不代表 GPU、分布式训练或真实大模型性能。一次随机种子和一个语料切分也不足以证明某个 optimizer 普遍更好。
