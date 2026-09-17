# 07 字符级 Transformer 设计契约

## 目标与边界

使用字符 token ID 序列，按当前位置预测下一个 token。该阶段的新增内容是模型结构；数据切分、训练集建词表与“验证集选择、测试集最终评估”的规则沿用此前阶段。固定配置的实际运行指标见运行报告，不以单次样例推断生成质量。

## 输入与目标

`build_sequence_windows()` 对每个数据切分独立插入 BOS/EOS，构造固定长度、错开一位的 `inputs` 和 `targets`，形状均为 `(examples, T)`。窗口可重叠；同一目标可能在多个窗口中出现。短于 `block_size` 的切分会明确报错，不会静默丢弃 EOS。

例如 token ID 为 `[3, 4, 5]`，`block_size=2`：

```text
inputs        targets
[BOS, 3]  ->  [3, 4]
[3, 4]    ->  [4, 5]
[4, 5]    ->  [5, EOS]
```

## 模型张量契约

配置默认值仅用于教学：`block_size=8`、`model_dim=16`、`num_heads=4`、`num_layers=2`、`feed_forward_dim=32`。`model_dim` 必须能被 `num_heads` 整除。

| 环节 | 形状 |
| --- | --- |
| 输入 token IDs | `(B, T)`，`1 <= T <= block_size` |
| token embedding + position embedding | `(B, T, D)` |
| Q、K、V（拆头后） | `(B, H, T, D/H)` |
| attention 分数与权重 | `(B, H, T, T)` |
| 多头拼接、输出投影、每个 block | `(B, T, D)` |
| 最终 logits | `(B, T, V)` |

每个 query 位置 `i` 只能读取 `j <= i`。mask 应在 softmax 前作用，使未来位置权重精确为零。注意 `Q_i·K_i` 是允许的对角线项。每个 head 使用同样的因果约束，但有自己的可学习投影。

采用 Pre-LN block：

```python
x = x + attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

FFN 按位置共享参数，不跨位置读取信息。模型最终对 `B*T` 个目标位置计算平均交叉熵。生成时只取最后位置的 logits，逐次采样并滚动保留不超过 `block_size` 的上下文。

## 训练与验证边界

`loss_and_gradients()` 经有限差分检查。训练循环复用阶段 06 的确定性 mini-batch 与 SGD/Adam 实现，但单独组织 Transformer 的模型参数、验证选择和最终评估。每次只用训练集更新参数、用验证集保存最优权重；训练完成后恢复最优权重，再读取测试集计算一次最终指标。推理模型保存配置和权重，不保存优化器与训练进度；阶段 06 的 checkpoint 不能直接用于此模型。

当前验收包括：形状、数值稳定性、causal 不泄漏、数值梯度、小数据过拟合、保存加载一致性、生成停止，以及固定配置下的训练/验证/测试报告。测试集不能用于选择结构或超参数。
