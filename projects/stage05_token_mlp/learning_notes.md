# 05 Token Embedding 与 MLP 学习资料

## 推荐实现顺序

1. 阅读 [design.md](design.md)，手工写出所有张量形状；
2. 完成 `build_context_targets()`，启用并通过 `test_data.py`；
3. 完成参数初始化、`forward()` 和 `predict_proba()`，启用并通过 `test_forward.py`；
4. 完成 `loss()`、`loss_and_gradients()` 和 `apply_gradients()`，启用并通过 `test_gradients.py`；
5. 完成保存、加载和 `generate_ids()`，启用并通过 `test_generation.py`；
6. 完成 `train_and_evaluate()`，运行真实语料实验；
7. 填写 [run_report.md](run_report.md) 和最终 [test_report.md](test_report.md)；
8. 所有证据通过后再勾选总路线图阶段 05。

## 每一步应能回答的问题

### 数据

- 为什么第一个样本需要两个 BOS？
- 为什么 target 比原文本多一个 EOS？
- 为什么不同数据集必须独立构造窗口？

### 前向传播

- token ID 为什么不能直接作为连续特征？
- embedding 查表前后形状如何变化？
- 为什么没有 `tanh` 时两层线性变换可以合并？
- softmax 应该沿哪个维度归一化？

### 反向传播

- 为什么 logits 梯度需要除以 batch size？
- `1-hidden**2` 来自哪里？
- 为什么 embedding 梯度要用累加而不是赋值？

### 生成

- 为什么每一步使用最近两个 token ID？
- temperature 改变什么？
- top-k 在模型计算前还是采样前应用？
- EOS 为什么能够让生成提前停止？
