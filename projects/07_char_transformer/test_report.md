# 07 测试报告

当前运行 `python3 -m pytest -q`：**70 passed**；其中阶段 07 为 **27 passed**，没有跳过的测试。

已覆盖：

- 训练集建词表、顺序切分、BOS/EOS 和输入/目标错位窗口；
- causal mask、attention 权重归一化、未来信息隔离、拆头合头形状与 batch 一致性；
- LayerNorm、FFN、残差 block、模型 logits 和所有位置的交叉熵；
- 两层模型中各参数梯度与有限差分抽样比较，包括重复 token 的 embedding 梯度；
- 推理模型保存加载后参数和 logits 一致，损坏的参数形状或缺失配置会报错；
- 固定 seed 生成、最后位置 logits、滚动上下文、temperature/top-k 参数边界与 EOS 停止；
- 小数据可过拟合到 loss 小于 0.01、验证集最佳参数恢复、固定 seed 训练复现，以及真实语料报告和保存模型的一致性。

这些测试验证的是当前教学实现的计算与数据边界。它们不证明长文本泛化、生成质量、GPU 性能或外部封存语料表现。阶段 07 未实现可恢复训练 checkpoint；推理模型文件不能替代训练状态。
