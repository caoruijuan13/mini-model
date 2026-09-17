# 08 学习笔记

从 07 到 08，首先保持输入、目标、架构和损失定义不变，只改变“梯度由谁算”。`forward()` 仍由开发者实现；PyTorch 把计算记录成可求导的图，`loss.backward()` 将梯度写到参数的 `.grad`，Adam 在 `optimizer.step()` 读取并更新参数。

建议按以下顺序复习：

1. `nn.Module` 注册子模块与参数：`nn.ModuleList` 中的 block 会进入 `model.parameters()` 和 `state_dict()`；
2. `(B, T, D)` 的 Q/K/V 分 head、因果 mask 与最后一个位置的 logits；
3. `nn.functional.cross_entropy` 接收 `(B*T, V)` logits 与 `(B*T,)` 目标 ID；输入不需先 softmax；
4. 一次 mini-batch 更新：`zero_grad → forward/loss → backward → step`；`model.train()` 只是模式切换，不替代后三步；
5. `model.eval()` 与 `torch.no_grad()` 分工不同：前者切换模块行为，后者停止构建求导图；
6. 验证选择必须深拷贝 `state_dict()`，而非仅保存其引用；最佳模型可能是 step 0；
7. 推理文件应绑定准确的词表顺序，`vocab_size` 相同并不保证 token ID 语义相同。

固定参数映射的 07/08 前向一致性测试证明两套实现计算的是同一结构；默认初始化和优化器数值仍可能不同。阶段 08 没有实现断点续训；推理模型文件不能代替包含 optimizer 状态与 step 的训练 checkpoint。
