# 08 设计边界

## 数据和模型

- 直接复用 07 的 `prepare_corpus()`：原文顺序 80/10/10 切分，仅训练文本决定词表，各切分独立建立 BOS/EOS 错位窗口；输入与目标均为 `(B, T)` 的整数 ID。
- token/position embedding 相加，经过 2 个 causal Pre-LN Transformer block、最终 LayerNorm 和词表投影，输出 `(B, T, V)` logits。
- 默认 40 个 token、block size 8、模型维度 16、4 个 head、FFN 维度 32，与 07 同为 5800 个可训练参数。attention 投影无 bias，以匹配 07 结构。
- `nn.functional.cross_entropy` 对全部 `B*T` 个目标位置求平均；PyTorch 自动求导替代 07 的手写 backward。

## 训练状态与数据边界

每个 step 从训练集取一个确定性 mini-batch，执行 `zero_grad → loss → backward → Adam.step`。验证集仅在固定间隔或最后一步计算 loss，不参与参数更新。最低验证 loss 的 `state_dict` 用深拷贝保存，早停后恢复。`best_step=0` 表示初始模型从未被超越；`completed_step` 是实际完成的参数更新次数。测试集不传入 `train_model()`，只在参数选择后评估一次。

固定 seed 控制 PyTorch 初始化和数据排列；生成另用局部随机数生成器。相同机器与当前实现下可复现，不承诺跨设备或不同 PyTorch 版本逐位一致。

## 推理文件

`.pt` 文件保存纯配置字典、`state_dict`，以及训练时的 token 顺序和特殊 token 定义。`load_with_tokenizer()` 同时恢复模型与词表；传入外部 tokenizer 时必须完全匹配。此文件不含 Adam 动量或 step 等训练状态，不是断点续训 checkpoint。

## 与 07 的对照边界

固定参数映射测试证明小输入的 logits 数值对齐；数据切分、目标定义、架构尺寸和可训练参数数量相同。但初始化随机数、浮点精度（07 为 NumPy float64，08 默认 PyTorch float32）及优化器实现不完全相同。单次运行 loss 的差别不构成框架优劣结论。
