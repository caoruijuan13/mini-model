# 08 测试报告

当前全量测试：`87 passed`；其中阶段 08 为 `17 passed`。运行命令：

```bash
python3 -m pytest -q
python3 -m pytest -q projects/stage08_pytorch_transformer/tests/test_stage08.py
```

覆盖的核心边界：

- 复用 07 的数据切分、词表与错位窗口；默认模型与 07 同为 5800 个参数；
- 固定参数映射时，完整窗口和短提示的 logits 与 07 数值对齐；attention 不读取未来位置；
- 可训练参数均获得有限梯度，小数据训练降低验证 loss；
- 从未改善时恢复初始模型，改善时恢复最佳验证模型；最后一步即使不落在评估间隔上也被评估；相同 seed 的短训练可复现；
- 保存、加载后 logits 一致；推理文件能还原词表并拒绝同大小但不同映射的 tokenizer；
- 固定 seed 生成、上下文滚动、EOS 停止，以及报告的保存路径。

这些测试证明阶段 08 的教学契约，不覆盖跨设备确定性、外部语料泛化或大模型性能。
