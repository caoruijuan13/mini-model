# 06 自动化测试报告

## 当前结果

```text
43 passed
```

其中阶段 06 新增 11 个测试，覆盖：

- 一轮 mini-batch 恰好覆盖全部样本；
- 由 seed 和全局 step 重建相同 batch；
- 同一 epoch 只生成一次 permutation，跨 epoch 才重新生成；
- SGD 更新值与 step 计数；
- Adam 第一步偏差修正、`m`、`v` 和状态往返；
- 第 5 步中断恢复与连续训练到第 12 步逐元素一致；
- checkpoint 同时恢复模型、optimizer 和 trainer 状态；
- 相同配置重复训练的历史与参数一致；
- checkpoint、推理模型和 JSON 报告彼此分离。

## 未验证内容

- 多进程、GPU 或跨机器的 bitwise reproducibility；
- checkpoint 写入中断时的原子性；
- 超大模型下的内存和序列化性能；
- 多个随机种子上的统计显著性；
- NumPy 之外框架的 optimizer 数值一致性。
