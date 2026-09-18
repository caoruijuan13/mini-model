# 06 训练工程基础学习资料

## 推荐顺序

1. 先区分模型参数、optimizer 状态和 trainer 状态；
2. 阅读 `DeterministicBatcher.batch_for_step()`，手算 step 如何映射到 epoch 和 batch，并确认每个 epoch 只生成一次 permutation；
3. 用一个标量参数手算一次 SGD；
4. 用同一个梯度手算 Adam 的 `m`、`v`、偏差修正和参数更新；
5. 运行 checkpoint 恢复等价性测试；
6. 阅读 early stopping，确认测试集没有进入训练循环；
7. 运行 SGD/Adam 对照并解释验证集与测试集排序为何可能不同。

## 应能回答的问题

### Mini-batch

- batch size 变小为什么会增加梯度噪声？
- 为什么最后一个不足 batch size 的 batch 仍应保留？
- 为什么固定模型 seed 还不足以保证训练完全复现？

### Optimizer

- SGD 保存 `step_count` 的意义是什么？
- Adam 的一阶矩和二阶矩分别描述什么？
- 为什么 Adam 初期需要 `1-beta^t` 偏差修正？
- 为什么 Adam 和 SGD 通常不应机械使用相同 learning rate？

### 恢复与验证

- checkpoint 为什么不能只保存模型权重？
- `best_step` 为什么通常小于 `completed_step`？
- early stopping 的 patience 按 step 还是验证次数计算？
- 验证集用于选择配置后，测试集还能承担什么角色？

## 建议练习

1. 把 `batch_size` 改为 1、32 和大于训练集大小，比较每步梯度范数；
2. 暂时去掉 Adam 偏差修正，观察前几步更新幅度；
3. 在第 5 步保存并恢复，再故意不加载 Adam 状态，比较第 6 步参数；
4. 用三个 seed 重复实验，观察单 seed 结论是否稳定。
