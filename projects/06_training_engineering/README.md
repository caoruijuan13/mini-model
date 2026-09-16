# 06 训练工程基础

本阶段不改变阶段 05 的 Token MLP 结构，而是把“模型如何计算”与“训练如何运行”分开，加入确定性 mini-batch、SGD/Adam、验证集 early stopping、checkpoint 恢复和训练观测记录。

## 运行

运行固定的 SGD/Adam 对照：

```bash
PYTHONPATH=projects/06_training_engineering python3 projects/06_training_engineering/compare_optimizers.py
```

运行默认 Adam 配置：

```bash
PYTHONPATH=projects/06_training_engineering python3 projects/06_training_engineering/train.py
```

使用完全相同的参数恢复 checkpoint：

```bash
PYTHONPATH=projects/06_training_engineering python3 projects/06_training_engineering/train.py --resume
```

命令行可以配置 `optimizer`、`learning-rate`、`batch-size`、`max-steps`、`eval-interval`、`patience-evaluations` 和 `seed`。恢复时训练配置必须与 checkpoint 一致，防止悄悄改变实验语义。

## 文件导航

- `model.py`：与阶段 05 相同语义的 Token MLP；
- `data.py`：数据切分、上下文样本和可由全局 step 重建的 mini-batch；
- `optimizers.py`：手写 SGD 与 Adam；
- `checkpoint.py`：模型参数、优化器状态和训练器状态的版本化保存；
- `train.py`：early stopping、恢复训练、日志和最终评估；
- `compare_optimizers.py`：固定控制变量的优化器对照；
- `design.md`：状态边界与算法设计；
- `learning_notes.md`：推荐学习顺序和检查问题；
- `test_report.md`、`run_report.md`：自动化与真实语料证据；
- `artifacts/`：checkpoint、推理模型、Tokenizer 和 JSON 实验报告。

训练 checkpoint 用于继续更新参数；推理模型只保存最佳模型参数；实验报告只保存指标和配置。三者不能互相替代。
