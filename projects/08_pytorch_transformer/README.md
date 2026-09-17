# 08：PyTorch 字符级 Transformer

本阶段**进行中，还没有可作为最终证据的训练实验指标**。数据切分、训练集建词表、BOS/EOS 和输入/目标错位窗口直接复用 07；模型结构和训练目标保持一致，只替换实现方式：你写 `nn.Module` 的前向计算，PyTorch 自动求梯度。

建议按 `model.py` 的 TODO 1–8，再到 `train.py` 的 TODO 9 完成。先让 attention 的因果性与 block 的形状测试通过，再实现整个模型，最后训练和保存/加载。`forward` 返回 logits `(B, T, V)`，不要提前做 softmax；训练时把 logits 摊成 `(B*T, V)`，targets 摊成 `(B*T,)`，交给框架交叉熵。

`nn.ModuleList`、`nn.Embedding`、`nn.Linear`、`nn.LayerNorm` 应注册为模型的子模块，才能进入 `parameters()` 和 `state_dict()`。训练的一步是 `zero_grad → forward → loss → backward → step`；这里不再手算注意力或 LayerNorm 的反向公式。

验收测试在 `tests/test_stage08.py`。当前模型前向、保存加载、训练 loss 下降和报告入口的测试已通过，但训练循环仍需修正验证集选参、早停和最佳权重恢复，不代表 08 已完成。08 报告暂不生成文本；07 的 NumPy 生成器不能直接用于 PyTorch 模型。

```bash
python3 -m pytest -q projects/08_pytorch_transformer/tests/test_stage08.py
```

对照 07 时请固定同一数据边界和可比的模型配置；随机初始化与优化器细节不同，单次 loss 差异不能证明某个框架更好。模型权重文件必须配套同一 tokenizer 词表和特殊 token ID，不能只凭 `vocab_size` 相等就认为兼容。
