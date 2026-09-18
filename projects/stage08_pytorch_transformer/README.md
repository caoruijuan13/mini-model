# 08：PyTorch 字符级 Transformer

本阶段沿用 07 的字符词表、顺序切分和错位窗口，改用 PyTorch 的 `nn.Module`、自动求导和 Adam。前向仍由我们定义；`loss.backward()` 计算梯度，`optimizer.step()` 更新参数。学习顺序、验收和边界见 [学习笔记](learning_notes.md)、[设计](design.md)、[测试报告](test_report.md) 与 [运行报告](run_report.md)。

从项目根目录运行：

```bash
python3 -m pytest -q projects/stage08_pytorch_transformer/tests/test_stage08.py
python3 -m projects.stage08_pytorch_transformer.train
```

训练每 25 次参数更新评估一次验证集，连续 12 次未改善时早停。返回和保存的都是最低验证 loss 对应的模型；测试集只在选定参数后评估。`artifacts/char_transformer.pt` 是可重建的推理文件，包含模型配置、权重和训练时的完整字符词表，无需另存 tokenizer JSON：

```python
from projects.stage08_pytorch_transformer.model import TorchCharTransformer

model, tokenizer = TorchCharTransformer.load_with_tokenizer(
    "projects/stage08_pytorch_transformer/artifacts/char_transformer.pt"
)
```

如果自行提供 tokenizer，`TorchCharTransformer.load(path, tokenizer=tokenizer)` 会核对完整 token 顺序与特殊 token，而不仅检查词表大小。模型文件只用于推理；它不包含优化器状态，不能精确恢复中断的训练。

单次短语料指标和生成样例只用于验证工作流，不代表真实语言建模质量，也不能据此判断 PyTorch 比 NumPy 实现更好。
