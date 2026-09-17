# 07 字符级 Transformer

已完成数据、模型、手写反向传播、训练、验证选择、推理模型保存加载与逐 token 生成的教学闭环。关键参数梯度经过有限差分检查，并有小数据过拟合与真实语料运行报告。结果只属于当前短语料和固定配置。

```bash
python3 -m projects.07_char_transformer.train
python3 -m pytest -q projects/07_char_transformer/tests
```

## 文件导航

| 文件 | 当前状态 | 职责 |
| --- | --- | --- |
| `data.py` | 已实现 | 原语料顺序切分、训练集建词表、BOS/EOS 与错位序列窗口 |
| `model.py` | 前向、loss、梯度和保存加载已实现 | token/position embedding、多层 block、logits、loss、梯度、保存加载 |
| `attention.py` | 已实现 | causal mask、scaled dot-product、multi-head attention |
| `layers.py` | 已实现 | LayerNorm、逐位置 FFN、Pre-LN 残差 block 及反向传播 |
| `generate.py` | 已实现 | 自回归逐 token 生成、temperature、top-k、EOS 停止 |
| `train.py` | 已实现 | mini-batch、Adam/SGD、验证选择、测试评估与报告 |
| `tests/` | 27 项测试通过 | 数据、模型、梯度、过拟合、保存加载、生成和训练边界 |

与阶段 05/06 保持相同的语料切分语义，但不修改历史实现。继续复用共享的 `projects.tokenization.CharTokenizer`。模型文件只保存配置和参数，不包含优化器状态、训练进度或词表；当前不保存 `tokenizer.json`，每次从固定的训练文本确定性生成词表。独立推理时必须提供与模型匹配的词表映射。

固定配置和实际指标见 [运行报告](run_report.md)，验证范围见 [测试报告](test_report.md)。本阶段不包含可恢复训练 checkpoint；`model.save()` 只保存推理权重。
