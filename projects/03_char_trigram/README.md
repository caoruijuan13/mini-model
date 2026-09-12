# 03：字符级 trigram 语言模型

这是从 bigram 独立出来的第三个学习项目。它使用真实公共领域语料，并根据前两个字符预测下一个字符：

```text
前两个字符 → trigram → P(下一个字符 | 前两个字符) → 改进采样
```

## 项目特点

- 语料：美国《独立宣言》公共领域节选 `REAL_CORPUS`
- 上下文长度：2 个字符
- 稀疏处理：未见 trigram 上下文时回退到 bigram 分布
- 采样：temperature `0.8` + top-k `5`
- 调参：只用验证集选择 smoothing
- 产物：`artifacts/char_trigram.npz`

## 文档

- [design.md](design.md)：模型、backoff 和采样设计
- [test_report.md](test_report.md)：独立测试说明
- [run_report.md](run_report.md)：实际训练指标和 sample
- [learning_notes.md](learning_notes.md)：学习导航

## 运行

```bash
PYTHONPATH=projects/03_char_trigram python3 projects/03_char_trigram/train.py
PYTHONPATH=projects/03_char_trigram python3 projects/03_char_trigram/generate.py
PYTHONPATH=projects/03_char_trigram python3 -m pytest -q projects/03_char_trigram/tests
```

项目 02 继续保留纯 bigram 实现和历史结果；项目 03 是独立的 trigram 项目，不共享模型产物。
