# 02：字符级 bigram 语言模型

这是阶段二，暂不和阶段一同时学习。

## 问题

输入当前字符，输出下一个字符的概率分布。

```text
输入当前字符 → bigram → P(下一个字符 | 当前字符) → 逐字符生成文本
```

## 学习目标

本阶段学习词表、条件概率、计数平滑、交叉熵、困惑度和采样。

## 语料对照

| 语料 | 用途 | 特点 |
|---|---|---|
| `DEFAULT_CORPUS` | 原有教学基线 | 人工编写的短文本重复 8 次，用于复现历史结果 |
| `REAL_CORPUS` | 真实文本对照 | 美国《独立宣言》公共领域节选，保留自然大小写和标点，不重复扩充 |

默认训练入口仍使用 `DEFAULT_CORPUS`，所以原有指标和模型产物不变。`REAL_CORPUS` 由新增测试和运行报告单独评估。

## 学习资料

- [design.md](design.md)：模型设计、训练逻辑、生成流程和边界
- [test_report.md](test_report.md)：测试覆盖和测试结论
- [run_report.md](run_report.md)：一次训练评估运行结果
- [learning_notes.md](learning_notes.md)：文档导航和学习顺序

## 运行

先训练、评估并保存模型：

```bash
PYTHONPATH=projects/02_char_bigram python3 projects/02_char_bigram/train.py
```

再加载已保存的模型生成文本：

```bash
PYTHONPATH=projects/02_char_bigram python3 projects/02_char_bigram/generate.py
```

运行测试：

```bash
PYTHONPATH=projects/02_char_bigram python3 -m pytest -q projects/02_char_bigram/tests
```

训练产物保存在 `artifacts/char_bigram.npz`。生成阶段只读取这个文件，不会重新训练。

实现文件：

- `data.py`：字符语料和顺序切分
- `model.py`：字符 bigram
- `train.py`：训练、评估和保存模型
- `generate.py`：加载模型并采样文本

本目录是一个独立的后续学习项目。阶段一完成后，再正式进入本目录学习。

采样对比：`sample()` 保留原始的全量概率采样逻辑；`sample_new()` 使用 `temperature=0.8` 和 `top_k=5`，先突出高概率字符，再过滤低概率候选。两者共享同一模型概率表，区别只发生在推理阶段。
