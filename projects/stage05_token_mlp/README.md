# 05：Token Embedding 与 MLP

这是第五个学习阶段。该阶段已经完成：使用阶段 04 的 `CharTokenizer` 产生 token ID，手写带 embedding 和非线性隐藏层的 NumPy MLP，根据前两个 token 预测下一个 token。

```text
context token IDs
  → embedding lookup
  → concatenate
  → linear + tanh
  → linear
  → logits
  → softmax
  → next-token probabilities
```

## 实现与验收顺序

本阶段按以下顺序完成实现和验收：

1. `test_data.py`：上下文和目标构造；
2. `test_forward.py`：参数初始化、embedding、前向传播和 softmax；
3. `test_gradients.py`：交叉熵、反向传播和梯度下降；
4. `test_generation.py`：自回归生成、temperature、top-k 和 EOS。

四组测试均已启用并通过，阶段 05 当前共有 11 个自动化测试。

## 固定基线

```text
context_size  = 2
embedding_dim = 8
hidden_dim    = 32
activation    = tanh
```

保持两个 token 的上下文，是为了与阶段 03 的 trigram 任务对齐。阶段 05 不再单独实现无隐藏层的线性模型。

## 文件结构

- `data.py`：语料读取、切分以及 context/target 构造；
- `model.py`：Embedding、MLP、反向传播、参数更新和持久化；
- `train.py`：全批量梯度下降、验证集模型选择和最终评估；
- `generate.py`：temperature、top-k 和 EOS 控制的 token ID 自回归生成；
- `design.md`：公式、张量形状和职责边界；
- `learning_notes.md`：推荐实现顺序和检查问题；
- `test_report.md`：分阶段测试规格；
- `run_report.md`：固定配置的实际训练结果、对照和限制；
- `tests/`：阶段 05 的可执行验收规格。

共享依赖：

- [CharTokenizer](../tokenization/char_tokenizer.py)；
- [公共语料](../../data/declaration_excerpt.txt)。

## 运行测试

运行全项目测试：

```bash
python3 -m pytest -q
```

单独运行阶段 05：

```bash
python3 -m pytest -q projects/stage05_token_mlp/tests
```

当前结果：

```text
11 passed
```

运行固定配置训练：

```bash
PYTHONPATH=projects/stage05_token_mlp python3 projects/stage05_token_mlp/train.py
```

## 阶段边界

本阶段使用 NumPy 手写全批量梯度下降。mini-batch、Adam、checkpoint 恢复和学习率调度属于阶段 06；attention 和 Transformer 属于阶段 07。
