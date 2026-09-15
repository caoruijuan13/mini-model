# 05 Token Embedding 与 MLP 测试报告

## 当前状态

阶段 05 的四组测试均已启用。当前重新运行阶段 05 与全项目测试，结果分别为 `11 passed` 和 `32 passed`。

## 测试分组

| 文件 | 验证内容 | 当前状态 |
| --- | --- | --- |
| `test_data.py` | BOS 左填充、EOS 目标、任意 context size、空序列、形状和输入不变性 | 3 个测试通过 |
| `test_forward.py` | 参数形状、固定 seed、logits、稳定 softmax | 3 个测试通过 |
| `test_gradients.py` | 数值梯度、完整梯度键、loss 下降 | 2 个测试通过 |
| `test_generation.py` | 脚本化滚动上下文、固定 seed、EOS、保存加载 | 3 个测试通过 |

## 运行方式

```bash
python3 -m pytest -q projects/05_token_mlp/tests/test_data.py
python3 -m pytest -q projects/05_token_mlp/tests/test_forward.py
python3 -m pytest -q projects/05_token_mlp/tests/test_gradients.py
python3 -m pytest -q projects/05_token_mlp/tests/test_generation.py
```

## 验证边界

这些测试证明当前实现的形状、梯度、更新方向、生成控制和持久化行为符合阶段 05 的规格，但不证明短语料能够训练出高质量自然语言模型。固定 seed 的生成样例只用于验证推理路径，不作为模型优于 trigram 的证据。
