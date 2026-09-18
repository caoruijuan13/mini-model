# 08 PyTorch 字符级 Transformer 运行报告

## 固定配置与数据

沿用 07 的 `data/declaration_excerpt.txt`、顺序 80/10/10 切分及仅训练文本建词表。词表 40 个 token；训练/验证/测试窗口分别为 515/59/60。block size 8、维度 16、4 个 attention head、2 个 block、FFN 维度 32，共 5800 个参数。Adam 学习率 0.002，batch size 16，seed 7，最多 1000 次更新；每 25 次验证，连续 12 次无改善则早停。

运行：`python3 -m projects.stage08_pytorch_transformer.train`。逐次评估记录、推理模型和词表嵌入文件位于被忽略的 `artifacts/` 目录，可重新生成。

## 当前结果

| 阶段 | 训练 loss / perplexity | 验证 loss / perplexity | 测试 loss / perplexity |
| --- | ---: | ---: | ---: |
| 08 PyTorch，最佳验证参数 | 2.285756 / 9.833116 | 2.826084 / 16.879234 | 2.861811 / 17.493177 |
| 07 NumPy，既有报告 | 2.159220 / 8.664375 | 2.849753 / 17.283519 | 3.065351 / 21.441981 |

08 的初始训练/验证 loss 为 3.862257 / 3.848142；第 200 次更新取得最低验证 loss，第 500 次更新后早停，最终指标和保存文件均使用第 200 步权重。运行耗时约 3.43 秒，只是本机一次观察，不作为速度基准。

生成设置：BOS 起始、seed 7、top-k 5、最多 80 个新 token。本次样例：

```text
ore it ther ir ind tonalen intont o e antheres the thareres ir tonens, ina isthe
```

## 解读限制

两个阶段使用相同文本切分、目标窗口、架构尺寸和参数数量，因而可对照工作流和数量级；但 PyTorch 与 NumPy 的初始化、浮点精度和 Adam 实现不完全相同。仅一份很短的语料、一个 seed、一次运行；测试文本也不是外部封存基准。上述单次指标差异不能证明框架优劣或真实文本质量，生成样例仅证明推理路径可用。重叠窗口会重复计入某些原文字符，perplexity 亦不宜与其他样本定义的阶段直接排名。
