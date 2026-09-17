# 07 字符级 Transformer 运行报告

## 固定配置

- 语料：`data/declaration_excerpt.txt`，沿原文顺序做 80/10/10 切分；
- 只从训练文本建立字符词表，包含 BOS/EOS/UNK，共 40 个 token；
- block size 8，model dimension 16，4 个 attention head，2 个 Transformer block，FFN hidden dimension 32；总计 5800 个参数；
- Adam，learning rate 0.002，batch size 16，seed 7，最多 1000 step；
- 每 25 step 在验证集评估，连续 12 次未改善则提前停止，恢复最低验证 loss 的参数；
- 每个切分独立构造重叠的错位窗口，训练/验证/测试窗口数为 515/59/60。

运行入口：`python3 -m projects.07_char_transformer.train`。推理模型与逐次验证记录保存在被忽略的 `artifacts/` 目录，可由同一命令重新生成。

## 当前结果

| 切分 | 最佳参数的 loss | perplexity |
| --- | ---: | ---: |
| 训练 | 2.159220 | 8.664375 |
| 验证 | 2.849753 | 17.283519 |
| 测试 | 3.065351 | 21.441981 |

初始训练/验证 loss 分别为 4.145907 / 3.967697。最佳验证参数在第 225 次更新取得；训练到第 525 次更新时触发 early stopping，最终报告与保存的推理模型均使用第 225 步参数。

| step | 训练 loss | 验证 loss | 说明 |
| ---: | ---: | ---: | --- |
| 0 | 4.145907 | 3.967697 | 随机初始化 |
| 225 | 2.159220 | 2.849753 | 验证最优，选为最终参数 |
| 525 | 1.539294 | 3.318610 | 训练继续下降，验证变差；不作为最终模型 |

本次训练耗时约 1.90 秒，仅记录当前机器的一次短运行，不作为性能基准。训练曲线的全部评估点、batch loss 和梯度范数在 `artifacts/experiment_report.json` 中。

## 固定 seed 生成样例

使用最佳参数，从 BOS 开始，`seed=7`、`top_k=5`、temperature 1.0、最多 80 个新 token：

```text
ese at tore aporrirt at aresend aat e theshenn otode om ththedhaaandrer thatsut 
```

这只证明自回归生成路径可以运行，不能单凭样例判断文本质量或宣称模型优于先前阶段。

## 证据边界

- 交叉熵对每个重叠窗口的所有目标位置取平均；相同原文字符可能在多个窗口中重复计入，因此与阶段 05/06 的样本定义不同，perplexity 不宜直接排名；
- 验证集用于选择最佳 step，测试集仅在选定参数后用于本次报告；但该测试文本属于仓库内可见教学数据，并非外部封存基准；
- 只有一个随机 seed、一份极短语料，测试数值是探索性教学记录，不能外推真实语料泛化能力；
- 当前模型文件只含推理配置和权重，不含词表映射、优化器状态或可恢复训练的 checkpoint。独立推理需要匹配的词表。
