# 06 真实语料运行报告

## 固定条件

- 语料：`data/declaration_excerpt.txt`；
- 与阶段 05 相同的顺序 80/10/10 train/valid/test 切分；
- 字符 Tokenizer 只由训练文本建立；
- 模型：context 2、embedding 8、hidden 32；
- batch size 32，seed 7，每 25 step 验证，patience 16 次验证；
- SGD learning rate 0.2；Adam learning rate 0.002；
- 上限 1500 step，以最低验证 loss 的参数做最终报告。

两个 optimizer 的 learning rate 不相同，因为更新尺度和状态定义不同。其他列出的条件保持一致。

## 结果

| optimizer | best step | stopped step | train perplexity | valid perplexity | test perplexity | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGD | 400 | 800 | 8.331658 | 16.019447 | 17.914837 | 0.308 s |
| Adam | 375 | 775 | 8.029470 | 18.150093 | 17.246045 | 0.274 s |

两次运行都由 early stopping 终止并恢复最佳验证参数。当前切分上，SGD 的验证困惑度更低，而 Adam 的测试困惑度更低；这不支持“某个 optimizer 普遍更好”的结论。配置选择应只依据训练集和验证集，不能根据测试集排序反向选择 optimizer。

当前测试切分属于仓库内可见的教学数据，并非外部封存基准。因此测试指标用于检查完整评估流程和描述当前固定实验，不应解释为无偏泛化估计。

耗时来自当前机器的一次短运行，只用于确认记录路径有效，不作性能结论。当前实现每个 epoch 只生成并缓存一次 permutation；这不改变最佳 step、loss 或 perplexity。原始配置、逐次验证 loss、batch loss 和梯度范数见 `artifacts/optimizer_comparison.json` 及两个 optimizer 的独立 JSON 报告。
