# 09 运行报告

2026-09-18 在当前工作树运行 `python3 -m projects.09_subword_tokenizer.compare`。语料为 `data/declaration_excerpt.txt`，沿用 07/08 的顺序 80/10/10 原文切分；读取时移除文件末尾换行。两种 tokenizer 仅在训练文本上建词表，BPE 默认 `max_merges=40`、`min_pair_frequency=2`，实际学习 40 条 merge。统计时逐行编码，不添加 BOS/EOS，保留每行原有换行。

| 指标 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| 原文字符数 | 521 | 65 | 66 |
| 行数 | 1 | 1 | 1 |
| 字符 tokenizer 的 token 数 | 521 | 65 | 66 |
| BPE tokenizer 的 token 数 | 299 | 52 | 52 |
| 每行平均 token 数：字符 / BPE | 521 / 299 | 65 / 52 | 66 / 52 |
| token 数减少 | 42.6% | 20.0% | 21.2% |

字符词表大小为 40，BPE 最终词表大小为 80，其中新增 40 个合并 token。由于每个切分目前只有一行，“每行平均”在此恰好等于总 token 数，不代表多行场景的平均表现。

训练、验证、测试文本中，训练词表未见字符的出现次数分别为 0、1、2；默认编码将它们映射为 `<UNK>`，所以验证与测试文本不能保证逐字符无损往返。另用重复的中英文、标点和代码样例检查：已知的 `def f(x):`、`中文!`、`return x\n` 能往返，未见的 `未` 变为 `<UNK>`。未知字符也可通过 `on_unknown="error"` 明确拒绝。

本报告只比较固定小语料上的 tokenizer 词表和序列长度，不比较模型 loss、perplexity 或生成质量；token 数减少不能单独证明语言模型性能提升，也不能外推到其他语料。当前实现不提供 byte fallback 或生产级 Unicode 处理。测试范围见 [测试报告](test_report.md)。
