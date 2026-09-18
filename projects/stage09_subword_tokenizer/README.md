# 09：手写最小 BPE Tokenizer

本阶段已完成字符初始化的最小 BPE：只用训练文本学习有序 merge 规则，再用冻结的规则编码验证、测试和新文本。不训练语言模型，也不比较不同 tokenizer 的模型 loss。算法由本项目实现；PyTorch 仅用于把 tokenizer 状态保存为二进制文件。

## 核心流程

1. 从训练文本建立基础字符词表，前置 `<UNK>`、`<BOS>`、`<EOS>`；字符排序与共享的 `CharTokenizer` 一致。
2. 统计当前 ID 序列中的相邻 pair；频数最高者优先，并列时选 ID 字典序最小者。
3. 从左到右做非重叠合并，记录规则并给新 token 分配下一个词表 ID；达到 `max_merges` 或低于 `min_pair_frequency` 时停止。
4. 编码时先逐字符映射到基础 ID，再按保存的 merge 顺序重放规则，不在新文本上重新学习。

例如训练 `abab`、设置 `max_merges=2, min_pair_frequency=1`，规则是 `a+b→ab`、`ab+ab→abab`；编码 `ababab` 得到 `[abab, ab]`。若最低频数保持默认的 2，第二条 pair 只出现一次，训练会停在 `ab`。

未知字符默认映射为 `<UNK>`，此时原字符不能无损恢复；`on_unknown="error"` 会拒绝未知字符。已知字符与标点在不额外插入特殊 token 的范围内可往返。保存格式是带格式名和版本号的 PyTorch 二进制文件，建议使用 `.pt` 后缀，只加载可信文件；它不是 JSON。

## 运行与证据

```bash
python3 -m pytest -q projects/stage09_subword_tokenizer/tests/test_stage09.py
python3 -m projects.stage09_subword_tokenizer.compare
```

当前阶段测试 `26 passed`；默认配置对照中，训练/验证/测试的字符 token 数分别为 `521/65/66`，BPE token 数为 `299/52/52`。具体配置、未知字符边界和结论限制见 [运行报告](run_report.md)；验收覆盖见 [测试报告](test_report.md)。[设计](design.md) 说明数据与算法契约，[学习笔记](learning_notes.md) 解释训练与编码的区别。保存产物可放入被忽略的 `artifacts/` 目录；运行对照命令不会自动保存 tokenizer。
