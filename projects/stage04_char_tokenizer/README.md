# 04：显式字符 Tokenizer

这是第四个学习阶段。项目 02、03 已经把字符隐式映射为整数；本阶段把这种映射整理成独立、稳定、可保存的 Tokenizer 接口。

```text
文本
  → CharTokenizer.encode()
  → token ID 序列
  → 后续模型

token ID 序列
  → CharTokenizer.decode()
  → 文本
```

## 学习目标

- 区分 token、vocabulary、token ID、Tokenizer 和 embedding；
- 只使用训练文本建立 vocabulary；
- 显式处理 `<UNK>`、`<BOS>` 和 `<EOS>`；
- 保证 encode/decode 和保存/加载行为稳定；
- 明确 Tokenizer、数据集和模型之间的职责边界。
- 用只接收 token ID 的 `TokenBigram` 验证训练和采样边界。

## 代码组织

阶段 04 分成共享实现和学习项目两层：

```text
projects/tokenization/char_tokenizer.py  共享实现，供阶段 04 及后续项目使用
projects/stage04_char_tokenizer/              演示、测试、报告和阶段文档
projects/stage04_char_tokenizer/model.py       只处理 token ID 的 Bigram 对照模型
data/declaration_excerpt.txt             阶段 04 及后续阶段的公共语料文件
```

项目 02、03 保持为独立历史基线，不依赖阶段 04。项目 05 及后续模型应复用 `projects.tokenization.CharTokenizer`，不再各自实现字符到 ID 的映射。

## 运行

运行演示并生成 Tokenizer 产物：

```bash
python3 projects/stage04_char_tokenizer/demo.py
```

运行阶段测试：

```bash
python3 -m pytest -q projects/stage04_char_tokenizer/tests
```

生成产物：

```text
projects/stage04_char_tokenizer/artifacts/char_tokenizer.json
```

该 JSON 文件包含格式版本、完整 vocabulary 和特殊 token 配置，不包含语料、bigram 计数或模型权重。

## 文档

- [design.md](design.md)：接口、词表顺序、未知字符策略和职责边界；
- [learning_notes.md](learning_notes.md)：概念关系和推荐阅读顺序；
- [test_report.md](test_report.md)：测试覆盖及其边界；
- [run_report.md](run_report.md)：当前语料上的实际编码结果。

## 阶段边界

本阶段实现字符级 Tokenizer，并保留一个只处理 token ID 的计数 Bigram 作为集成验证。它不实现 embedding、神经网络或 BPE。Token ID 只是词表索引，本身不是模型学习到的语义表示。
