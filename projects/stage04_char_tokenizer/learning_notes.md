# 04 字符 Tokenizer 学习资料

## 推荐顺序

1. 阅读 [design.md](design.md)，区分 token、vocabulary、token ID 和 embedding；
2. 阅读共享实现 [char_tokenizer.py](../tokenization/char_tokenizer.py)；
3. 阅读 [model.py](model.py)，观察模型如何只使用 token ID 训练和采样；
4. 运行 [demo.py](demo.py)，观察训练 vocabulary、特殊 token、编码和生成结果；
5. 阅读 [test_report.md](test_report.md) 和测试断言；
6. 对照项目 02、03 中的 `vocab`、`to_id` 和 `ids`，理解抽取前后的职责变化；
7. 阅读 [run_report.md](run_report.md)，观察训练集之外字符如何进入 `<UNK>`。

## 核心关系

```text
Corpus：原始文本数据
Token：Tokenizer 产生的离散文本单位
Vocabulary：所有合法 token 的集合
Token ID：token 在 vocabulary 中的编号
Embedding：阶段 05 根据 token ID 查到的可学习向量
```

阶段 04 的 Tokenizer 输出 token ID；配套 `TokenBigram` 只用于验证模型边界。它仍不学习 embedding，神经网络参数从阶段 05 开始。
