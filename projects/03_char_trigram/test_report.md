# 03 字符 trigram 测试报告

## 1. 测试范围

测试文件为 [tests/test_char_trigram.py](tests/test_char_trigram.py)。项目 02 的 bigram 测试保持独立，不被本项目替代。

## 2. 执行方式

```bash
PYTHONPATH=projects/03_char_trigram python3 -m pytest -q projects/03_char_trigram/tests
```

## 3. 测试内容

### 3.1 概率与真实语料评估

验证三维概率矩阵沿下一字符维度求和为 1，并检查真实语料验证 loss 和测试 perplexity 为有限值。

### 3.2 新采样策略

验证 `sample_new()` 使用相同 seed 时结果一致，并准确返回指定长度。

### 3.3 保存和加载

验证词表、trigram 概率、bigram backoff 概率、上下文可见状态及加载后的采样结果保持一致。

## 4. 当前结果

```text
3 passed
```

项目 03 共通过 3 个测试。

## 5. 测试边界

这些测试验证算法结构和运行行为，不证明生成文本具有人类级语言连贯性，也不证明当前短语料足以训练高质量 trigram。
