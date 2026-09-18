# 04 字符 Tokenizer 运行报告

## 1. 运行方式

```bash
python3 projects/stage04_char_tokenizer/demo.py
```

演示从 [公共语料](../../data/declaration_excerpt.txt) 读取文本，按照 `80%/10%/10%` 顺序切分，只使用训练文本建立 vocabulary，然后分别编码训练、验证和测试文本。

## 2. 当前配置

| 项目 | 数值 |
| --- | ---: |
| 原始字符数 | 652 |
| 训练字符数 | 521 |
| 验证字符数 | 65 |
| 测试字符数 | 66 |
| 训练字符种类 | 37 |
| 特殊 token 数 | 3 |
| vocabulary size | 40 |

编码每个数据集时加入一个 `<BOS>` 和一个 `<EOS>`，因此 token 数比原始字符数多 2。

## 3. 当前输出

```text
vocab_size = 40
train_tokens = 523
valid_tokens = 67
test_tokens = 68
valid_unknown_tokens = 1
test_unknown_tokens = 2
train_bigram_examples = 522
encoded_sample = We hold
encoded_sample_ids = [1, 16, 21, 3, 24, 30, 27, 20, 2]
round_trip_ok = True
generated_tokens = 100
generated_text = Were t tome anghtopuigh Gomopond at, Cr athser ioven d Thewenme f e uitese tsthalthathecof theat d
```

验证集中有 1 个训练 vocabulary 未见字符 `z`，测试集中有 2 个未见字符 `S` 和 `k`。它们编码为 `<UNK>`，证明 vocabulary 没有使用验证集或测试集信息。

`TokenBigram` 使用 `vocab_size=40` 训练，从 `<BOS>` 的 ID 开始采样，最多生成 100 个 token。模型返回 ID 序列，演示代码再调用 Tokenizer 解码。生成文本仍只有一个 token 的上下文，不作为语言质量证据。

## 4. 产物

生成的 [char_tokenizer.json](artifacts/char_tokenizer.json) 保存格式版本、40 个 token 及特殊 token 配置。模型后续应加载这个产物，而不是根据推理输入重建 vocabulary。

## 5. 结果边界

- `round_trip_ok=True` 只适用于 vocabulary 已知的样例；
- 未知字符转换为 `<UNK>` 后不能恢复原字符；
- bigram 示例和采样只证明 token ID 可以作为模型输入输出；
- 当前计数 Bigram 没有 embedding、梯度或神经网络参数；
- 本阶段没有把生成样例作为模型质量指标。
