# 04 字符 Tokenizer 设计报告

## 1. 问题

项目 02、03 在模型内部重复完成了字符词表、`to_id`、未知字符检查和字符串编码。阶段 04 将这些稳定职责抽取为独立组件：

```text
文本 ⇄ CharTokenizer ⇄ token ID 序列
```

Tokenizer 不负责训练模型，只负责定义文本如何转换成模型可接收的离散编号。

## 2. 输入与输出

编码输入是 Python 字符串，输出是整数 ID 列表：

```python
ids = tokenizer.encode("We", add_bos=True, add_eos=True)
```

解码输入是 ID 序列，输出是 token 文本：

```python
text = tokenizer.decode(ids, skip_special_tokens=True)
```

## 3. Vocabulary

字符 vocabulary 只从训练文本建立。特殊 token 固定排列在普通字符之前，普通字符按 Unicode 顺序排列：

```text
0  <UNK>
1  <BOS>
2  <EOS>
3+ 训练文本中出现的字符
```

确定性排序保证相同训练文本和配置得到相同 token ID。内部使用 `tuple[str, ...]` 保存 vocabulary，并以只读映射公开 token ID，而不是使用可变字典或一个字符串；特殊 token 和未来的 subword token 都可能包含多个字符。

## 4. 特殊 token

- `<UNK>`：表示训练 vocabulary 中没有的字符；
- `<BOS>`：表示序列开始；
- `<EOS>`：表示序列结束。

是否在编码结果中加入 BOS/EOS 由调用参数控制。未知字符支持两种显式策略：

```text
use_unk  → 转换为 <UNK> 的 ID
error    → 抛出 ValueError
```

## 5. 保存与加载

Tokenizer 保存为带版本号的 UTF-8 JSON：

```json
{
  "format": "mini-model-char-tokenizer",
  "version": 1,
  "tokens": [],
  "special_tokens": {}
}
```

推理时必须加载训练阶段保存的同一个 Tokenizer，不能根据输入文本重新生成 vocabulary，否则相同 ID 可能代表不同 token。

## 6. 职责边界

| 能力 | 负责组件 |
| --- | --- |
| 文本切成字符 token | Tokenizer |
| token 与 ID 双向转换 | Tokenizer |
| 特殊 token 和未知字符 | Tokenizer |
| train/valid/test 切分 | 数据处理 |
| 构造 context/target 样本 | 数据集 |
| 计算概率或 loss | `TokenBigram` 等模型 |
| temperature 和 top-k | 采样策略 |
| token ID 转换为向量 | 阶段 05 的 embedding |

`make_bigram_examples()` 放在阶段演示中，用于证明样本构造发生在 Tokenizer 之后，但它不是 `CharTokenizer` 的方法。`TokenBigram` 只接收 `vocab_size` 和 token ID，不持有字符词表或 `to_id`。

采样也保持同一边界：

```text
<BOS> → start_id
TokenBigram.sample() → generated_ids
CharTokenizer.decode() → generated_text
```

模型每一步使用上一步生成的 ID 选择概率行。`length` 表示包含起始 ID 的最大 token 数；如果提供 `eos_id`，模型采样到 EOS 后提前停止。

## 7. 与项目 02、03 的关系

项目 02、03 保持冻结，用于展示隐式字符编码和统计语言模型。共享 Tokenizer 从阶段 04 开始供新项目使用：

```text
02、03：独立历史基线

projects/tokenization
       ↑
      04
       ↑
   05 及后续阶段
```

这种依赖方向允许后续代码复用，同时不让历史阶段反向依赖后来才学习的抽象。

## 8. 限制

- 当前一个 Unicode 字符就是一个普通 token；
- 没有 Unicode normalization；
- 没有 `<PAD>` 和 attention mask；
- 没有 BPE 或其他 subword 合并规则；
- `<UNK>` 可以保证编码继续，但会丢失原始未知字符的信息；
- Tokenizer 和计数 Bigram 正确不代表模型具有语言理解能力。
