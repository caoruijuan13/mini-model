# 09 设计契约

## 数据边界

`prepare_text_splits()` 原样复用 07/08 的语料读取与顺序 80/10/10 切分。`BPETokenizer.train()` 只接收 `train_text`；验证和测试文本只能交给已冻结的 `encode()`。`compare_tokenizers()` 对相同的原文行分别计数，默认不添加 BOS/EOS，使统计单位一致。

## 最小 BPE 定义

- 初始单位：单个 Unicode 字符；初始词表沿用 `CharTokenizer.from_text(train_text)` 的特殊 token 和排序规则，但 BPE 的 pair/merge 算法独立实现；
- pair：当前 token ID 序列中每一对相邻 ID，计数时允许重叠；
- 选取：先比较出现次数，频数相同则选 `(left_id, right_id)` 字典序最小者；低于 `min_pair_frequency` 或达到 `max_merges` 时停止；
- 合并：一次规则应用从左到右替换非重叠 pair；第 `i` 条规则创建的新 ID 为 `len(base_tokens) + i`；新 token 的文本是左右 token 文本的拼接；
- 编码：将新文本拆成字符 ID，再按训练时记录的 merge 顺序逐条应用；不能用验证/测试文本重新学习规则；
- 解码：按 ID 找回文本并拼接。已知字符在不插入特殊 token 的范围内应可逆；未知字符若映射为 `<UNK>`，原文信息已经丢失。
- 持久化：保存初始/最终词表、有序 merge、特殊 token、格式名和版本；加载时校验格式、版本和词表/规则的一致性。当前使用 PyTorch 二进制 `.pt` 文件，仅加载可信来源。

本教学版本不实现 byte fallback、词边界预处理、生产级 Unicode 规范化或最快的 BPE 算法。普通空格、标点和换行也属于可统计的字符；是否跨行合并由训练输入序列决定，不得跨训练/验证/测试边界。

## 输出和比较

`compare_tokenizers()` 不比较两套模型的 loss；只比较同一原文、同一行分段方式下的词表大小、总 token 数和每行平均 token 数。不同 tokenizer 的“每 token perplexity”单位不同，不能直接排名。09 阶段也不声称子词必然改善 08 的模型效果。
