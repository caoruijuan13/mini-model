# 04 字符 Tokenizer 测试报告

## 1. 测试范围

测试文件为 [tests/test_char_tokenizer.py](tests/test_char_tokenizer.py) 和 [tests/test_token_bigram.py](tests/test_token_bigram.py)，验证共享 `CharTokenizer` 的编码契约，以及只接收 token ID 的 `TokenBigram` 训练和采样行为。

## 2. 执行方式

```bash
python3 -m pytest -q projects/04_char_tokenizer/tests
```

## 3. 测试内容

### 3.1 确定性 vocabulary 与往返转换

验证特殊 token 顺序固定、普通字符按确定顺序排列、公开映射不可修改，并检查已知文本经过 encode/decode 后保持一致。

### 3.2 特殊 token 与空文本

验证 BOS/EOS 只在显式请求时加入，解码时可以保留或跳过特殊 token，空文本也有明确结果。

### 3.3 训练 vocabulary 与未知字符

只从训练文本 `aaab` 建立 vocabulary，再编码包含新字符的 `ac`，分别验证 `<UNK>` 和错误两种策略。

### 3.4 保存与加载

验证 JSON 格式名称和版本，并确认加载后的 vocabulary、映射和编码结果与保存前一致。

### 3.5 数据集职责和错误边界

验证 token ID 产生之后才能构造 bigram 样本，同时检查越界 ID 和布尔值 ID 会产生明确错误。

### 3.6 Token Bigram 概率

验证模型使用 `vocab_size` 创建概率矩阵、浮点平滑不会被整数截断，并检查每一行概率归一化为 1。

### 3.7 Token ID 采样

验证每一步根据上一个生成 ID 选择概率分布、固定 seed 可复现、返回值是 token ID 列表，并支持采样到 EOS 后提前停止。

### 3.8 状态和参数错误

验证未训练模型不能采样、越界 token ID 被拒绝，并明确要求生成长度至少包含起始 token。

## 4. 当前结果

```text
9 passed
```

## 5. 测试边界

这些测试验证字符级 Tokenizer 和计数 Bigram 的接口契约，不验证 BPE、Unicode normalization、padding、embedding、神经网络训练或语言生成质量。
