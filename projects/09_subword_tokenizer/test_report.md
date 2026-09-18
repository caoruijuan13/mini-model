# 09 测试报告

2026-09-18 在当前工作树运行：

```text
python3 -m pytest -q projects/09_subword_tokenizer/tests/test_stage09.py
26 passed

python3 -m pytest -q
113 passed
```

阶段测试覆盖配置上下限、与 07 相同的原文顺序切分、词表与 merge 顺序校验、重叠 pair 计数、频数和并列选择、从左到右非重叠替换，以及 `abab` 的两轮手算结果。还验证了规则顺序编码与已知文本往返、未知字符和 BOS/EOS 处理、非法 `on_unknown` 策略、非整数 token ID 拒绝、保存加载 ID 稳定性，以及不兼容格式名或版本的加载拒绝。对照脚本的测试确认两种 tokenizer 使用同一行分段，BPE 只从训练文本学习。

这些测试证明当前教学实现的指定行为，不证明其覆盖生产级 Unicode 规范化、byte fallback、词边界预处理、吞吐性能或不可信文件安全性。`torch.load` 读取的二进制 tokenizer 文件只应来自可信来源。真实语料指标另见 [运行报告](run_report.md)。
