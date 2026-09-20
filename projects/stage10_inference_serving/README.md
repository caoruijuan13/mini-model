# 10：评估、推理与服务化

本阶段从阶段 08 已保存的字符级 Transformer 推理文件出发，学习如何在不读取训练语料、不创建优化器的情况下加载模型，并逐步建立评估、单条与批量推理、流式生成和性能测量。当前处于**进行中**；加载、生成、perplexity 评估及冷启动与预热后性能测量已有基础实现，实际性能报告尚未完成。完整目标见 [项目路线图](../../ROADMAP.md)。

## 当前入口

从项目根目录运行：

```bash
python3 -m pytest -q projects/stage10_inference_serving/tests/test_stage10.py
```

当前可用的 Python 入口是 `InferenceRuntime.load(path)`，它调用阶段 08 的 `load_with_tokenizer()`，一次加载模型权重、配置和完整字符词表：

```python
from projects.stage10_inference_serving.runtime import InferenceRuntime

runtime = InferenceRuntime.load(
    "projects/stage08_pytorch_transformer/artifacts/char_transformer.pt"
)
```

示例文件由阶段 08 的训练命令生成，位于被忽略的 `artifacts/` 目录。加载仅接受可信的本地模型文件；文件缺失或内容不兼容时，错误会传给调用者。

`runtime.py` 已实现生成、perplexity 评估和性能测量：批量方法当前按顺序逐条调用单条方法，使用 `seed + 输入序号`，并非一次张量批处理；流式方法每生成一个普通 token 就产出对应片段，EOS 只结束迭代；perplexity 对每段文本独立添加 BOS/EOS，并对每个目标 token 计数一次。`benchmark_from_path()` 分别记录加载冷启动和未预热首请求，`benchmark()` 在已加载运行时上执行额外预热并报告延迟分布和生成吞吐。冷启动测量不会主动清除操作系统文件缓存。实际性能报告和对外服务入口仍待完成。

## 文件导航

- [设计](design.md)：输入输出、数据边界和后续接口契约。
- [学习笔记](learning_notes.md)：本阶段的概念与实现顺序。
- [测试报告](test_report.md)：已验证的加载与生成行为、待补的验收证据。
- [运行报告](run_report.md)：当前运行结果以及性能报告所需的记录项。
- `runtime.py`：加载、单条、顺序批量、流式生成、perplexity 评估与性能测量入口。
- `tests/test_stage10.py`：不依赖训练语料的加载与生成契约测试。

阶段 09 的 BPE tokenizer 尚未与阶段 08 的字符模型共同训练，不能直接替换该推理文件中的字符词表。
