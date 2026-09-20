# 10 运行报告

2026-09-20 运行 `python3 -m projects.stage10_inference_serving.report`，使用阶段 08 重新生成的 `mini-model-char-transformer` v1 推理文件。模型有 5,800 个参数，参数本身占 23,200 bytes；设备为 CPU，环境为 macOS 27.0 arm64、Python 3.12.3、PyTorch 2.3.1。

## 固定切分评估

语料为 `data/declaration_excerpt.txt`，沿用顺序 80/10/10 切分：训练 521、验证 65、测试 66 个字符。运行时为每个切分独立添加 BOS/EOS，每个目标 token 只计算一次。验证 perplexity 为 `17.090847`，测试 perplexity 为 `17.965282`。这一口径不同于阶段 08 对重叠定长窗口求平均的训练报告，因此数值不应直接当作回归差异。

固定 `seed=7`、`temperature=1.0`、`top_k=5` 和 40 个新增 token 时：

- `We hold` → ` ine t ther ir ind tonalen intont o e an`
- `Governments` → `t i is ther er ind tonalen intont o e an`

两个样例的流式片段拼接均与非流式 `generated_text` 相同。它们只说明接口一致且模型学到少量局部字符模式，文本仍不连贯，不能据此宣称生成质量良好。

## 性能记录

固定输入为 `We hold` 和 `Governments`，含 BOS 时分别为 8 和 12 个 token；每条最多生成 16 个 token。加载耗时 `5.60 ms`，加载后未预热首请求耗时 `16.08 ms`。加载测量没有清除操作系统文件缓存，因此不是硬件级全冷启动。

在 3 次预热后进行了 10 次正式顺序批量测量。每次含 2 个 prompt，均生成 32 个 token。均值 `72.99 ms`、P50 `50.68 ms`、P95 `223.67 ms`、标准差 `55.73 ms`，总计 320 个生成 token，吞吐 `438.43 token/s`。P95 采用 nearest-rank，样本只有 10 次，容易受单次调度抖动影响。

测量进程的生命周期峰值 RSS 为 178,946,048 bytes（约 170.66 MiB）。它包含 Python、PyTorch、模型加载、评估和此前分配，并非模型独占峰值；模型参数字节数才是独立可计算的 23,200 bytes。以上数字只属于该教学小模型和本次 CPU 环境，不代表大模型、GPU、网络或并发服务性能。机器生成的完整 JSON 位于被 Git 忽略的 `projects/stage10_inference_serving/artifacts/inference_report.json`，可用同一命令重新生成。
