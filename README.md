# 从小模型开始，逐步理解和实践机器学习

本项目采用“输入 → 模型 → 输出”的方式，一次学习一个问题，通过手写实现、对照实验和自动化测试逐步理解机器学习与语言模型。

按主题归类的完整概念说明、公式、常见混淆和自测问题见 [项目知识库](KNOWLEDGE_BASE.md)。

当前已完成逻辑回归、字符 bigram、字符 trigram、显式字符 Tokenizer、Token MLP、训练工程基础、字符级 Transformer、PyTorch 模型实践、手写 Subword Tokenizer，以及评估、推理与服务化十个阶段。后续学习顺序、阶段目标和验收标准见 [项目路线图](ROADMAP.md)。

编号阶段的目录统一以 `stageXX_` 开头，因此可以使用普通 Python 导入语句，例如 `from projects.stage08_pytorch_transformer.model import TorchCharTransformer`。`projects/tokenization` 是跨阶段共享包，不属于编号阶段。

## 当前状态

| 阶段 | 状态 | 核心内容 | 当前证据 |
| --- | --- | --- | --- |
| [01 逻辑回归](projects/stage01_logistic_regression/) | 已完成 | sigmoid、二元交叉熵、梯度下降、early stopping | 6 个测试；测试集 accuracy `91.8333%` |
| [02 字符 bigram](projects/stage02_char_bigram/) | 已完成 | 一个字符预测下一个字符、平滑、困惑度、采样、持久化 | 3 个测试；教学语料与真实语料对照 |
| [03 字符 trigram](projects/stage03_char_trigram/) | 已完成 | 两字符上下文、验证集选择平滑、bigram backoff | 3 个测试；真实语料测试 perplexity `17.423710` |
| [04 显式字符 Tokenizer](projects/stage04_char_tokenizer/) | 已完成 | vocabulary、特殊 token、encode/decode、Token Bigram 采样 | 9 个测试；只使用训练文本建立词表 |
| [05 Token MLP](projects/stage05_token_mlp/) | 已完成 | embedding、隐藏层、反向传播、梯度下降和自回归生成 | 11 个测试；最佳验证 perplexity `16.681468` |
| [06 训练工程基础](projects/stage06_training_engineering/) | 已完成 | mini-batch、SGD/Adam、early stopping、checkpoint 和可复现训练 | 11 个测试；中断恢复与连续训练逐元素一致 |
| [07 字符级 Transformer](projects/stage07_char_transformer/) | 已完成 | causal attention、多层 block、手写梯度、训练、验证选择与生成 | 27 个测试；最佳验证 perplexity `17.283519`；测试 perplexity `21.441981` |
| [08 PyTorch 模型实践](projects/stage08_pytorch_transformer/) | 已完成 | PyTorch 前向、autograd、验证选择、生成和自包含推理文件 | 19 个测试；验证 perplexity `16.879234`；测试 perplexity `17.493177` |
| [09 手写 Subword Tokenizer](projects/stage09_subword_tokenizer/) | 已完成 | 字符初始化 BPE、规则编码、特殊 token 与版本化保存加载 | 26 个测试；训练集 token 数 `521→299`，验证集 `65→52`，测试集 `66→52` |
| [10 评估、推理与服务化](projects/stage10_inference_serving/) | 已完成 | 版本化加载、评估、批量与流式生成、本地接口和性能测量 | 31 个测试；固定语料评估与当前环境实测报告 |

“已完成”表示当前阶段形成了独立代码、设计说明、运行报告和自动化测试；它只代表教学项目的当前验收，不代表生产模型能力。

当前全量测试结果：

```text
146 passed
```

阶段 07 和 08 均已有固定配置下的训练、验证与测试报告；09 已完成字符与 BPE tokenizer 的独立对照，结果和未知字符边界见 [阶段 09 运行报告](projects/stage09_subword_tokenizer/run_report.md)。**10 评估、推理与服务化** 已完成版本化加载、固定切分评估、单条与顺序批量生成、流式生成、本地命令行接口和当前环境性能报告。07/08 的短语料、单 seed 指标仅是教学证据；09 的 token 数下降也不能证明模型 loss 或生成质量改善；10 的 CPU 指标不能外推到大模型、GPU 或并发网络服务。完整边界见 [阶段 07 运行报告](projects/stage07_char_transformer/run_report.md)、[阶段 08 运行报告](projects/stage08_pytorch_transformer/run_report.md)、[阶段 10 运行报告](projects/stage10_inference_serving/run_report.md) 和 [项目路线图](ROADMAP.md)。

## Token 在当前项目中的位置

Token 是模型处理文本时使用的离散单位。它可以是一个字符、一个词或词的一部分。

项目 02 和项目 03 实际已经采用“一个字符就是一个 token”的方式，只是还没有独立的 Tokenizer：

```text
文本字符 c
  → to_id[c]
  → ids 中的整数
  → 作为概率矩阵的行列索引
```

当前代码中的概念对应关系是：

| Token 概念 | 当前变量 | 含义 |
| --- | --- | --- |
| token | `character`、`token` | 02–08 的语言模型主要用字符；09 的 BPE 还可用合并后的字符片段 |
| vocabulary | `tokenizer.tokens` | 字符、特殊 token，以及 09 中按规则新增的片段 |
| token → token ID | `tokenizer.token_to_id`、`base_tokens` | 字符 tokenizer 显式映射字符；09 用基础词表位置和有序 merge 分配 ID |
| token ID 序列 | `train_ids` 等 | 模型实际接收的离散整数序列 |
| 上下文/目标 token ID | `context_ids`、`targets` | Token MLP 的自监督训练样本 |
| token embedding | `params["embedding"]` | 由 token ID 查表获得并通过梯度学习的浮点向量 |
| 输出分数与概率 | `logits`、`probs` | 每个候选 token ID 的分数与下一 token 概率 |

因此，阶段 04 不是第一次“使用 token”，而是第一次把隐含的字符编码逻辑抽成明确、可保存、可测试、可复用的 Tokenizer 接口。

阶段 04 的共享实现位于 [projects/tokenization](projects/tokenization/)，学习入口、设计和验证证据位于 [04 字符 Tokenizer](projects/stage04_char_tokenizer/)。后续项目复用共享实现，不再重复定义字符词表和 ID 映射。

## 训练范式

项目按训练目标和反馈来源进行主分类：

| 学习范式 | 核心特征 | 当前项目 | 后续方向 |
| --- | --- | --- | --- |
| 监督学习 | 数据明确提供输入和目标标签 | 01 逻辑回归 | 多分类、回归、结构化预测 |
| 自监督学习 | 从原始数据自动构造预测目标 | 02 bigram、03 trigram、05 Token MLP、07 Transformer | 掩码建模 |
| 无监督学习 | 没有明确标签，学习数据结构或分布 | 暂无 | 聚类、降维、密度估计、表示学习 |
| 强化学习 | 根据环境奖励学习行动策略 | 暂无 | 多臂老虎机、价值函数、策略优化 |

Tokenization 本身不是一种训练范式，而是把原始输入转换为模型离散输入单位的数据处理过程。Embedding 也不是独立训练范式，而是模型内部可学习的表示层。

## 快速开始

运行全部测试：

```bash
python3 -m pytest -q
```

分别运行九个已完成阶段：

```bash
python3 projects/stage01_logistic_regression/main.py
PYTHONPATH=projects/stage02_char_bigram python3 projects/stage02_char_bigram/train.py
PYTHONPATH=projects/stage02_char_bigram python3 projects/stage02_char_bigram/generate.py
PYTHONPATH=projects/stage03_char_trigram python3 projects/stage03_char_trigram/train.py
PYTHONPATH=projects/stage03_char_trigram python3 projects/stage03_char_trigram/generate.py
python3 projects/stage04_char_tokenizer/demo.py
PYTHONPATH=projects/stage05_token_mlp python3 projects/stage05_token_mlp/train.py
PYTHONPATH=projects/stage06_training_engineering python3 projects/stage06_training_engineering/compare_optimizers.py
python3 -m projects.stage07_char_transformer.train
python3 -m projects.stage08_pytorch_transformer.train
python3 -m projects.stage09_subword_tokenizer.compare
python3 -m projects.stage10_inference_serving.report
python3 -m projects.stage10_inference_serving.cli "We hold" --top-k 5
```

阶段 08 的 PyTorch 模型、实验和推理文件说明见 [08 README](projects/stage08_pytorch_transformer/README.md)。
阶段 09 的 BPE 算法、手算顺序与对照结果见 [09 README](projects/stage09_subword_tokenizer/README.md)。
阶段 10 的评估、推理接口和性能报告见 [10 README](projects/stage10_inference_serving/README.md)。
