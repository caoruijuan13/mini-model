# 02 字符 bigram 学习资料

本项目将设计、测试和运行结果分开记录：

- [design.md](design.md)：问题、输入输出、计数模型、损失、生成和边界
- [test_report.md](test_report.md)：测试用例、测试目的和测试结论
- [run_report.md](run_report.md)：一次训练、评估和保存模型的实际运行
- [README.md](README.md)：运行入口和文件导航
- [model.py](model.py)：bigram 模型实现
- [train.py](train.py)：训练评估入口
- [generate.py](generate.py)：加载模型后的生成入口
- [tests/test_char_bigram.py](tests/test_char_bigram.py)：自动化测试

建议阅读顺序：先看设计中的输入输出和概率矩阵，再看 `model.py` 的 `fit()`、`loss()` 和 `sample()`，最后对照测试报告理解每个断言验证了什么。
