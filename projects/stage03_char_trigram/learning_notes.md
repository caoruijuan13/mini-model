# 03 字符 trigram 学习资料

- [README.md](README.md)：项目定位和运行方式
- [design.md](design.md)：三维概率、稀疏上下文、backoff 和采样
- [test_report.md](test_report.md)：测试逻辑与边界
- [run_report.md](run_report.md)：真实语料训练和生成结果
- [data.py](data.py)：真实语料与数据切分
- [model.py](model.py)：trigram 模型
- [train.py](train.py)：训练、评估和保存
- [generate.py](generate.py)：加载和生成
- [tests/test_char_trigram.py](tests/test_char_trigram.py)：自动化测试

建议先理解项目 02 的二维 bigram 概率矩阵，再阅读本项目的三维 trigram 概率矩阵和 bigram backoff。
