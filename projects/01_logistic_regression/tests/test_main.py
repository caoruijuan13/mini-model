"""针对 main.py 的学习型测试。

这些测试不只检查最终准确率，也分别检查数据、概率、损失和梯度更新。
"""

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
import sys

sys.path.insert(0, str(ROOT))

from data import make_data
from model import Model, sigmoid

SPEC = importlib.util.spec_from_file_location("logistic_main", ROOT / "main.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)


def test_make_data_is_reproducible_and_has_expected_shapes():
    """相同 seed 应产生相同数据，切分比例和标签形状也应正确。"""
    train_a, valid_a, test_a = MAIN.make_data(100, seed=7)
    train_b, valid_b, test_b = MAIN.make_data(100, seed=7)

    # 训练/验证/测试分别是 60/20/20；每个样本有两个特征。
    assert train_a.x.shape == (60, 2)
    assert valid_a.x.shape == (20, 2)
    assert test_a.x.shape == (20, 2)
    assert train_a.y.shape == (60,)

    # 固定随机种子是为了让实验可以复现。
    assert np.array_equal(train_a.x, train_b.x)
    assert np.array_equal(valid_a.y, valid_b.y)
    assert set(np.unique(train_a.y)).issubset({0, 1})


def test_sigmoid_returns_probability_and_is_monotonic():
    """sigmoid 输出应在 [0,1] 内，并且输入越大，输出不能变小。"""
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    p = sigmoid(z)

    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) >= 0.0)
    assert np.isclose(p[2], 0.5)


def test_binary_cross_entropy_known_value():
    """当 logit=0 时 p=0.5，标签 0/1 的平均 loss 都应为 log(2)。"""
    model = Model(data_num=1, learning_rate=0.1, seed=7)
    model.w[:] = 0.0
    model.b = 0.0
    x = np.zeros((2, 1))
    y = np.array([0, 1])

    assert np.isclose(model.loss(x, y), np.log(2.0))


def test_one_gradient_step_moves_towards_positive_label():
    """对于 x=1、y=1 且 p=0.5，梯度下降应增大 w、b 并降低 loss。"""
    model = Model(data_num=1, learning_rate=0.1, seed=7)
    model.w[:] = 0.0
    model.b = 0.0
    x = np.array([[1.0]])
    y = np.array([1])
    before = model.loss(x, y)

    after = model.step(x, y)

    assert model.w[0] > 0.0
    assert model.b > 0.0
    assert after < before


def test_training_improves_and_generalizes():
    """完整训练后，验证集最佳 loss 应低于初始值，测试准确率应达到基线。"""
    train, valid, test = make_data(1000, seed=17)
    model = Model(data_num=2, learning_rate=0.7, seed=17)
    initial_valid_loss = model.loss(valid.x, valid.y)

    best_valid_loss = model.train(train, valid, epochs=300)
    test_accuracy = MAIN.test_result(model, test.x, test.y)

    assert best_valid_loss < initial_valid_loss
    assert test_accuracy > 0.8


def test_training_restores_best_validation_parameters():
    """训练结束后，当前参数应达到记录的最佳验证集 loss。"""
    train, valid, _ = make_data(1000, seed=17)
    model = Model(data_num=2, learning_rate=0.7, seed=17)

    best_valid_loss = model.train(
        train,
        valid,
        epochs=300,
        patience=10,
        min_delta=1e-4,
    )

    assert np.isclose(model.loss(valid.x, valid.y), best_valid_loss, atol=1e-12)
