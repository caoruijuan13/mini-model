import numpy as np

from data import make_data
from model import Model


def test_result(model, x: np.ndarray, y: np.ndarray) -> float:
    predict_result = model.predict_result(x)
    return np.mean(predict_result == y)

def test_learning():
    num = 3000
    seed = 3
    train, valid, test = make_data(num, seed)
    print(train.x.shape, train.y.shape)
    model = Model(2, 0.7, seed)
    best_loss = model.train(train, valid, epochs=1000, patience=30, min_delta=1e-4)
    test_res = test_result(model, test.x, test.y)
    print(f"{model.w[0]:.2f}", f"{model.w[1]:.2f}", 
          f"{model.b:.2f}", f"{best_loss:.6f}", f"{test_res:.6f}")


if __name__ == "__main__":
    test_learning()
