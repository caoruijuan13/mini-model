import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    bounded_z = np.clip(z, -40, 40)  # 极端值下避免 exp 数值溢出。
    return 1 / (1 + np.exp(-bounded_z))

class Model:
    seed: int
    data_num: int
    learning_rate:float

    def __init__(
        self,
        data_num: int = 2,
        learning_rate: float = 0.1,
        seed: int = 7,
    ):
        self.data_num = data_num
        self.learning_rate = learning_rate
        self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=(self.data_num,))
        self.b = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.w) + self.b
        return sigmoid(z)

    def predict_result(self, x: np.ndarray) -> np.ndarray:
        return (self.predict(x) >= 0.5).astype(np.int64)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict(x), 1e-8, 1 - 1e-8)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        error = p - y
        gradient_w = x.T @ error / len(x)
        gradient_b = np.mean(error)
        self.w -= gradient_w * self.learning_rate
        self.b -= gradient_b * self.learning_rate
        return self.loss(x, y)

    def train(
        self,
        train,
        valid,
        epochs: int = 1000,
        patience: int = 30,
        min_delta: float = 1e-4,
    ) -> float:
        """训练模型，并恢复验证集表现最好的参数。"""
        best_valid_loss = float("inf")
        best_w = self.w.copy()
        best_b = self.b
        wait = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.step(train.x, train.y)
            valid_loss = self.loss(valid.x, valid.y)
            if valid_loss < best_valid_loss - min_delta:
                best_valid_loss = valid_loss
                best_w = self.w.copy()
                best_b = self.b
                wait = 0
            else:
                wait += 1
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                print(epoch, f"{self.w[0]:.2f}", f"{self.w[1]:.2f}",
                    f"{self.b:.2f}", 
                    f"{best_w[0]:.2f}", f"{best_w[1]:.2f}", 
                    f"{best_b:.2f}", 
                    f"train_loss: {train_loss:.2f}",
                    f"valid_loss: {valid_loss:.2f}")
            if wait >= patience:
                print(f"early stop at epoch {epoch}")
                break

        # 无论是 early stopping 还是跑满 epochs，都使用验证集最佳参数。
        self.w = best_w
        self.b = best_b
        return best_valid_loss
    
