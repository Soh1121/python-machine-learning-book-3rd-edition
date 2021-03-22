import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):   # 活性化関数はステップ関数
        return np.where(0.0 <= self.net_input(X), 1, -1)


def plot_decision_region(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ["s", "x", "o", "^", "v"]
    colors = ["red", "blue", "lightgreen", "gray", "cyan"]
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[: 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[: 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
# print(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

#### 2.3
a = np.arange(3)
# print(a)
b = np.arange(3, 6)
# print(b)
# print(a * b)

s = os.path.join(
    "https://archive.ics.uci.edu",
    "ml",
    "machine-learning-databases",
    "iris",
    "iris.data")
# print("URL:", s)
df = pd.read_csv(s, header=None, encoding="utf-8")
# print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == "Iris-versicolor", 1, -1)
# print(y)

X = df.iloc[:100, [0, 2]].values
# print(x)

# plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
# plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
# plt.xlabel="sepal length [cm]"
# plt.ylabel="sepal width [cm]"
# plt.legend(loc="upper left")
# plt.show()

# パーセプトロンオブジェクトの生成（インスタンス化）
ppn = Perceptron(eta=0.1, n_iter=10)

# 訓練データへのモデルの適合
ppn.fit(X, y)

# エポックと誤分類の関係を表す折れ線グラフをプロット
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")

#  軸のラベルの設定
# plt.xlabel = "Epochs"
# plt.ylabel = "Number of update"

# 図の表示
# plt.show()

