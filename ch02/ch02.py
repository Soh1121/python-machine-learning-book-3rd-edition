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


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1] + 1))
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(0.0 <= self.activation(self.net_input(X)), 1, -1)


class AdalinSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        self._initialize_weight(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weight(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weight(xi, target)
        else:
            self._update_weight(X, y)
        return self

    def _initialize_weight(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weight(self, xi, target):
        output = self.activation(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def activation(self, X):
        return X

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(0.0 <= self.activation(self.net_input(X)), 1, -1)


def plot_decision_region(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ["s", "x", "o", "^", "v"]
    colors = ["red", "blue", "lightgreen", "gray", "cyan"]
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)

    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    # 軸の範囲の固定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor="black")


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


# # 決定領域のプロット
# plot_decision_region(X, y, classifier=ppn)

# # 軸のラベルの設定
# plt.xlabel("sepal length [cm]")
# plt.ylabel("petal length [cm]")

# # 凡例の設定
# plt.legend(loc="upper left")

# # 図の表示
# plt.show()

# # 描画領域を1行2列に分割
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# # 勾配降下法によるADALINEの学習をada1として実行
# ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
# # エポック数とコストの関係を表す折れ線グラフをプロット（縦軸のコストは常用対数）
# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
# # 軸のラベルの設定
# ax[0].set_xlabel = "Epochs"
# ax[0].set_ylabel = "log(Sum-squared-error"
# # タイトルの設定
# ax[0].set_title = "Adaline - Learning rate 0.01"
# # 勾配降下法によるADALINEの学習（学習率 eta=0.0001）をada2として実行
# ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
# # エポック数とコストの関係を表す折れ線グラフをプロット（縦軸のコストは常用対数）
# ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker="o")
# # 軸のラベルの設定
# ax[1].set_xlabel = "Epochs"
# ax[1].set_ylabel = "log(Sum-squared-error"
# # タイトルの設定
# ax[1].set_title = "Adaline - Learning rate 0.0001"
# plt.show()

# データのコピー
X_std = np.copy(X)
# 各列の標準化
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# # print(X_std[:5, :])

# # 勾配降下法によるADALINEの学習（標準化後、学習率eta=0.01）
# ada_gd = AdalineGD(n_iter=15, eta=0.01)
# # モデルの適合
# ada_gd.fit(X_std, y)
# # 境界領域のプロット
# plot_decision_region(X_std, y, classifier=ada_gd)
# # タイトルの設定
# plt.title("Adaline - Gradient Descent")
# # 軸のラベルの設定
# plt.xlabel("sepal length [standardized]")
# plt.ylabel("petal length [standardized]")
# # 凡例の設定
# plt.legend(loc="upper left")
# # 図の表示
# plt.tight_layout()
# plt.show()
# # エポック数とコストの関係を表す折れ線グラフのプロット
# plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker="o")
# # 軸のラベルの設定
# plt.xlabel("Epochs")
# plt.ylabel("Sum-squared-error")
# # 図の表示
# plt.tight_layout()
# plt.show()
