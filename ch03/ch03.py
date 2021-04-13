from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_region(X, y, classifier, resolution=0.02, test_idx=None):
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
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c="", edgecolor="black", alpha=1.0, linewidth=1, marker="o", s=100, label="test set")


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# print("Class labels:", np.unique(y))

# 全体の30%がテストデータとなるよう訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# データセットのクラスラベルを確認
# print("Label counts in y: ", np.bincount(y))

# 訓練データセットのクラスラベルを確認
# print("Label counts in y_train: ", np.bincount(y_train))

# テストデータセットのクラスラベルを確認
# print("Label counts in y_test", np.bincount(y_test))

# インスタンス変数scにStandardScalerクラスを代入
sc = StandardScaler()
# 訓練データの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 学習率eta0.1、乱数シードrandom_state1でパーセプトロンのインスタンスをppn変数に生成
ppn = Perceptron(eta0=0.1, random_state=1)
# 訓練データをモデルに適合させる
ppn.fit(X_train_std, y_train)

# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
# 誤分類データの個数を表示
# print("Misclassified examples: %d" % (y_test != y_pred).sum())

# 分類の正解率を表示
# print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

# 分類機に定義されているscoreメソッドを使って正解率を表示する
# print("Accuracy: %.3f" % ppn.score(X_test_std, y_test))

# 訓練データとテストデータの特徴量を行方向に結合し、変数X_combined_stdに代入
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合し、変数y_combinedに代入
y_combined = np.hstack((y_train, y_test))
# 決定境界のプロット
plot_decision_region(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
# 軸のラベルの設定
plt.xlabel = "petal length [standardized]"
plt.ylabel = "petal width [standardized]"
# 凡例の設定
plt.legend(loc="upper left")
# グラフを表示
plt.show()
