from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
# scikitlearnのsvmモジュールからSVCクラスをインポート
from sklearn.svm import SVC
# メモリ容量を考慮してモデルのインスタンス生成を行うクラスのインポート
from sklearn.linear_model import SGDClassifier
# 決定木を実装するためにscikit-learnのtreeモジュールからDecisionTreeClassifierをインポート
from sklearn.tree import DecisionTreeClassifier
# scikit-learnからtreeモジュールをインポート
from sklearn import tree
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("ggplot")


# ロジスティック回帰の実装
class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) -(1 - y).dot(np.log(1 - output))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(0.0 <= self.net_input(X), 1, 0)


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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_1(z):
    return -np.log(sigmoid(z))


def cost_0(z):
    return -np.log(1 - sigmoid(z))


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
# plot_decision_region(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
# # 軸のラベルの設定
# plt.xlabel = "petal length [standardized]"
# plt.ylabel = "petal width [standardized]"
# # 凡例の設定
# plt.legend(loc="upper left")
# # グラフを表示
# plt.show()


# # -7以上7未満の範囲にある値のシグモイド関数をプロット
# # 0.1間隔で-7以上7未満のデータを生成し変数zに代入
# z = np.arange(-7.0, 7.0, 0.1)
# # 生成したデータでシグモイド関数を実行し、変数phi_zに代入
# phi_z = sigmoid(z)
# # 元のデータとシグモイド関数の出力を線グラフでプロット
# plt.plot(z, phi_z)
# # z = 0 に垂直線を追加する
# plt.axvline(color="k")
# # y軸の上限／下限を設定
# plt.ylim(-0.1, 1.1)
# #  軸のラベルを設定
# plt.xlabel("z")
# plt.ylabel("$\phi(z)$")
# # y軸の目盛りを追加
# plt.yticks([0.0, 0.5, 1.0])
# # Axesクラスのオブジェクトを取得
# ax = plt.gca()
# # y軸の目盛りに合わせて水平グリッド線を追加
# ax.yaxis.grid(True)
# # グラフを表示
# plt.tight_layout()
# plt.show()

# # シグモイド関数の様々な値に対する単一の訓練データの分類コストを示すグラフをプロット
# # 0.1間隔で-10以上10未満のデータを生成し、変数zへ代入
# z = np.arange(-10, 10, 0.1)
# # シグモイド関数を実行し、変数phi_zへ代入
# phi_z = sigmoid(z)
# # y=1のコストを計算する関数を実行し、変数c1へ代入
# c1 = [cost_1(x) for x in z]
# # 結果をプロット
# plt.plot(phi_z, c1, label="J(W) if y=1")
# # y=0のコストを計算する関数を実行し、変数c0へ代入
# c0 = [cost_0(x) for x in z]
# # 結果をプロット
# plt.plot(phi_z, c0, label="J(W) if y=0")
# # x軸とy軸の上限 / 下限を設定
# plt.ylim(0.0, 5.1)
# plt.xlim(0, 1)
# # 軸のラベルを設定
# plt.xlabel("$\phi$(z)")
# plt.ylabel("J(w)")
# # 凡例を設定
# plt.legend(loc="upper center")
# # グラフを表示
# plt.tight_layout()
# plt.show()

# # ロジスティック回帰の実装の確認
# # 学習データをIris-SetosaとIris-Versicolorのみに絞りり、特徴量を変数X_train_01_subnetへ、目的変数を変数y_train_01_subnetへ代入
# X_train_01_subnet = X_train[(y_train == 0) | (y_train == 1)]
# y_train_01_subnet = y_train[(y_train == 0) | (y_train == 1)]
# # ロジスティック回帰のインスタンスを生成し、変数lrgdへ代入
# lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
# # モデルを訓練データに適合させる
# lrgd.fit(X_train_01_subnet, y_train_01_subnet)
# # 決定領域をプロット
# plot_decision_region(X=X_train_01_subnet, y=y_train_01_subnet, classifier=lrgd)
# # ラベルを設定
# plt.xlabel("petal length [standardized]")
# plt.ylabel("petal width [standardized]")
# # 凡例の表示
# plt.legend(loc="upper left")
# # プロットを表示
# plt.tight_layout()
# plt.show()

# scikit-learnを使ってより最適なロジスティック回帰を実装
#  ロジスティック回帰のインスタンスを生成し、変数lrに代入する
lr = LogisticRegression(C=100.0, random_state=1, solver="lbfgs", multi_class="ovr")
# 訓練データをモデルに適合させる
lr.fit(X_train_std, y_train)
# # 決定境界をプロット
# plot_decision_region(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# # 軸のラベルを設定
# plt.xlabel("petal length [standardized]")
# plt.ylabel("petal width [standardized]")
# # 凡例を設定
# plt.legend(loc="upper left")
# # グラフを表示
# plt.tight_layout()
# plt.show()

# # クラス予測を実施
# # クラスの所属確率を算出
# print(lr.predict_proba(X_test_std[:3, :]))
# # クラスのラベルの予測値を算出
# print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
# # 最尤のクラスラベルを表示
# print(lr.predict(X_test_std[:3, :]))
# # 1行のデータの場合は2次元配列に変換して表示
# print(lr.predict(X_test_std[0, :].reshape(1, -1)))

# # 正則化の強さを可視化する
# # 空のリストを生成（重み係数、逆正則化パラメータ）
# weights, params = [], []
# # 10個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
# for c in np.arange(-5, 5):
#     # ロジスティック回帰モデルを生成
#     lr = LogisticRegression(C=10.**c, random_state=1, solver='lbfgs', multi_class='ovr')
#     # 学習を実行
#     lr.fit(X_train_std, y_train)
#     # 重み係数を格納
#     weights.append(lr.coef_[1])
#     # 逆正則化パラメータを格納
#     params.append(10.**c)
# # 重み係数をNumpy配列に変換
# weights = np.array(weights)
# # 横軸に逆正則化パラメータ、縦軸に重み係数をプロット
# plt.plot(params, weights[:, 0], label='petal length')
# plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# # 横軸を対数スケールに設定
# plt.xscale('log')
# plt.show()

# # SVMを利用したIrisデータセットの品種分類
# # 線形SVMのインスタンスを生成
# svm = SVC(kernel='linear', C=1.0, random_state=1)
# # 線形SVMのモデルに訓練データを適合させる
# svm.fit(X_train_std, y_train)
# # 境界領域とデータをプロット
# plot_decision_region(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# # X軸のラベルをセット
# plt.xlabel('petal length [standardized]')
# # Y軸のラベルをセット
# plt.ylabel('petal width [standardized]')
# # 凡例を左上にセット
# plt.legend(loc='upper left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # メモリ容量を考慮した機械学習モデルのインスタンス生成
# # 確率的勾配降下法バージョンのパーセプトロンを生成
# ppn = SGDClassifier(loss='perceptron')
# # 確率的勾配降下法バージョンのロジスティック回帰を生成
# lr = SGDClassifier(loss='log')
# # 確率的勾配降下法バージョンのSVM（損失関数=ヒンジ関数）を生成
# svm = SGDClassifier(loss='hinge')

# # 線形分離不可能なデータの生成と確認
# 乱数シードを指定
np.random.seed(1)
# 標準正規分布に従う乱数で200行2列の行列を生成
X_xor = np.random.randn(200, 2)
# 2つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# 排他的論理和の値が真の場合は1、偽の場合は-1を割り当てる
y_xor = np.where(y_xor, 1, -1)
# # ラベル1を青のxでプロット
# plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
# # ラベル-1を赤の四角でプロット
# plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
# # 軸の範囲を設定
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# # 凡例を右上に
# plt.legend(loc='upper right')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # カーネルSVMの訓練を行い、XORデータを分割する非線形の決定境界を描けるか確認
# # RBFカーネルによるSVMのインスタンスを生成
# svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
# # 訓練データを適合
# svm.fit(X_xor, y_xor)
# # 決定境界をプロット
# plot_decision_region(X_xor, y_xor, classifier=svm)
# # 凡例を左上に表示
# plt.legend(loc='upper left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # RBFカーネルSVMをIrisデータセットに適用
# # RBFカーネルによるSVMのインスタンスを生成（2つのパラメータを変更）
# svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
# # 訓練データを適合
# svm.fit(X_train_std, y_train)
# # 決定境界をプロット
# plot_decision_region(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# # 軸のラベルを設定
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# # 凡例を左上に表示
# plt.legend(loc='upper left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # RBFカーネルSVMをIrisデータセットにパラメータを極端にして適用
# # RBFカーネルによるSVMのインスタンスを生成（γパラメータを変更）
# svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
# # 訓練データを適合
# svm.fit(X_train_std, y_train)
# # 決定境界をプロット
# plot_decision_region(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# # 軸のラベルを設定
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# # 凡例を左上に表示
# plt.legend(loc='upper left')
# # プロットを表示
# plt.tight_layout()
# plt.show()


# # 3種類の不純度条件を視覚的に比較するために不純度の指標をプロット
# # ジニ不純度の関数を定義
# def gini(p):
#     return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))


# # エントロピーの関数を定義
# def entropy(p):
#     return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# # 分類誤差の関数を定義
# def error(p):
#     return 1 - np.max([p, 1 - p])


# # 確率を表す配列を生成（0から0.99まで0.01刻み）
# x = np.arange(0.0, 1.0, 0.01)

# # 配列の値を元にエントロピー、分類誤差を計算
# # 3項演算子を利用してp = 0のときに対応
# ent = [entropy(p) if p != 0 else None for p in x]
# # スケーリングバージョンのエントロピー
# sc_ent = [e*0.5 if e else None for e in ent]
# # 分類誤差
# err = [error(i) for i in x]
# # 図の作成を開始
# fig = plt.figure()
# ax = plt.subplot(111)
# # エントロピー（2種）、ジニ不純度、分類誤差のそれぞれをループ処理
# for i, lab, ls, c in zip(
#     [ent, sc_ent, gini(x), err],
#     ['Entropy', 'Entropy (scaled)', 'Gini impurity', 'Misclassification error'],
#     ['-', '-', '--', '-.'],
#     ['black', 'lightgray', 'red', 'green', 'cyan']
#     ):
#     line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
# # 凡例の設置（中央の上）
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
# # 2本の水平線の破線を引く
# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# # 横軸の上限/下限を設定
# plt.ylim([0, 1.1])
# # ラベルを設定
# plt.xlabel('p(i=1)')
# plt.ylabel('impurity index')
# # グラフの表示
# plt.show()

# ジニ不純度を指標とする決定木のインスタンスを生成
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
# 決定木のモデルを訓練データに適合させる
tree_model.fit(X_train, y_train)
# # 訓練データとテストデータを図示するために結合（標準化を行わない）
# X_combined = np.vstack((X_train, X_test))
# y_combined = np.hstack((y_train, y_test))
# # 決定境界をプロット
# plot_decision_region(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
# # 軸のラベルを設定
# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# # 凡例を左上に表示
# plt.legend(loc='upper left')
# # グラフの表示
# plt.tight_layout()
# plt.show()

# scikit-learnを用いくて決定木モデルを可視化
# treeモジュールのplot_tree()関数を利用
tree.plot_tree(tree_model)
# 決定木を可視化
plt.show()
