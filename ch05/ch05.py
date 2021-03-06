# DataFrameやサンプルデータを用いるためpandasをpdとしてインポートする
import pandas as pd
# 訓練データセットとテストデータセットに分割するためscikit-learnのmodel_selectionモジュールからtrain_test_splitをインポートする
from sklearn.model_selection import train_test_split
# 標準化を行うためにscikit-learnのpreprocessingモジュールからStandardScalerをインポート
from sklearn.preprocessing import StandardScaler
# PCAを行うため、scikit-learnのdecompositionモジュールからPCAをインポート
from sklearn.decomposition import PCA
# ロジスティック回帰を行うため、scikit-learnのlinear_modelモジュールからLogisticRegressionをインポート
from sklearn.linear_model import LogisticRegression
# LDAを用いるためにscikit-learnのdisxriminant_analysisモジュールからLinearDiscriminantAnalysisをLDAとしてインポート
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# 共分散行列の固有対を取得するためにnumpyをnpとしてインポート
import numpy as np
# グラフをプロットするためにmatplotlibからpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# PCAによる次元圧縮後の回帰性能を確認するため、決定領域をプロットする関数を用意
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)


# Wineデータセットをインポートする
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data', header=None)
# 2列目以降のデータをXに、1列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# 平均と標準偏差を用いて標準
sc = StandardScaler()
# 訓練データの標準化
X_train_std = sc.fit_transform(X_train)
# テストデータの標準化
X_test_std = sc.fit_transform(X_test)

# 共分散行列を作成
cov_mat = np.cov(X_train_std.T)
# 固有値と固有ベクトルを計算
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('\nEigenvalues \n%s' % eigen_vals)

# # 分散説明率の累積和を確認する
# # 固有値を計算
# tot = sum(eigen_vals)
# # 分散説明率を計算
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# # 分散説明率の累積和を取得
# cum_var_exp = np.cumsum(var_exp)
# # 分散説明率の棒グラフを作成
# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
# # 分散説明率の累積和の階段グラフを作成
# plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumlative explained variance')
# # ラベル名を設定
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# # 凡例を良きところに表示
# plt.legend(loc='best')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# 固有値の大きいものから順に固有対を並べ替える
# (固有値, 固有ベクトル)のタプルのリストを作成
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# (固有値, 固有ベクトル)のタプルを大きいものから順に並べ替え
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# 固有ベクトルから射影行列Wを作成
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# print('Matrix W:\n', w)

# 0行目のデータに対し、射影行列を用いてデータを変換
# print(X_train_std[0].dot(w))

# データ全体を射影行列を用いてデータ変換
# X_train_pca = X_train_std.dot(w)

# # 変換後の訓練データセットをプロット
# # 色とマーカーの設定
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# # 「クラスラベル」「点の色」「点の種類」の組み合わせからなるリストを生成してプロット
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)
# # 軸のラベルを設定
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# # 凡例を左下に表示
# plt.legend(loc='lower left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# scikit-learnを用いてPCAによる変換データを使ったロジスティック回帰を確認
# 主成分数を指定して、PCAのインスタンスを生成
pca = PCA(n_components=2)
# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# 次元削減
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# 削減したデータセットでロジスティック回帰モデルを適合
lr.fit(X_train_pca, y_train)
# 決定領域をプロット
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# # 軸のラベルを設定
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# # 凡例を左下に表示
# plt.legend(loc='lower left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # テストデータでロジスティック回帰の結果を確認
# # 決定領域をプロット
# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# # 軸のラベルを設定
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# # 凡例を左下に表示
# plt.legend(loc='lower left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 分散説明率を確認
# # PCAインスタンスを生成
# pca = PCA(n_components=None)
# # PCAを実行し、データを変換
# X_train_pca = pca.fit_transform(X_train_std)
# # 分散説明率を計算
# print(pca.explained_variance_ratio_)

# 平均ベクトルを生成する
# 浮動小数ん点数の小数点以下の表示桁数を設定
np.set_printoptions(precision=4)
# 平均ベクトルを生成
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    # print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

# クラス内変動行列の生成
# 特徴量の個数
d = 13
# S_Wを算出
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
# print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# クラスラベルが一様に分布しているか確認
# print('Class label distribution: %s' % np.bincount(y_train)[1:])

# スケーリングされたクラス内変動行列を計算
# 特徴量の個数
d = 13
# スケーリングしたクラス内変動行列を計算
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
# print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# クラス間変動S_Bの算出
# すべてのデータ点を対象とした全体平均を算出
mean_overall = np.mean(X_train_std, axis=0)
# 特徴量の個数を設定
d = 13
# クラス間変動S_Bを算出
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    # 列ベクトルを生成
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
# print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

# 固有値を計算
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# 固有値を大きいものから降順で並び替える
# 固有値対を計算
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# 固有値を降順でソート
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
# print('Eigenvalues in descending order:\n')
# for eigen_val in eigen_pairs:
#     print(eigen_val[0])

# # クラスの判別情報を計測
# # 固有値の実数部の総和を求める
# tot = sum(eigen_vals.real)
# # 分散説明率とその累積和を計算
# discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
# cum_discr = np.cumsum(discr)
# # 棒グラフと階段グラフを描画
# plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='Individual "discriminability"')
# plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative "discriminability"')
# # 軸のラベルを設定
# plt.ylabel('"Discriminability" ratio')
# plt.xlabel('Linear Discriminants')
# # y軸の上限、下限を設定
# plt.ylim([-0.1, 1.1])
# # 凡例を適切な位置に表示
# plt.legend(loc='best')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 変換行列を作成
# # 2つの固有ベクトルから変換行列を作成
# w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
# # print('Matrix W:\n', w)

# # 新しい特徴量空間にデータ点を射影する
# # 標準化した訓練データに変換行列をかける
# X_train_lda = X_train_std.dot(w)
# # プロット用の設定
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# # 変換後のデータを散布図にプロット
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train == l, 0], X_train_lda[y_train == l, 1] * (-1), c=c, label=l, marker=m)
# # 軸のラベルを設定
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# # 凡例を右下に表示
# plt.legend(loc='lower right')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# scikit-learnによる線形判別分析
# 次元数を指定して、LDAのインスタンスを生成
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# ロジスティック回帰を実行
# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# ロジスティック回帰をLDAを適用した訓練データで学習
lr = lr.fit(X_train_lda, y_train)
# # 決定境界を描画
# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# # 軸のラベルを設定
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# # 凡例を左下に表示
# plt.legend(loc='lower left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# テストデータでの結果を確認
# テストデータをLDAに適用
X_test_lda = lda.transform(X_test_std)
# # 決定境界を描画
# plot_decision_regions(X_test_lda, y_test, classifier=lr)
# # 軸のラベルを設定
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# # 凡例を左下に表示
# plt.legend(loc='lower left')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# カーネル主成分分析を実装
# ユークリッド距離の2乗の計算のため、scipyのspatialモジュールのdistanceクラスからpdist関数をインポート
# ペアごとの距離を正方行列に変換するため、scipyのspatialモジュールのdistanceクラスからsquareform関数をインポート
from scipy.spatial.distance import pdist, squareform
# 指数関数を計算するためにscipyからexpをインポート
from scipy import exp
# 中心化されたカーネル行列から固有値対を取得するため、scipyのlinalgモジュールからeigh関数をインポート
from scipy.linalg import eigh


def rbf_kernel_pca(X, gamma, n_components):
    """RBFカーネルPCAの実装
    パラメータ
    ------------
    X: {Numpy ndarray}, shape = [n_examples, n_features]

    gamma: float
        RBFカーネルのチューニングパラメータ

    n_components: int
        返される主成分の個数

    戻り値
    ------------
    alphas {NumPy ndarray}, shape = [n_examples, k_features]
        射影されたデータセット
    lambdas: list
        固有値
    """

    # M x N 次元のデータセットでペアごとのユークリッド距離の2乗を計算
    sq_dists = pdist(X, 'sqeuclidean')

    # ペアごとの距離を正方行列に変換
    mat_sq_dists = squareform(sq_dists)

    # 対称カーネル行列を計算
    K = exp(-gamma * mat_sq_dists)

    # カーネル行列を中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 中心化されたカーネル行列から固有値対を取得
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # 上位k個の固有ベクトル（射影されたデータ点）を収集
    alphas = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    # 対応する固有値を収集
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas


# # 半月型の分離を試行
# # 2つの半月形データを作成してプロット
# # 半月形データを作成するため、scikit-learnのdatasetsモジュールからmake_moonsをインポート
from sklearn.datasets import make_moons
# # データセットを作成
# X, y = make_moons(n_samples=100, random_state=123)
# # 作成したデータセットの散布図をプロット
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 標準のPCAを用いて主成分に投影したらどうなるか
# # PCAのインスタンスを生成
# scikit_pca = PCA(n_components=2)
# # PCAを学習しXに適用
# X_spca = scikit_pca.fit_transform(X)
# # グラフの数と配置、サイズを指定
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# # 1番目のグラフ領域に散布図をプロット
# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # 2番目のグラフ領域に散布図を見やすくなるよう若干ずらしてプロット
# ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
# # PCAの結果をプロットした軸のラベルを設定
# ax[0].set_xlabel('PC 1')
# ax[0].set_ylabel('PC 2')
# # 1次元の特徴量軸に射影したときのy軸の上限・下限を設定
# ax[1].set_ylim([-1, 1])
# # 1次元の特徴両軸に射影したときの目盛りを空に設定
# ax[1].set_yticks([])
# # 1次元の特徴両軸に射影したときのx軸のラベルを設定
# ax[1].set_xlabel('PC 1')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # カーネルPCA関数を試行
# # データ、チューニングパラメータ、次元数を指定してカーネル関数を実行
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# グラフの数と配置、サイズを指定
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# # 第1主成分と第2主成分の散布図を作成
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # 1次元の特徴量軸に射影したときの散布図を作成
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
# # カーネルPCAの結果をプロットした軸のラベルを設定
# ax[0].set_xlabel('PC 1')
# ax[0].set_ylabel('PC 2')
# # 1次元の特徴量軸に射影したときのy軸の上限・下限を設定
# ax[1].set_ylim([-1, 1])
# # 1次元の特徴両軸に射影したときの目盛りを空に設定
# ax[1].set_yticks([])
# # 1次元の特徴両軸に射影したときのx軸のラベルを設定
# ax[1].set_xlabel('PC 1')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 同心円用のデータセットを作成
# # 同心円用のデータを作成するため、scikit-learnのdatasetsモジュールからmake_circlesをインポート
from sklearn.datasets import make_circles
# # データセットを作成
# X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
# # 散布図の作成
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 標準のPCAの結果を確認
# # データをPCAで変換してからプロット
# # pcaインスタンスの作成
# scikit_pca = PCA(n_components=2)
# # XにPCAを適用
# X_spca = scikit_pca.fit_transform(X)
# # グラフの数と配置、サイズを指定
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# # 第1主成分と第2主成分の散布図を作成
# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # 1次元の特徴量軸に射影したときの散布図を作成
# ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
# # カーネルPCAの結果をプロットした軸のラベルを設定
# ax[0].set_xlabel('PC 1')
# ax[0].set_ylabel('PC 2')
# # 1次元の特徴量軸に射影したときのy軸の上限・下限を設定
# ax[1].set_ylim([-1, 1])
# # 1次元の特徴両軸に射影したときの目盛りを空に設定
# ax[1].set_yticks([])
# # 1次元の特徴両軸に射影したときのx軸のラベルを設定
# ax[1].set_xlabel('PC 1')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # カーネルPCAを試行
# # データをRBFカーネルPCAで変換してからプロット
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# # グラフの数と配置、サイズを指定
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# # 第1主成分と第2主成分の散布図を作成
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# # 1次元の特徴量軸に射影したときの散布図を作成
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
# # カーネルPCAの結果をプロットした軸のラベルを設定
# ax[0].set_xlabel('PC 1')
# ax[0].set_ylabel('PC 2')
# # 1次元の特徴量軸に射影したときのy軸の上限・下限を設定
# ax[1].set_ylim([-1, 1])
# # 1次元の特徴両軸に射影したときの目盛りを空に設定
# ax[1].set_yticks([])
# # 1次元の特徴両軸に射影したときのx軸のラベルを設定
# ax[1].set_xlabel('PC 1')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 新しい半月形データセットを作成し、RBFカーネルPCAの新しい実装を使って1次元の部分空間に射影
# # データセットを作成
# X, y = make_moons(n_samples=100, random_state=123)
# # RDFカーネルPCAを適用
# alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
# # テストデータに含まれるデータを試しに射影してみる
# x_new = X[25]
# print(x_new)
# # 元の射影を確認
# x_proj = alphas[25]
# print(x_proj)


# 新しいデータ点を射影する関数を作成
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


# x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
# print(x_reproj)

# # 最初の主成分への射影をプロット
# plt.scatter(alphas[y == 0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
# plt.scatter(alphas[y == 1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
# plt.scatter(x_proj, 0, color='black', label='Original projection of point X[25]', marker='^', s=100)
# plt.scatter(x_reproj, 0, color='green', label='Remapped point X[25]', marker='x', s=500)
# # 目盛りを削除
# plt.yticks([], [])
# # 凡例を右上に表示
# plt.legend(scatterpoints=1)
# # プロットを表示
# plt.tight_layout()
# plt.show()

# scikit-learnのカーネル主成分分析を試行
# カーネルPCAを行うため、scikit-learnのdecompositionモジュールからKernelPCAをインポート
from sklearn.decomposition import KernelPCA
# 半月形データセットを作成
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

# 最初の2つの主成分に変換された半月形データをプロット
plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
# 軸のラベルを設定
plt.xlabel('PC 1')
plt.ylabel('PC 2')
# プロットを表示
plt.tight_layout()
plt.show()
