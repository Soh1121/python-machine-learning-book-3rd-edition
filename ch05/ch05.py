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

# クラスの判別情報を計測
# 固有値の実数部の総和を求める
tot = sum(eigen_vals.real)
# 分散説明率とその累積和を計算
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
# 棒グラフと階段グラフを描画
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='Individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative "discriminability"')
# 軸のラベルを設定
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
# y軸の上限、下限を設定
plt.ylim([-0.1, 1.1])
# 凡例を適切な位置に表示
plt.legend(loc='best')
# プロットを表示
plt.tight_layout()
plt.show()
