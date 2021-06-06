# DataFrameやサンプルデータを用いるためpandasをpdとしてインポートする
import pandas as pd
# 訓練データセットとテストデータセットに分割するためscikit-learnのmodel_selectionモジュールからtrain_test_splitをインポートする
from sklearn.model_selection import train_test_split
# 標準化を行うためにscikit-learnのpreprocessingモジュールからStandardScalerをインポート
from sklearn.preprocessing import StandardScaler
# 共分散行列の固有対を取得するためにnumpyをnpとしてインポート
import numpy as np
# グラフをプロットするためにmatplotlibからpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt


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
X_train_pca = X_train_std.dot(w)

# 変換後の訓練データセットをプロット
# 色とマーカーの設定
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
# 「クラスラベル」「点の色」「点の種類」の組み合わせからなるリストを生成してプロット
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
# 軸のラベルを設定
plt.xlabel('PC 1')
plt.ylabel('PC 2')
# 凡例を左下に表示
plt.legend(loc='lower left')
# プロットを表示
plt.tight_layout()
plt.show()
