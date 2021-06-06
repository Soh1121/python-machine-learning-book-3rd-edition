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

# 分散説明率の累積和を確認する
# 固有値を計算
tot = sum(eigen_vals)
# 分散説明率を計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を取得
cum_var_exp = np.cumsum(var_exp)
# 分散説明率の棒グラフを作成
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
# 分散説明率の累積和の階段グラフを作成
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumlative explained variance')
# ラベル名を設定
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
# 凡例を良きところに表示
plt.legend(loc='best')
# プロットを表示
plt.tight_layout()
plt.show()
