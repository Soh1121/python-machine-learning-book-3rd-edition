from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


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
