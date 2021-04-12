from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# print("Class labels:", np.unique(y))

# 全体の30%がテストデータとなるよう訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# データセットのクラスラベルを確認
print("Label counts in y: ", np.bincount(y))

# 訓練データセットのクラスラベルを確認
print("Label counts in y_train: ", np.bincount(y_train))

# テストデータセットのクラスラベルを確認
print("Label counts in y_test", np.bincount(y_test))
