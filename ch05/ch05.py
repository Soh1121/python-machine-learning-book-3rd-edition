# DataFrameやサンプルデータを用いるためpandasをpdとしてインポートする
import pandas as pd
# 訓練データセットとテストデータセットに分割するためscikit-learnのmodel_selectionモジュールからtrain_test_splitをインポートする
from sklearn.model_selection import train_test_split
# 標準化を行うためにscikit-learnのpreprocessingモジュールからStandardScalerをインポート
from sklearn.preprocessing import StandardScaler

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
