# DataFrameオブジェクトを用いるためpandasをpdとしてインポートする
import pandas as pd
# CSVフォーマットのデータを読み込むためにioモジュールからStringIOをインポートする
from io import StringIO


# 表形式のデータで欠損値を見てみる
# サンプルデータを作成
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''
# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
# print(df)

# 各特徴量の欠測値をカウント
# print(df.isnull().sum())

# 欠測値を含む行を削除
# print(df.dropna())

# 欠測値を含む列を削除
# print(df.dropna(axis=1))

# すべての列がNaNである行だけを削除
# print(df.dropna(how='all'))
# 非NaNの値が4つ未満の行を削除
print(df.dropna(thresh=4))
# 特定の列にNaNが含まれている行だけ削除
print(df.dropna(subset=['C']))
