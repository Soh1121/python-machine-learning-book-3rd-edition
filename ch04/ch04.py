# DataFrameオブジェクトを用いるためpandasをpdとしてインポートする
import pandas as pd
# CSVフォーマットのデータを読み込むためにioモジュールからStringIOをインポートする
from io import StringIO
# 欠測値を補完するためにscikit-learnのimputeモジュールからSimpleImputerクラスをインポートする
from sklearn.impute import SimpleImputer
# クラスラベルを整数値に変換するためにscikit-learnのpreprocessingモジュールからLabelEncoderクラスをインポートする
from sklearn.preprocessing import LabelEncoder
# ndarayを扱うためにnumpyをnpとしてインポートする
import numpy as np


# # 表形式のデータで欠損値を見てみる
# # サンプルデータを作成
# csv_data = '''A,B,C,D
#               1.0,2.0,3.0,4.0
#               5.0,6.0,,8.0
#               10.0,11.0,12.0,'''
# # サンプルデータを読み込む
# df = pd.read_csv(StringIO(csv_data))
# # print(df)

# 各特徴量の欠測値をカウント
# print(df.isnull().sum())

# 欠測値を含む行を削除
# print(df.dropna())

# 欠測値を含む列を削除
# print(df.dropna(axis=1))

# すべての列がNaNである行だけを削除
# print(df.dropna(how='all'))
# 非NaNの値が4つ未満の行を削除
# print(df.dropna(thresh=4))
# 特定の列にNaNが含まれている行だけ削除
# print(df.dropna(subset=['C']))

# # 欠測値をscikit-learnを用いて補完する
# # 欠測値補完のインスタンスを生成（平均値補完）
# imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# # データを適合
# imr.fit(df.values)
# # 補完を実行
# imputed_data = imr.transform(df.values)
# print(imputed_data)

# pandasを使った平均値補完
# print(df.fillna(df.mean()))

# カテゴリデータのエンコーディング
# サンプルデータを生成
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

# # 順序特徴量を生成
# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
# print(df)
# # 整数値を文字列表現に戻す
# inv_size_mapping = {v: k for k, v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)

# クラスラベルをエンコーディング
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
# 整数と暮らすラベルを対応させるディクショナリを生成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
# scikit-learnのLabelEncoderクラスを利用する場合
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))
