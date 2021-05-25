# DataFrameオブジェクトを用いるためpandasをpdとしてインポートする
import pandas as pd
# CSVフォーマットのデータを読み込むためにioモジュールからStringIOをインポートする
from io import StringIO
# 欠測値を補完するためにscikit-learnのimputeモジュールからSimpleImputerクラスをインポートする
from sklearn.impute import SimpleImputer
# クラスラベルを整数値に変換するためにscikit-learnのpreprocessingモジュールからLabelEncoderクラスをインポートする
from sklearn.preprocessing import LabelEncoder
# 名義特徴量にダミー特徴量を割り振るためにscikit-learnのpreprocessingモジュールからOneHotEncoderをインポートする
from sklearn.preprocessing import OneHotEncoder
# 複数の特徴量からなる配列の列を選択的に変換したいときのために、scikit-learnのcomposeモジュールからColumnTransformerをインポートする
from sklearn.compose import ColumnTransformer
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

# # カテゴリデータのエンコーディング
# # サンプルデータを生成
# df = pd.DataFrame([
#     ['green', 'M', 10.1, 'class2'],
#     ['red', 'L', 13.5, 'class1'],
#     ['blue', 'XL', 15.3, 'class2']
# ])
# # 列名を設定
# df.columns = ['color', 'size', 'price', 'classlabel']
# # print(df)

# # 順序特徴量を生成
# # Tシャツのサイズと整数を対応させるディクショナリを生成
# size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# # Tシャツのサイズを整数に変換
# df['size'] = df['size'].map(size_mapping)
# print(df)
# # 整数値を文字列表現に戻す
# inv_size_mapping = {v: k for k, v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)

# # クラスラベルをエンコーディング
# class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# # print(class_mapping)
# # クラスラベルを整数に変換
# df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)
# # 整数とクラスラベルを対応させるディクショナリを生成
# inv_class_mapping = {v: k for k, v in class_mapping.items()}
# # 整数からクラスラベルに変換
# df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)
# # scikit-learnのLabelEncoderクラスを利用する場合
# class_le = LabelEncoder()
# # クラスラベルから整数に変換
# y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))

# # color列についても文字列を整数値に変換
# # Tシャツの色、サイズ、価格を抽出
# X = df[['color', 'size', 'price']].values
# # LabelEncoderを用いてクラスラベルから整数に変換
# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

# # one-hotエンコーディングの実装
# # Tシャツの色、サイズ、価格を抽出
# X = df[['color', 'size', 'price']].values
# # one-hotエンコーダの生成
# color_ohe = OneHotEncoder()
# # one-hotエンコーディングを実施
# print(X[:, 0])
# print(X[:, 0].reshape(-1, 1))
# print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# # 複数の特徴量からなる配列の列を選択的に変換
# # Tシャツの色、サイズ、価格を抽出
# X = df[['color', 'size', 'price']].values
# # columnTransformerの生成
# c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]), ('nothing', 'passthrough', [1, 2])])
# # 複数の特徴量からなる配列の列を選択的にonehotで変換
# print(c_transf.fit_transform(X).astype(float))

# # one-hotエンコーディングを実行
# print(pd.get_dummies(df[['price', 'color', 'size']]))

# # 1列目を削除したone-hotエンコーディング
# print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

# # OneHotEncoderを用いて冗長な列を削除
# color_ohe = OneHotEncoder(categories='auto', drop='first')
# c_transf = ColumnTransformer([('onehot', color_ohe, [0]), ('nothing', 'passthrough', [1, 2])])
# print(c_transf.fit_transform(X).astype(float))

# pandasライブラリを使ってUGI Machine Learning RepositoryからWineデータセットを直接読み込む
# wineデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# 列名を指定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline']
# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))
# wineデータセットの先頭５行を表示
print(df_wine.head())
