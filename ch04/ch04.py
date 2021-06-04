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
# 訓練データとテストデータに分割するためにscikit-learnのmodel_selectionモジュールからtrain_test_splitをインポート
from sklearn.model_selection import train_test_split
# 正規化を行うためにsklearnのpreprocesingモジュールからMinMaxScalerをインポート
from sklearn.preprocessing import MinMaxScaler
# 標準化を行うためにsklearnのpreprocessingモジュールからStandardScalerをインポート
from sklearn.preprocessing import StandardScaler
# L1正則化をロジスティック回帰で用いるためにscikit-learnのlinear_modelモジュールからLogisticRegressionをインポート
from sklearn.linear_model import LogisticRegression
# デフォルトの特徴量を評価する指標のためにscikit-learnのmetricsモジュールからaccuracy_scoreをインポートする
from sklearn.metrics import accuracy_score
# 推定器のモデルをディープコピーするためにscikit-learnのbaseモジュールからcloneをインポート
from sklearn.base import clone
# k最近傍法分類機を用いるためにscikit-learnのneighborsモジュールからKNeighborsClassifierをインポート
from sklearn.neighbors import KNeighborsClassifier
# 特徴量の重要度を評価するため、scikit-learnのensembleモジュールからRandomForestClassifierをインポート
from sklearn.ensemble import RandomForestClassifier
# 正則化パスをプロットするためにmatplotlibのpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt
# ndarayを扱うためにnumpyをnpとしてインポートする
import numpy as np
# 特徴量の組み合わせを作成するためにitertoolsからcombinationsをインポート
from itertools import combinations


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
# # クラスラベルを表示
# print('Class labels', np.unique(df_wine['Class label']))
# # wineデータセットの先頭５行を表示
# print(df_wine.head())

# scikit-learnのmodel_selectionサブモジュールを用いて訓練データとテストデータに分割
# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割（全体の30%をテストデータにする）
# オプション
# test_size：テストデータに割り当てるサンプル数の割合
# stratify：訓練データセットとテストデータセットのクラス比率をもとのデータセットと同じに成るように設定
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# scikit-learnを用いた正規化
mms = MinMaxScaler()
# 訓練データをスケーリング
X_train_norm = mms.fit_transform(X_train)
# print(X_train_norm)
# テストデータをスケーリング
X_test_norm = mms.fit_transform(X_test)
# print(X_test_norm)

# # 標準化と正規化を実際に計算
# ex = np.array([0, 1, 2, 3, 4, 5])
# print('standardized:', (ex - ex.mean()) / ex.std())
# print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

# scikit-learnを用いた標準化
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
# print(X_train_std)
X_test_std = stdsc.fit_transform(X_test)
# print(X_test_std)

# L1正則化ロジスティック回帰のインスタンスを生成
# LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')

# # L1正則化ロジスティック回帰のインスタンスを生成：逆正則化パラメータC=1.0はデフォルト値であり、
# # 値を大きくしたり小さくしたりすると、正則化の効果を強めたり弱めたりできる
# lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# # 訓練データに適合
# lr.fit(X_train_std, y_train)
# # 訓練データに対する正解率の表示
# # print('Training accuracy:', lr.score(X_train_std, y_train))

# # テストデータに対する正解率の表示
# # print('Test accuracy:', lr.score(X_test_std, y_test))

# # モデルの切片を確認
# # print(lr.intercept_)

# # モデルの重み係数の表示
# print(lr.coef_)

# # 描画の準備
# fig = plt.figure()
# ax = plt.subplot(111)
# # 各係数の色のリスト
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
#           'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
# # 空のリストを生成（重み係数、逆正則化パラメータ）
# weights, params = [], []
# # 逆正則化パラメータの値ごとに処理
# for c in np.arange(-4., 6.):
#     lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10**c)

# # 重み係数をNumPy配列に変換
# weights = np.array(weights)
# # 各重み係数をプロット
# for column, color in zip(range(weights.shape[1]), colors):
#     # 横軸を逆正則化パラメータ、縦軸を重み係数とした主線グラフ
#     plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

# # y=0に黒い波線を引く
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# # 横軸の範囲の設定
# plt.xlim([10**(-5), 10**5])
# # 軸のラベルを設定
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# # 横軸を対数スケールに設定
# plt.xscale('log')
# # 凡例を表示
# plt.legend()
# # グラフ外に凡例を移動
# ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
# # プロットの表示
# plt.tight_layout()
# plt.show()


# 逐次後退選択（SBS）アルゴリズムの実装
# class SBS():
#     """
#     逐次後退選択（sequential backward selection）を実行するクラス
#     """

#     def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
#         self.scoring = scoring              # 特徴量を評価する指標
#         self.estimator = clone(estimator)   # 推定器をディープコピー
#         self.k_features = k_features        # 選択する特徴量の数
#         self.test_size = test_size          # テストデータの割合
#         self.random_state = random_state    # 乱数シードを固定するrandom_state

#     def fit(self, X, y):
#         # 訓練データとテストデータに分割
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=self.test_size, random_state=self.random_state)
#         # すべての特徴量の個数、列インデックス
#         dim = X_train.shape[1]
#         self.indices_ = tuple(range(dim))
#         self.subsets_ = [self.indices_]
#         # すべての特徴量を用いてスコアを算出
#         score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
#         # スコアを格納
#         self.scores_ = [score]
#         # 特徴量が指定した個数に成るまで処理を繰り返す
#         while self.k_features < dim:
#             # 空のスコアリストを作成
#             scores = []
#             # 空の列インデックスリストを作成
#             subsets = []
#             # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
#             for p in combinations(self.indices_, r=dim - 1):
#                 # スコアを算出して格納
#                 score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
#                 scores.append(score)
#                 # 特徴量の部分集合を表す列インデックスのリストを格納
#                 subsets.append(p)

#             # 最良のスコアのインデックスを抽出
#             best = np.argmax(scores)
#             # 最良のスコアとなる列インデックスを抽出して格納
#             self.indices_ = subsets[best]
#             self.subsets_.append(self.indices_)
#             # 特徴量の個数を１つだけ減らして次のステップへ
#             dim -= 1
#             # スコアを格納
#             self.scores_.append(scores[best])

#         # 最後に格納したスコア
#         self.k_score_ = self.scores_[-1]
#         return self

#     def transform(self, X):
#         # 抽出した特徴量を返す
#         return X[:, self.indices_]

#     def _calc_score(self, X_train, y_train, X_test, y_test, indices):
#         # 指定された列番号indicesの特徴量を抽出してモデルを適合
#         self.estimator.fit(X_train[:, indices], y_train)
#         # テストデータを用いてクラスラベルを予測
#         y_pred = self.estimator.predict(X_test[:, indices])
#         # 真のラベルと予測値を用いてスコアを算出
#         score = self.scoring(y_test, y_pred)
#         return score


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# # k最近傍法分類器のインスタンスを生成（近傍点数=5）
# knn = KNeighborsClassifier(n_neighbors=5)
# # 逐次後退選択のインスタンスを生成（特徴量の個数が1になるまで特徴量を選択）
# sbs = SBS(knn, k_features=1)
# # 逐次後退選択を実行
# sbs.fit(X_train_std, y_train)

# # # 特徴量の個数のリスト
# # k_feat = [len(k) for k in sbs.subsets_]
# # # 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
# # plt.plot(k_feat, sbs.scores_, marker='o')
# # # y軸の上限下限を設定
# # plt.ylim([0.7, 1.02])
# # # 各軸のラベルを設定
# # plt.ylabel('Accuracy')
# # plt.xlabel('Number of features')
# # # グリッドを表示
# # plt.grid()
# # # プロットを表示
# # plt.tight_layout()
# # plt.show()

# # 13個の特徴量から順次減らしているため、k=3のときの特徴量はリストのindex=10に保存されている
# k3 = list(sbs.subsets_[10])
# # 3個の特徴量が何か表示
# # print(df_wine.columns[1:][k3])

# # 元のテストデータセットでKNN分類器の性能を確認
# # 13個のすべての特徴量を用いてモデルを適合
# knn.fit(X_train_std, y_train)
# # 訓練の正解率を出力
# print('Train accuracy:', knn.score(X_train_std, y_train))
# # テストの正解率を出力
# print('Test accuracy:', knn.score(X_test_std, y_test))

# # 3つの特徴量を用いてモデルを適合
# knn.fit(X_train_std[:, k3], y_train)
# # 訓練の正解率を出力
# print('Train accuracy:', knn.score(X_train_std[:, k3], y_train))
# # テストの正解率を出力
# print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))

# ランダムフォレストで特徴量の重要度を評価する
# Wineデータセットの特徴量の名称
feat_labels = df_wine.columns[1:]
# ランダムフォレストオブジェクトの生成（決定木の個数=500）
forest = RandomForestClassifier(n_estimators=500, random_state=1)
# モデルを適合
forest.fit(X_train, y_train)
# 特徴量の重要度を抽出
importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]
# 重要度の降順で特徴量の名称、重要度を表示
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
# プロットのタイトルを表示
plt.title('Feature Importances')
# 特徴量の重要度を降順で棒グラフ作成
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
# X軸のラベルを設定
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
# x軸の上限、下限を設定
plt.xlim([-1, X_train.shape[1]])
# プロットを表示
plt.tight_layout()
plt.show()
