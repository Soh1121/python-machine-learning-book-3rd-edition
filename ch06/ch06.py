# データセットを直接読み込んだりするためにpandasをpdとしてインポート
import pandas as pd
# ラベルを変換するためにscikit-learnのpreprocessingモジュールからLabelEncoderをインポート
from sklearn.preprocessing import LabelEncoder
# データセットを訓練データセットとテストデータセットに分割するためにsklearnのmodel_selectionモジュールからtrain_test_split関数をインポート
from sklearn.model_selection import train_test_split
# パイプラインを構築するため、scikit-learnのpipelineモジュールからmake_piplineをインポート
from sklearn.pipeline import make_pipeline
# 標準化を行うためにscikit-learnのpreprocessingモジュールからStandardScalerをインポート
from sklearn.preprocessing import StandardScaler
# 主成分分析を行うため、scikit-learnのdecompositionモジュールからPCAをインポート
from sklearn.decomposition import PCA
# ロジスティック回帰を用いるため、scikit-learnのlinear_modelモジュールからLogisticRegressionをインポート
from sklearn.linear_model import LogisticRegression
# 層化k分割交差検証を行うため、scikit-learnのmodel_selectionモジュールからStratifiedKFoldをインポート
from sklearn.model_selection import StratifiedKFold
# 交差検証の正解率を算出するため、scikit-learnのmodel_selectionモジュールからcross_val_scoreをインポート
from sklearn.model_selection import cross_val_score
# 交差検証を実行するため、scikit-learnのmodel_selectionモジュールからlearning_curve関数をインポート
from sklearn.model_selection import learning_curve
# 要素数をカウントするためなどにnumpyをnpとしてインポート
import numpy as np
# プロットを作成するためにmatplotlibのpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt


# UCIのWebサイトからpandasライブラリのread_csv関数を使ってデータセットを直接読み込む
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)

# 30個の特徴量をXに割り当てる
X = df.loc[:, 2:].values
# 30個のラベルをyに割り当てる
y = df.loc[:, 1].values
# LabelEncoderインスタンスを生成
le = LabelEncoder()
# yを変換する
y = le.fit_transform(y)
# print(le.classes_)

# # 悪性腫瘍をクラス1、良性腫瘍をクラス0で表す
# print(le.transform(['M', 'B']))

# データセットを訓練データセットとテストデータセットに分割する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# # 連結する処理としてスケーリング、主成分分析、ロジスティック回帰を指定
# pipe_lr = make_pipeline(StandardScaler(),
#                         PCA(n_components=2),
#                         LogisticRegression(random_state=1, solver='lbfgs'))
# # パイプラインで学習を実行
# pipe_lr.fit(X_train, y_train)
# # パイプラインで推定を実行
# pipe_lr.predict(X_test)
# # print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# # # 層化k分割交差検証を試行
# # # 分割元データ、分割数、乱数生成器の状態を指定し、層化k分割交差検証イテレータを表すStratifriedKFoldクラスのインスタンス化
# # kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
# # scores = []
# # # イテレータのインデックスと要素をループ処理：
# # for k, (train, test) in enumerate(kfold):
# #     # データをモデルに適合
# #     pipe_lr.fit(X_train[train], y_train[train])
# #     # テストデータの正解率を算出
# #     score = pipe_lr.score(X_train[test], y_train[test])
# #     # リストに正解率を追加
# #     scores.append(score)
# #     # 分割の番号、0以上の要素数、正解率を出力
# #     print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
# # # 正解率の平均と標準偏差を出力
# # print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# # scikit-learnを用いて層化k分割交差検証を実行
# # 推定器estimator、訓練データX、予測値y、分割数cv、CPU数n_jobsを指定
# scores = cross_val_score(estimator=pipe_lr,
#                          X=X_train, y=y_train,
#                          cv=10, n_jobs=1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# scikit-learnの学習曲線関数を使ってモデルを評価
# 標準化をしてロジスティック回帰を行うパイプラインを作成
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                        solver='lbfgs', max_iter=10000))
# learning_curve関数で交差検証による正解率を算出
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10, n_jobs=1)
# 訓練データとテトスデータの平均と標準偏差を算出
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 訓練データにおける正解率をプロット
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
# fill_between関数で平均±標準偏差の幅を塗りつぶす
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.5, color='blue')
# テストデータにおける正解率をプロット
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# fill_between関数で平均±標準偏差の幅を塗りつぶす
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.5, color='green')
# 目盛りを表示
plt.grid()
# 軸のラベルを設定
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
# 凡例を右下に表示
plt.legend(loc='lower right')
# y軸の上限、下限を設定
plt.ylim([0.8, 1.03])
# プロットを表示
plt.tight_layout()
plt.show()
