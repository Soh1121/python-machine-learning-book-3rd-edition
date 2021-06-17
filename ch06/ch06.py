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
# 学習曲線を作成するため、scikit-learnのmodel_selectionモジュールからlearning_curve関数をインポート
from sklearn.model_selection import learning_curve
# 検証曲線を作成するため、scikit-learnのmodel_selectionモジュールからvalidation_curve関数をインポート
from sklearn.model_selection import validation_curve
# SVMを用いるためにscikit-learnのsvmモジュールからSVCをインポート
from sklearn.svm import SVC
# グリッドサーチを行うため、scikit-learnのmodel_selectionモジュールからGridSearchCVをインポート
from sklearn.model_selection import GridSearchCV
# 決定木を用いるためにscikit-learnのtreeモジュールからDecisionTreeClassifierをインポート
from sklearn.tree import DecisionTreeClassifier
# 混同行列を生成するためにscikit-learnのmetricsモジュールからconfusion_matrixをインポート
from sklearn.metrics import confusion_matrix
# 適合率、再現率、F値を算出するために、scikit-learnのmetricsモジュールからprecision_scoreとrecall_score、f1_scoreをインポート
from sklearn.metrics import precision_score, recall_score, f1_score
# グリッドサーチでカスタムスコアを利用するためにscikit-learnのmetricsモジュールからmake_scorerをインポート
from sklearn.metrics import make_scorer
# ROC曲線を算出し、曲線下面積を算出するためにscikit-learnのmetricsモジュールからroc_curve, aucをインポート
from sklearn.metrics import roc_curve, auc
# データをリサンプリングするためにscikit-learnのutilsモジュールからresampleをインポート
from sklearn.utils import resample
# 要素数をカウントするためなどにnumpyをnpとしてインポート
import numpy as np
# プロットを作成するためにmatplotlibのpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt
# 線形補間を行うためにscipyからinterp関数をインポート
from scipy import interp


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

# # scikit-learnの学習曲線関数を使ってモデルを評価
# # 標準化をしてロジスティック回帰を行うパイプラインを作成
# pipe_lr = make_pipeline(StandardScaler(),
                        # LogisticRegression(penalty='l2', random_state=1,
                        # solver='lbfgs', max_iter=10000))
# # learning_curve関数で交差検証による正解率を算出
# train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),
#                                                         cv=10, n_jobs=1)
# # 訓練データとテトスデータの平均と標準偏差を算出
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# # 訓練データにおける正解率をプロット
# plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
# # fill_between関数で平均±標準偏差の幅を塗りつぶす
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.5, color='blue')
# # テストデータにおける正解率をプロット
# plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# # fill_between関数で平均±標準偏差の幅を塗りつぶす
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.5, color='green')
# # 目盛りを表示
# plt.grid()
# # 軸のラベルを設定
# plt.xlabel('Number of training examples')
# plt.ylabel('Accuracy')
# # 凡例を右下に表示
# plt.legend(loc='lower right')
# # y軸の上限、下限を設定
# plt.ylim([0.8, 1.03])
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 検証曲線を使ってモデルを評価
# # validation_curve関数によるモデルのパラメータを変化させ、交差検証による正解率を算出
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# train_scores, test_scores = validation_curve(estimator=pipe_lr,
#                                              X=X_train, y=y_train,
#                                              param_name='logisticregression__C',
#                                              param_range=param_range, cv=10)
# # 平均と標準偏差を算出
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# # 訓練データにおける検証曲線をプロット
# plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
# # fill_between関数で平均±標準偏差を塗りつぶす
# plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.5, color='blue')
# # テストデータにおける検証曲線をプロット
# plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# # fill_between関数で平均±標準偏差を塗りつぶす
# plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.5, color='green')
# # 目盛りを表示
# plt.grid()
# # X軸を対数スケールに
# plt.xscale('log')
# # 凡例を右下をに表示
# plt.legend(loc='lower right')
# # 軸のラベルを表示
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# # y軸の上限・下限を設定
# plt.ylim([0.8, 1.0])
# # プロットを表示
# plt.tight_layout()
# plt.show()

# SVMのパイプライン訓練とチューニング
# パイプラインを作成
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
# パラメータの幅を設定
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# グリッドサーチを行うパラメータを設定
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
# # ハイパーパラメータ値のリストparam_gridaを指定し、グリッドサーチを行うGridSearchCVクラスをインスタンス化
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring='accuracy', cv=10, refit=True, n_jobs=1)
# # 訓練を実行
# gs.fit(X_train, y_train)
# # # モデルの最良スコアを出力
# # print(gs.best_score_)
# # # 最良スコアとなるパラメータ値を出力
# # print(gs.best_params_)

# # テストデータセットを用いて、選択されたモデルの性能を評価
# # ベストなモデルの取得
# clf = gs.best_estimator_
# # 最良モデルでSVMを実行
# # clf.fit(X_train, y_train)
# # スコアを算出
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# 入れ子式交差検証を実装
# グリッドサーチでパラメータチューニング
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring='accuracy', cv=2)
# 交差検証でスコアを算出
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# # 決定木で深さパラメータのみチューニング
# # ハイパーパラメータ値として決定木の深さパラメータを指定し、
# # グリッドサーチを行うGridSearchCVクラスをインスタンス化
# gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
#                   param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
#                   scoring='accuracy', cv=2)
# # 性能を確認
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# 混同行列を算出
# SVMをパイプラインで学習
pipe_svc.fit(X_train, y_train)
# 予測値を算出
y_pred = pipe_svc.predict(X_test)
# テストと予測のデータから混同行列を生成
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# # print(confmat)

# # 混同行列を図示
# # 図のサイズを指定
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# # matshow関数で行列からヒートマップを作成
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):       # クラス0の繰り返し処理
#     for j in range(confmat.shape[1]):   # クラス1の繰り返し処理
#         ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')    # 件数を表示
# # 軸のラベルを設定
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # 適合率、再現率、F1スコアを出力
# # 適合率を出力
# print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
# # 再現率を出力
# print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
# # F1スコアを出力
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# # GridSearchCVで正解率以外の指標も使える
# # グリッドサーチの値を設定
# c_gamma_range = [0.01, 0.1, 1.0, 10.0]
# param_grid = [{'svc__C': c_gamma_range, 'svc__kernel': ['linear']},
#               {'svc__C': c_gamma_range, 'svc__gamma': c_gamma_range,
#                'svc__kernel': ['rbf']}]
# # カスタムの性能指標を設定
# scorer = make_scorer(f1_score, pos_label=0)
# # グリッドサーチのオブジェクトを生成
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring=scorer,
#                   cv=10, n_jobs=-1)
# # グリッドサーチを実施
# gs.fit(X_train, y_train)
# # 結果を出力
# print(gs.best_score_)
# # パラメータを出力
# print(gs.best_params_)

# # ROC曲線を確認
# # スケーリング、主成分分析、ロジスティック回帰を指定して、Pipelineクラスをインスタンス化
# pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
#                         LogisticRegression(penalty='l2', random_state=1,
#                                            solver='lbfgs', C=100.0))
# # 2つの特徴量を抽出
# X_train2 = X_train[:, [4, 14]]
# # 層化k分割交差検証イテレータを表すStrratifiedKFoldクラスをインスタンス化
# cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
# # 作画領域を指定
# fig = plt.figure(figsize=(7, 5))
# # TPRの平均値を初期化
# mean_tpr = 0.0
# # 0から1までの間で100個の要素を生成
# mean_fpr = np.linspace(0, 1, 100)
# # すべてのTPRを記憶するリストを初期化
# all_tpr = []
# for i, (train, test) in enumerate(cv):
#     # predict_probaメソッドで確率を予測、fitメソッドでモデルに適合させる
#     probas = pipe_lr.fit(X_train2[train],
#                          y_train[train]).predict_proba(X_train2[test])
#     # roc_curve関数でROC曲線の性能を計算してプロット
#     fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
#     mean_tpr += interp(mean_fpr, fpr, tpr)  # FPR（X軸）とTPR（Y軸）を線形補間
#     mean_tpr[0] = 0.0
#     roc_auc = auc(fpr, tpr)                 # 曲線下面積（AUC）を計算
#     plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i + 1, roc_auc))

# # 当て推量をプロット
# plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
# # FPR、TPR、ROC AUCそれぞれの平均を計算してプロット
# mean_tpr /= len(cv)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
# # 完全に予測が正解したときのROC曲線をプロット
# plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance')
# # 軸の上限・下限を設定
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# # 軸のラベルを設定
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# # 凡例を右下に設定
# plt.legend(loc="lower right")
# # プロットを表示
# plt.tight_layout()
# plt.show()

# # マイクロ平均を性能指標に指定
# pre_scorer = make_scorer(score_func=precision_score,
#                          pos_label=1,
#                          greater_is_better=True,
#                          average='micro')

# クラスの不均衡に対する処理を検証
# 不均衡なデータセットを作成
# 良性腫瘍のすべてと悪性腫瘍の最初の40個をデータとして準備
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

# すべて0で予測した場合
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

# # 少数派クラスのアップサンプリング
# # 現在の少数派クラスのサンプル数を確認
# print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])

# データ点の個数がクラス0とおなじになるまで新しいデータ点を復元抽出
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
# print('Number of class 1 examples after:', X_upsampled.shape[0])

# アップサンプリングしたデータを結合
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

# すべて0で予測した場合
y_pred = np.zeros(y_bal.shape[0])
print(np.mean(y_pred == y_bal) * 100)
