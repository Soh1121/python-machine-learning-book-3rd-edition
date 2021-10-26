# 小数点以下の切り上げを行うためにmathモジュールをインポート
import math
# 組み合わせを用いるため、scipyのspecialモジュールからcombをインポート
from scipy.special import comb
# ndarrayを生成するためにnumpyをnpとしてインポート
import numpy as np
# プロットを作成するためmatplotlibのpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt
# 基本的な機能をただで手に入れるため、親クラスBaseEstimatorとClassifierMixinをインポート
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
# クラスラベルが0から始まるようにエンコードするため、LabelEncoderをインポートする
from sklearn.preprocessing import LabelEncoder
# アンサンブルの個々の分類機のパラメータにアクセスできるようにするため、_name_estimatorsをインポートする
from sklearn.pipeline import _name_estimators
# 試しに機械学習を実行するため、sklearnからdatasetsをインポートする
from sklearn import datasets
# テストデータセットに分割するためにsklearnのmodel_selectionクラスからtrain_test_splitメソッドをインポートする
from sklearn.model_selection import train_test_split
# ロジスティック回帰を用いるため、sklearnのlinear_modelクラスからLogisticRegressionをインポートする
from sklearn.linear_model import LogisticRegression
# 決定木分類器を用いるため、sklearnのtreeクラスからDecisionTreeClassifierをインポートする
from sklearn.tree import DecisionTreeClassifier
# k最近傍法分類器を用いるため、sklearnのneighborsクラスからKNeighborsClassifierをインポートする
from sklearn.neighbors import KNeighborsClassifier
# パイプラインを構築するためsklearnのpipelineクラスからPipelineをインポートする
from sklearn.pipeline import Pipeline
# 標準化を用いるためsklearnのpreprocessingからStandardScalerをインポートする
from sklearn.preprocessing import StandardScaler
# 分割交差検証を使うため、sklearnのmodel_selectionクラスから、cross_val_scoreをインポートする
from sklearn.model_selection import cross_val_score
# ROC曲線を計算するために、sklearnのmetricsクラスから、roc_curveメソッドをインポートする
from sklearn.metrics import roc_curve
# 正解率を計算するために、sklearnのmetricsクラスから、aucメソッドをインポートする
from sklearn.metrics import auc
# 0, 1の組み合わせを生成するためにitertoolsからproductメソッドをインポートする
from itertools import product

from sklearn.base import clone


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error ** k *
             (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)


# 多数決アンサンブル分類機の実装
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ 多数決アンサンブル分類器

    パラメータ
    -------------
    classifiers : array-like, shape = [n_classifiers]
        アンサンブルのさまざまな分類器

    vote : str, {'classlabel', 'probability'} (default: 'classlabel')
        'probability'の場合、クラスラベルの予測はクラスの所属確率の
        argmaxに基づく（分類機が調整済みであることが推奨される）

    weights : array-like, shape = [n_classifiers] (optional, default=None)
        'int'または'float'型の値のリストが提供された場合、分類機は重要度で重み付けされる
        'weight=None'の場合は均一な重みを使用
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key,
                                  value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ 分類機を学習させる

        パラメータ
        ------------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            訓練データからなる行列

        y : array-likek, shape = [n_examples]
            クラスラベルのベクトル

        戻り値
        ------------
        self : object
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability"
                             "or 'classlabel'; got (vote=%r)" % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError("Number of classifiers and weights must be equal; "
                             "got %d weights, %d classifiers"
                             % (len(self.weights), len(self.classifiers)))
        # LabelEncoder を使ってクラスラベルが0から始まるようにエンコードする
        # self.predictのnp.argmax呼び出しで重要となる
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Xのクラスラベルを予測する

        パラメータ
        ------------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            訓練データからなる行列

        戻り値
        ------------
        maj_vote : array-like, shape = [n_examples]
            予測されたクラスラベル

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel'での多数決
            # clf.predict呼び出しの結果を収集
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            # 各データ点のクラス確率に重みを掛けて足し合わせた値が最大となる
            # 列番号を配列として返す
            maj_vote = np.apply_along_axis(
                lambda x:
                np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions)

            # 各データ点に確率の最大値を与えるクラスラベルを抽出
            maj_vote = self.labelenc_.inverse_transform(maj_vote)
            return maj_vote

    def predict_proba(self, X):
        """ Xのクラス確率を予測する

        パラメータ
        ------------
        X : {array-like, sparse matrix}, shape = [n_examples, n_classes]
            訓練ベクトル：n_examplesはデータ点の個数、n_featuresは特徴量の個数

        戻り値
        ------------
        avg_proba : array-like, shape = [n_examples, n_classes]
            各データ点に対する各クラスで重み付けた平均確率

        """
        probas = np.asarray([clf.predict_proba(X)
                            for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ GridSearchの実行時に分類器のパラメータ名を取得 """
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            # キーを"分類器の名前__パラメータ名 "、
            # 値をパラメータの値とするディクショナリを生成
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
                return out

# # アンサンブルエラーを算出する
# print(ensemble_error(n_classifier=11, error=0.25))

# # 誤分類率を0.0以上1.0以下で変化させてアンサンブルの誤分類率を計算し、アンサンブルとベース分類器の誤分類の関係をグラフ化
# # ベース分類器の誤分類率を設定
# error_range = np.arange(0.0, 1.01, 0.01)
# # アンサンブルの誤分類率を算出
# ens_errors = [ensemble_error(n_classifier=11, error=error)
#               for error in error_range]
# # アンサンブルの誤分類率をプロット
# plt.plot(error_range, ens_errors,
#          label='Ensemble error', linewidth=2)
# # ベース分類器の誤分類率をプロット
# plt.plot(error_range, error_range,
#          linestyle='--', label='Base error', linewidth=2)
# # 軸のラベルを設定
# plt.xlabel('Base error')
# plt.ylabel('Base/Ensemble error')
# # 凡例を左上に表示
# plt.legend(loc='upper left')
# # 目盛線を表示
# plt.grid(alpha=0.5)
# # プロットを表示
# plt.show()


iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
clf1 = LogisticRegression(penalty='l2', C=0.001,
                          solver='lbfgs',
                          random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')
# for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
#     scores = cross_val_score(estimator=clf,
#                              X=X_train,
#                              y=y_train,
#                              cv=10,
#                              scoring='roc_auc')
#     print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# colors = ['black', 'orange', 'blue', 'green']
# linestyles = [':', '--', '-.', '-']
# for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
#     # 陽性クラスのラベルは1であることが前提
#     y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
#     roc_auc = auc(x=fpr, y=tpr)
#     plt.plot(fpr, tpr,
#              color=clr,
#              linestyle=ls,
#              label='%s (auc = %0.2f)' % (label, roc_auc))

# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1],
#          linestyle='--',
#          color='gray',
#          linewidth=2)
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.grid(alpha=0.5)
# plt.xlabel('False positive rate (FPR)')
# plt.ylabel('True positive rate (TPR)')
# plt.show()

# 標準化を行うため、Scalerを用意
sc = StandardScaler()
# 学習データに標準化を適用
X_train_std = sc.fit_transform(X_train)
# 決定領域を描画する最小値、最大値を生成
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
# グリッドポイントを生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 描画領域を2行2列に分割
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))
# 決定領域のプロット、青や赤の散布図の作成などを実行
# 変数idxは各分類器を描画する行と列の位置を表すタプル
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                  X_train_std[y_train==0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                  X_train_std[y_train==1, 1],
                                  c='green',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-.35, -5.,
         s='Sepal width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5,
         s='Petal length [standardized]',
         ha='center', va='center',
         fontsize=12, rotation=90)
plt.show()
