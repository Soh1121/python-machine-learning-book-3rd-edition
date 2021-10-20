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

from sklearn.base import clone
# アンサンブルの個々の分類機のパラメータにアクセスできるようにするため、_name_estimatorsをインポートする
from sklearn.pipeline import _name_estimators


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
            self.classifiers_append(fitted_clf)
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
        else: # 'classlabel'での多数決

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
            maj_vote - self.lablenc_.inverse_transform(maj_vote)
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
