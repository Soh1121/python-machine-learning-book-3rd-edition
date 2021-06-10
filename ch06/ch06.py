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

# 連結する処理としてスケーリング、主成分分析、ロジスティック回帰を指定
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))
# パイプラインで学習を実行
pipe_lr.fit(X_train, y_train)
# パイプラインで推定を実行
pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
