# 小数点以下の切り上げを行うためにmathモジュールをインポート
import math
# 組み合わせを用いるため、scipyのspecialモジュールからcombをインポート
from scipy.special import comb
# ndarrayを生成するためにnumpyをnpとしてインポート
import numpy as np
# プロットを作成するためmatplotlibのpyplotモジュールをpltとしてインポート
import matplotlib.pyplot as plt


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error** k *
             (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)


# アンサンブルエラーを算出する
print(ensemble_error(n_classifier=11, error=0.25))

# 誤分類率を0.0以上1.0以下で変化させてアンサンブルの誤分類率を計算し、アンサンブルとベース分類器の誤分類の関係をグラフ化
# ベース分類器の誤分類率を設定
error_range = np.arange(0.0, 1.01, 0.01)
# アンサンブルの誤分類率を算出
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]
# アンサンブルの誤分類率をプロット
plt.plot(error_range, ens_errors,
         label='Ensemble error', linewidth=2)
# ベース分類器の誤分類率をプロット
plt.plot(error_range, error_range,
         linestyle='--', label='Base error', linewidth=2)
# 軸のラベルを設定
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
# 凡例を左上に表示
plt.legend(loc='upper left')
# 目盛線を表示
plt.grid(alpha=0.5)
# プロットを表示
plt.show()
