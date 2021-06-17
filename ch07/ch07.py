# 小数点以下の切り上げを行うためにmathモジュールをインポート
import math
# 組み合わせを用いるため、scipyのspecialモジュールからcombをインポート
from scipy.special import comb


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error** k *
             (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)


# アンサンブルエラーを算出する
print(ensemble_error(n_classifier=11, error=0.25))
