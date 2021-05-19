# DataFrameオブジェクトを用いるためpandasをpdとしてインポートする
import pandas as pd
# CSVフォーマットのデータを読み込むためにioモジュールからStringIOをインポートする
from io import StringIO


# 表形式のデータで欠損値を見てみる
# サンプルデータを作成
csv_data = '''A, B, C, D
              1.0, 2.0, 3.0, 4.0
              5.0, 6.0, , 8.0
              10.0, 11.0, 12.0, '''
# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
print(df)
