import os
import glob
import pandas as pd

# 获取当前文件夹中所有的 CSV 文件
csv_files = glob.glob('*.csv')
print(csv_files)
for csv_file in csv_files:
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 打乱 DataFrame 的行
    df = df.sample(frac=1).reset_index(drop=True)

    # 将打乱后的 DataFrame 写入新的 CSV 文件
    df.to_csv('shuffled_' + csv_file, index=False)