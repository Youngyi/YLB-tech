import pandas as pd
import numpy as np
from data_preprocess.generate_random_NaN import listdir


def std_to_average(df):
    return (df - df.mean()) / (df.max() - df.min())


def preprocess_df(df):
    df_part1 = df[['ts', 'wtid']]
    df_part2 = df.drop(['ts', 'wtid'], axis=1)
    df_part2 = (df_part2 - df_part2.mean()) / (df_part2.max() - df_part2.min())
    df = pd.concat([df_part1, df_part2], sort=False, axis=1)
    return df

# input_path = '/Users/yangyucheng/Desktop/SCADA/train'
# df_test_1 = pd.read_csv(listdir(input_path)[0], index_col=0)
# df1 = preprocess_df(df_test_1)
# print(df1.head())