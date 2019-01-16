import pandas as pd
import numpy as np
from data_preprocess.generate_random_NaN import listdir


def std_to_average(df):
    return (df - df.mean()) / (df.max() - df.min())


input_path = '/Users/yangyucheng/Desktop/SCADA/train'
df_test_1 = pd.read_csv(listdir(input_path)[0], index_col=0)
df_test_1 = df_test_1.drop(['ts', 'wtid'], axis=1)
df1 = (df_test_1 - df_test_1.mean()) / (df_test_1.max() - df_test_1.min())
print(df1.head())