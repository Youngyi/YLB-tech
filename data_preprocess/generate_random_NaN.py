import pandas as pd
import os
import random

input_path = '/Users/yangyucheng/Desktop/SCADA/train'
file_root_name = '201807.csv'


def listdir(path):
    list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list)
        else:
            list.append(file_path)
    return list


def random_raplace_data_to_NaN(dataframe, percent):
    if percent > 100 or percent < 0:
        raise ValueError
    else:
        for index, row in dataframe.iterrows():
            for col_name in dataframe.columns:
                random_num = random.randint(1,100)
                if random_num <= percent:
                    dataframe.loc[index,col_name] = 0


    return dataframe


def raplace_colum_data_to_NaN(dataframe, colum_list):
    for index, row in dataframe.iterrows():
        for col_name in dataframe.columns:
            if col_name in colum_list:
                row[col_name] = pd.NaT
    return dataframe


# df = pd.read_csv(listdir(input_path)[0])
# print(df.head())
# df = random_raplace_data_to_NaN(df, 10)
# print(df.head())