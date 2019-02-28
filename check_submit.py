#coding=utf-8
import pandas as pd
data = pd.read_csv('sub_DCIC.csv')
for columname in data.columns:
    if data[columname].count() != len(data):
        loc = data[columname][data[columname].isnull().values==True].index.tolist()
        print('列名："{}", 共{}行有缺失，第{}行位置有缺失值'.format(columname,len(loc),loc))