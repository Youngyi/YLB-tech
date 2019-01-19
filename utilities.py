#coding=utf-8
import para
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self):
        self.t = ['M'] + ['i4'] + ['f4']*15+['i4']+['f4']*3+['i4']+['f4']*26+['i4']+['f4']*5+['i4']+['f4']*12+['i4']+['f4']*2

    def __call__(self, train_data_path, machine_num):
        file_path = train_data_path +  '{0:03d}'.format(machine_num+1) + '/201807.csv'
        full_data = []
        p_miss_data = []
        with open(file_path,'r') as file:
            data = csv.reader(file)
            data.__next__() #去除第一行
            for record in data:
                if any([r==''for r in record]):
                    p_miss_data.append(record) #部分字段缺失条目
                else:
                    full_data.append(record)
            
        # record to be filled
        file_path = train_data_path+'template_submit_result.csv'
        all_miss_data = []
        with open(file_path,'r') as file:
            data = csv.reader(file)
            data.__next__()
            for record in data:
                if record[1] == str(machine_num):
                    all_miss_data.append(record) #全部字段缺失条目

        
        raw_data_str = np.array(full_data)
        raw_data = []

        for i in range(70):
            s = raw_data_str[:,i].astype(self.t[i])
            raw_data.append(s)

        return np.array(raw_data).T,[p_miss_data,all_miss_data]


# class used for pre-process data 
class PreProc:
    def __init__(self, data, types):
        self.features = [True if t=='f4' else False for t in types] # 68个字段预处理分类操作
        self.continuous, self.discrete = self.classify(data)
        self.proc_cont = self.standardization(self.continuous)
        self.proc_disc = self.one_hot(self.discrete)
        
    def classify(self, data):
        return data[:,self.features], data[:,np.logical_not(self.features)]
        
    def standardization(self, data):
        self.stda = StandardScaler()
        data = data.astype('f4')
        return self.stda.fit_transform(data)
        
    def one_hot(self,data):
        self.enc = OneHotEncoder(categories='auto')
        return self.enc.fit_transform(data).todense()
