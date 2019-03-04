#coding=utf-8
import para
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
class DataLoader:
    def __init__(self):
        self.t = ['M'] + ['i4'] + ['f4']*15+['i4']+['f4']*3+['i4']+['f4']*26+['i4']+['f4']*5+['i4']+['f4']*12+['i4']+['f4']*2

    def __call__(self, train_data_path, machine_num):
        data = pd.read_csv(train_data_path+str(machine_num).zfill(3)+'/201807.csv',parse_dates=[0])
        res = pd.read_csv(train_data_path + 'template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
        res = res[res['wtid']==machine_num]
        data = res.merge(data, on=['wtid','ts'],how = 'outer')
        data = data.sort_values(['wtid','ts']).reset_index(drop = True)
        return data


# class used for pre-process data 
class PreProc:
    def __init__(self,types):
        self.con_features = [True if t=='f4' else False for t in types] # 69个字段预处理分类操作
        self.dis_features = np.logical_not(self.con_features)
        self.post_con_feature = []
        self.pp_model = []
        for f in self.con_features:
            if f:
                self.pp_model.append({'min':0.,'max':0.})
            else:
                self.pp_model.append(set())
        
    def fit(self,data):
        for i,f in enumerate(self.con_features):
            if f: # continuous 
                d = data[:,i].astype('f4')
                self.pp_model[i]['min'] = min(self.pp_model[i]['min'], np.min(d))
                self.pp_model[i]['max'] = max(self.pp_model[i]['max'], np.max(d))
            else: # discreted
                from collections import Counter
                d = data[:,i].astype('i4')
                for key in Counter(d):
                    if key not in self.pp_model[i]:
                        self.pp_model[i].add(key)
        self.post_con_features = []
        for i in range(len(self.con_features)):
            if self.con_features[i]:
                self.post_con_features.append(True)
            else:
                for num in range(len(self.pp_model[i])):
                    self.post_con_features.append(False)
        print(len(self.post_con_features))
        
    def transform(self,data):
        '''
        data: (69,) -->单条数据，用于dataset transforms
        data: (N,69) -->多条数据,N为batch_size
        data: (N,timestep,69)-->多条时序数据
        '''
        # print(data.shape)
        nd = len(data.shape)
        if nd <= 2:
            res = []
            for i in range(len(self.con_features)):
                d = data[:,i]
                if self.con_features[i]: # con
                    d = d.double().reshape(-1, 1)
                    stda = MinMaxScaler()
                    stda.fit([[self.pp_model[i]['min']], [self.pp_model[i]['max']]])
                    res.append(stda.transform(d))
                else: #dis
                    d = d.int().reshape(-1, 1)
                    enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
                    enc.fit([[c] for c in self.pp_model[i]])
                    res.append(enc.transform(d).todense())
            return torch.tensor(np.concatenate(res,axis=1))

        elif data.ndimension() == 3:
            res = []
            for i in range(len(self.con_features)):
                # for time_step in range(data.shape[1]):
                d = data[:, :, i]
                if self.con_features[i]:  # con
                    d = d.double().reshape(-1, 1)
                    stda = MinMaxScaler()
                    stda.fit([[self.pp_model[i]['min']], [self.pp_model[i]['max']]])
                    res.append(stda.transform(d))
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
                    enc.fit([[c] for c in self.pp_model[i]])
                    res.append(enc.transform(d).todense())
            return torch.tensor(np.concatenate(res, axis=1))
        else:
            raise NotImplementedError


    def recover(self,data):
        #data: (N,141) -->(N,69)
        self.post_con_features = []
        for i in range(len(self.con_features)):
            if self.con_features[i]:
                self.post_con_features.append(True)
            else:
                for num in range(len(self.pp_model[i])):
                    self.post_con_features.append(False)
        res = []
        i = 0
        j = 0
        while i < len(self.post_con_features):
            if self.post_con_features[i]:    # con
                d = data[:, i]
                d = d.double().reshape(-1, 1)
                stda = MinMaxScaler()
                stda.fit([[self.pp_model[j]['min']], [self.pp_model[j]['max']]])
                res.append(stda.inverse_transform(d))
                i = i + 1
                j = j + 1
            else:  # dis
                k = i
                while(k<141 and (not self.post_con_features[k])):
                    k+=1
                d = data[:,i:k]
                d = d.int()
                enc = OneHotEncoder(categories='auto')
                enc.fit([[c] for c in self.pp_model[j]])
                res.append(enc.inverse_transform(d))
                i = k
                j = j + 1
        return np.concatenate(res, axis=1)

'''
# 测试代码
# '''
# if __name__ == '__main__':
#     dl = DataLoader()
#     machine_num = 0 # 0号为001，1号为002，以此类推
#     import para
#     raw_data = dl(para.train_data,machine_num) #加载数据
#
#     data = raw_data[:,1:] #移除时间列
#     pp = PreProc(dl.t[1:])
#     pp.fit(data)#预处理
#
#     inputs = pp.transform(data) #预处理输出
#     print(inputs.shape)
#     print(inputs[1])
