#coding=utf-8
import para
import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
class DataLoader:
    def __init__(self):
        self.t = ['M'] + ['i4'] + ['f4']*15+['i4']+['f4']*3+['i4']+['f4']*26+['i4']+['f4']*5+['i4']+['f4']*12+['i4']+['f4']*2

    def __call__(self, train_data_path, machine_num):
        file_path = train_data_path +  '{0:03d}'.format(machine_num+1) + '/201807.csv'
        data = np.loadtxt(file_path,dtype=np.str,delimiter=',',skiprows=1)
        mask = (data!='').all(axis=1)
        full_data = data[mask]
#         p_miss_data = data[np.logical_not(mask)]
            
#         # record to be filled
#         file_path = train_data_path+'template_submit_result.csv'
#         data = np.loadtxt(file_path,dtype=np.str,delimiter=',',skiprows=1)
#         mask = data[:,1]==str(machine_num)
#         all_miss_data = data[mask]

#         raw_data_str = np.array(full_data)
#         raw_data = full_data

#         for i in range(70):
#             s = raw_data_str[:,i].astype(self.t[i])
#             raw_data.append(s)

        return full_data#,[p_miss_data,all_miss_data]


# class used for pre-process data 
class PreProc:
    def __init__(self,types):
        self.con_features = [True if t=='f4' else False for t in types] # 69个字段预处理分类操作
        self.dis_features = np.logical_not(self.con_features)
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

        
    def transform(self,data):
        '''
        data: (69,) -->单条数据，用于dataset transforms
        data: (N,69) -->多条数据，未用到
        '''
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
                enc = OneHotEncoder(categories='auto')
                enc.fit([[c] for c in self.pp_model[i]])
                res.append(enc.transform(d).todense())
        return torch.tensor(np.concatenate(res,axis=1))

'''
测试代码
'''
if __name__ == '__main__':
    dl = DataLoader()
    machine_num = 0 # 0号为001，1号为002，以此类推
    import para
    raw_data = dl(para.train_data,machine_num) #加载数据

    data = raw_data[:,1:] #移除时间列
    pp = PreProc(dl.t[1:]) 
    pp.fit(data)#预处理

    inputs = pp.transform(data) #预处理输出
    print(inputs.shape)
    print(inputs[0])
