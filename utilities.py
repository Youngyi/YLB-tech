#coding=utf-8
import para
import csv
import numpy as np
import torch
import os
import pandas as pd
import pickle
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
        # self.dis_features = np.logical_not(self.con_features)
        self.pp_model = []
        for f in self.con_features:
            if f:
                self.pp_model.append({'sum':0.,'square_sum':0.,'num':0})
            else:
                self.pp_model.append(set())
        
    def fit(self,data):
        for i,f in enumerate(self.con_features):
            if f: # continuous 
                d = data[:,i][np.logical_not(np.isnan(data[:,i].astype('f4')))].astype('f4')
                # self.pp_model[i]['min'] = min(self.pp_model[i]['min'], np.min(d))
                # self.pp_model[i]['max'] = max(self.pp_model[i]['max'], np.max(d))
                self.pp_model[i]['sum'] += np.sum(d)
                self.pp_model[i]['square_sum'] += np.sum(d**2)
                self.pp_model[i]['num'] += d.shape[0]
            else: # discreted
                from collections import Counter
                d = data[:,i][np.logical_not(np.isnan(data[:,i].astype('f4')))].astype('i4')
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
        # print(len(self.post_con_features))




    def transform(self,data):
        '''
        data: (69,) -->单条数据，用于dataset transforms
        data: (B,69) -->多条数据,B为batch_size
        data: (B,L,69)-->多条时序数据
        data: (B,T,L,69)-->[Batch,DataType,Length,Variable]
        '''
        # print(data.shape)
        nd = len(data.shape)
        if nd <= 2:
            res = []
            for i in range(len(self.con_features)):
                d = data[:,i]
                if self.con_features[i]: # con
                    d = d.double().reshape(-1, 1)
                    squ_sum = self.pp_model[i]['square_sum']
                    sum = self.pp_model[i]['sum']
                    num = self.pp_model[i]['num']
                    mean = sum/num
                    var = squ_sum/num - mean**2 #平方的均值-均值的平方
                    stda = StandardScaler()
                    stda.fit([[mean-np.sqrt(1.5*var)],[mean],[mean+np.sqrt(1.5*var)]])
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
                    squ_sum = self.pp_model[i]['square_sum']
                    sum = self.pp_model[i]['sum']
                    num = self.pp_model[i]['num']
                    mean = sum/num
                    var = squ_sum/num - mean**2 #平方的均值-均值的平方
                    stda = StandardScaler()
                    stda.fit([[mean-np.sqrt(1.5*var)],[mean],[mean+np.sqrt(1.5*var)]])
                    res.append(stda.transform(d))
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    enc = OneHotEncoder(categories='auto',handle_unknown='ignore')
                    enc.fit([[c] for c in self.pp_model[i]])
                    res.append(enc.transform(d).todense())
            return torch.tensor(np.concatenate(res,axis=1)).reshape(data.shape[0],data.shape[1],-1)

        elif data.ndimension() == 4:
            res = []
            for i in range(len(self.con_features)):
                # for time_step in range(data.shape[1]):
                d = data[:, 0, :, i]

                if self.con_features[i]:  # con
                    d = d.double().reshape(-1, 1)
                    squ_sum = self.pp_model[i]['square_sum']
                    sum = self.pp_model[i]['sum']
                    num = self.pp_model[i]['num']
                    mean = sum / num
                    var = squ_sum / num - mean ** 2  # 平方的均值-均值的平方
                    stda = StandardScaler()
                    stda.fit([[mean - np.sqrt(1.5 * var)], [mean], [mean + np.sqrt(1.5 * var)]])
                    res.append(stda.transform(d))
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
                    enc.fit([[c] for c in self.pp_model[i]])
                    res.append(enc.transform(d).todense())

            # transformed_data = np.concatenate(res, axis=1)
            # transformed_data.resize((data.shape[0], 1, data.shape[2], 141))
            # transformed_data = torch.tensor(transformed_data)
            transformed_data = torch.tensor(np.concatenate(res, axis=1)).reshape(data.shape[0], 1, data.shape[2], 141)

            res = []
            for i in range(len(self.con_features)):
                # for time_step in range(data.shape[1]):
                d = data[:, 1, :, i] #last observed
                if self.con_features[i]:  # con
                    d = d.double().reshape(-1, 1)
                    squ_sum = self.pp_model[i]['square_sum']
                    sum = self.pp_model[i]['sum']
                    num = self.pp_model[i]['num']
                    mean = sum / num
                    var = squ_sum / num - mean ** 2  # 平方的均值-均值的平方
                    stda = StandardScaler()
                    stda.fit([[mean - np.sqrt(1.5 * var)], [mean], [mean + np.sqrt(1.5 * var)]])
                    res.append(stda.transform(d))
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
                    enc.fit([[c] for c in self.pp_model[i]])
                    res.append(enc.transform(d).todense())
            # transformed_last_observed = np.concatenate(res, axis=1)
            # transformed_last_observed.resize((data.shape[0], 1, data.shape[2], 141))
            transformed_last_observed = torch.tensor(np.concatenate(res, axis=1)).reshape(data.shape[0], 1, data.shape[2], 141)

            res = []
            kv = {0:33, 16:19, 20:19, 47:3, 53:2, 66:2}
            for i in range(len(self.con_features)):
                # for time_step in range(data.shape[1]):
                d = data[:, 2, :, i] #mask
                if self.con_features[i]:  # con
                    d = d.double().reshape(-1, 1)
                    res.append(stda.transform(d))
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    l = []
                    for j in range(d.shape[0]):
                        if d[i][0] == 0:
                            l.append(np.zeros((1, kv.get(i))))
                        else:
                            l.append(np.ones((1, kv.get(i))))
                    res.append(np.concatenate(l, axis=0))
            # transformed_mask = np.concatenate(res, axis=1)
            # transformed_mask.resize((data.shape[0], 1, data.shape[2], 141))
            transformed_mask = torch.tensor(np.concatenate(res, axis=1)).reshape(data.shape[0], 1, data.shape[2], 141)

            res = []
            for i in range(len(self.con_features)):
                # for time_step in range(data.shape[1]):
                d = data[:, 2, :, i]  # delta
                if self.con_features[i]:  # con
                    d = d.double().reshape(-1, 1)
                    res.append(d)
                else:  # dis
                    d = d.int().reshape(-1, 1)
                    l = []
                    for j in range(d.shape[0]):
                        l.append(np.zeros((1, kv.get(i))) + d[j][0].numpy())
                    res.append(np.concatenate(l, axis=0))
            # transformed_delta = np.concatenate(res, axis=1)
            # transformed_delta.resize((data.shape[0], 1, data.shape[2], 141))
            # print(transformed_data.shape)
            # print(transformed_last_observed.shape)
            # print(transformed_mask.shape)
            # print(transformed_delta.shape)
            Delta = np.concatenate(res, axis=1)
            Delta = Delta / Delta.max()
            transformed_delta = torch.tensor(Delta).reshape(data.shape[0], 1, data.shape[2], 141)
            output = torch.cat([transformed_data,transformed_last_observed,transformed_mask,transformed_delta],dim=1)

            return output.float()


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
                squ_sum = self.pp_model[j]['square_sum']
                sum = self.pp_model[j]['sum']
                num = self.pp_model[j]['num']
                mean = sum/num
                var = squ_sum/num - mean**2 #平方的均值-均值的平方
                stda = StandardScaler()
                stda.fit([[mean-np.sqrt(1.5*var)],[mean],[mean+np.sqrt(1.5*var)]])
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
测试代码
'''
if __name__ == '__main__':
    dl = DataLoader()
    pp = None
    if not os.path.exists("meta2.pkl"): # 预处理meta不存在
        pp = PreProc(dl.t[1:])
        for machine_num in range(1,34):
            raw_data = dl(para.train_data,machine_num) #加载数据
            data = raw_data.values[:,1:] #移除时间列
            pp.fit(data)
            print('机器 {0} 处理完毕。'.format(str(machine_num)),flush=True)
        #保存预处理meta
        output_hal = open("meta.pkl", 'wb')
        s = pickle.dumps(pp)
        output_hal.write(s)
        output_hal.close()
        print('保存预处理meta完成。',flush=True)
    else: #加载预处理meta
        with open("meta.pkl",'rb') as file:
            pp = pickle.loads(file.read())
        print('加载预处理meta完成。',flush=True)
    # dl = DataLoader()
    data = dl(para.train_data,1)[0:100]
    import torch 
    a = pp.transform(torch.tensor(data.values[:,1:].astype('f4')))
    print(a)
    print(pp.recover(a))
