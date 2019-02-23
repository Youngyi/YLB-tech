import pandas as pd
import numpy as np
from tqdm import *
import matplotlib.pyplot as plt
import pickle
import os
import sys
from datetime import *
import time
import math
import torch
import para
import exp2

from torch.utils.data.dataset import Dataset

class DataLoader:
    def __init__(self):
        self.t = ['M'] + ['i4'] + ['f4']*15+['i4']+['f4']*3+['i4']+['f4']*26+['i4']+['f4']*5+['i4']+['f4']*12+['i4']+['f4']*2

    def __call__(self, data):
        mask = (data!='').all(axis=1)
        full_data = data[mask]
        # print(full_data.shape)
        # p_miss_data = data[np.logical_not(mask)]
        #
        # # record to be filled
        # file_path = train_data_path+'template_submit_result.csv'
        # data = np.loadtxt(file_path,dtype=np.str,delimiter=',',skiprows=1)
        # mask = data[:,1]==str(machine_num)
        # all_miss_data = data[mask]
        #
        # raw_data_str = np.array(full_data)
        # raw_data = full_data
        #
        # for i in range(70):
        #     s = raw_data_str[:,i].astype(self.t[i])
        #     raw_data.append(s)

        return full_data #,p_miss_data,all_miss_data

class MyDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.dl = DataLoader()
        self.data = self.dl(data.values[:,1:])
        self.l = [3598, 3606, 3619, 3597, 3594, 3594, 3595, 3595, 3592, 3596, 3597, 3633, 3635, 3642, 3633, 3572, 3551, 3572, 3594, 3597, 3599, 3636, 3632, 3598, 3651, 3638, 3626, 3623, 3632, 3635, 3637, 3562, 3626]
        self.l2 = [359846, 360660, 361993, 359744, 359457, 359464, 359510, 359535, 359219, 359608, 359752, 363304, 363512, 364214, 363300, 357219, 355158, 357234, 359494, 359743, 359995, 363685, 363212, 359844, 365176, 363810, 362638, 362355, 363215, 363533, 363743, 356215, 362674]
        self.current_mn = self.data[0][0]
        self.transforms = transforms
        self.last_index = self.l[self.current_mn-1]
        self.num_of_last_data = self.l2[self.current_mn-1] - self.l[self.current_mn-1]*100
    def __getitem__(self, index):

        if self.transforms is not None:
            data = self.transforms.transform(self.data[index])
        else:
            # data = self.data[index]
            data = self.data[index*100:index*100+100,:]
        if index == self.last_index:
            data = self.data[self.l2[self.current_mn-1]-100:self.l2[self.current_mn-1], :]
        data = pd.DataFrame(data=data,
          index=np.array(range(0, 100)),
          columns=np.array(range(0, 69)))
        flag = False
        for i in range(69):
            if data[i].isnull().any():
                flag = True
        origin_data = data
        if flag:
            using_teacher_enforcing = True

            data = data.fillna(0)
        else:
            using_teacher_enforcing = False
        # 2        # print(data)
        # using_teacher_enforcing = data.isnan().any()
        data = data.values
        return torch.tensor(data.astype('f4')), using_teacher_enforcing

    def __len__(self):
        return self.last_index+1   #machine number minus 1


def metric_fun(pred , label):
    return math.exp(-100*abs(label - pred)/max(abs(label),10**(-15)))

def forecast(input_tensor, encoder, decoder, use_teacher_forcing):
    encoder_outputs = torch.zeros(100, encoder.hidden_size)
    encoder_hidden = encoder.initHidden()

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.zeros_like(input_tensor[0])  # B x F
    output = []

    decoder_hidden = encoder_hidden.view(para.batch_size, -1)  # B x H
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(100):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            decoder_input = input_tensor[di]  # Teacher forcing
        return input_tensor
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(100):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            decoder_input = decoder_output.detach()  # detach from history as input
            output.append(decoder_output)
        return torch.stack(output, dim=1)

def get_data(i):
    """
    get_ori_data
    """
    start = time.clock()
    data = pd.read_csv('/Users/yangyucheng/Desktop/SCADA/dataset/' + str(i).zfill(3) + '/201807.csv', parse_dates=[0])
    res = pd.read_csv('/Users/yangyucheng/Desktop/SCADA/template_submit_result.csv', parse_dates=[0])[['ts', 'wtid']]

    res = res[res['wtid'] == i]
    # res['flag'] = 1
    data = res.merge(data, on=['wtid', 'ts'], how='outer')
    data = data.sort_values(['wtid', 'ts']).reset_index(drop=True)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    return data
df = pd.DataFrame()
dl = DataLoader()
size = []

with open("meta.pkl", 'rb') as file:
    pp = pickle.loads(file.read())
encoder = exp2.EncoderRNN(141,para.hidden_size)
decoder = exp2.DecoderRNN(141,para.hidden_size,141)
np.set_printoptions(threshold = np.inf)

for i in tqdm(range(1,34)):
    data = get_data(i)
    dataset = MyDataset(data)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=1,
                                                 shuffle=False)
    res = []
    for step, batch_x in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        use_teacher_forcing = batch_x[1]
        batch_x = batch_x[0]
        batch_x = pp.transform(batch_x)
        # print(pp.recover(batch_x).shape)
        batch_x = batch_x.view(para.sequence_length, -1, 141).float()
        if use_teacher_forcing:
            result = forecast(batch_x, encoder, decoder,use_teacher_forcing=use_teacher_forcing)
        else:
            result = batch_x
        result = result[:,0,:]
        result = pp.recover(result) #numpy.ndarray
        if step == len(dataset_loader)-1: # the last step
            result = result[100-dataset.num_of_last_data:100,:]
        res.append(result)
    data = np.concatenate(res,axis=0)
    print(data.shape)
    # df = pd.concat([df,data],axis = 0) not completed


sub = df[df.flag==1].copy().reset_index(drop = True)
del sub['flag']
res = pd.read_csv('/Users/yangyucheng/Desktop/SCADA/template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
DF = res.merge(sub, on=['wtid','ts'],how = 'outer')
DF.to_csv('/Users/yangyucheng/Desktop/SCADA/submit/sub_DCIC.csv',index=False)