import numpy as np
import sys
# sys.path.append("..")
from utilities import DataLoader, PreProc
import para
import os 
import pickle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from model import EncoderRNN, DecoderRNN
from earlystopping import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
'''
shape 解释
B: batch_size
L: sequence_length
F: feature_number
H: hidden_size
'''

class MyDataset(Dataset):
    def __init__(self, file_path, pp, transforms=None):
        self.dl = DataLoader()
        self.file_path = file_path
        # 未处理不连续L序列数
        # self.l = [3464, 3446, 3476, 3449, 3429, 3468, 3463, 3463, 3444, 3440, 3456, 3458, 3504, 3478, 3511, 3407, 3395, 3414, 3453, 3478, 3448, 3458, 3455, 3479, 3469, 3502, 3475, 3464, 3488, 3461, 3473, 3397, 3468]
        # 处理后连续L序列数
        self.total = [3362,3325,3357,3340,3326,3365,3364,3363,3337,3336,3353,3354,3406,3370,3403,3301,3299,3299,3342,3369,3335,3349,3347,3360,3367,3395,3363,3339,3397,3370,3366,3281,3348]
        self.val_mask = [np.random.rand(t) for t in self.total] #随机数mask
        self.l = [m[m>=0.005].shape[0] for m in self.val_mask] #mask>0.5%是训练集
        self.current_mn = None
        self.data = None
        self.file_counter = 0
        self.full_counter = 0
        self.pp = pp
        self.transforms = transforms
        self.validset = []

    def getValidSet(self):
        return np.array(self.validset)

    def __getitem__(self, index):
        if index == 0: #新epoch重置counter和valid set
            self.file_counter = 0
            self.full_counter = 0
            self.validset = []
        # 寻找index对应的machine_num
        machine_num = 0
        for i in range(len(self.l)):
            if index>=self.l[i]:
                index-=self.l[i]
            else:
                machine_num = i+1
                break
        # 加载数据集
        if self.current_mn != machine_num:
            self.current_mn = machine_num
            self.file_counter = 0
            self.full_counter = 0
            self.data = self.dl(para.train_data,machine_num)

        flag = True # True: 尚未得到新序列， False: 得到新序列
        while flag:
            d = self.data[self.file_counter*para.sequence_length:self.file_counter*para.sequence_length+para.sequence_length]
            if not d.isna().any().any(): # 不存在缺失
                d = d.values # to numpy
                diff = d[1:,0] - d[:-1,0]
                diff = np.array([di.total_seconds() for di in diff])
                if all(diff<14): # 所有记录连续
                    if self.val_mask[self.current_mn-1][self.full_counter]>=0.005: # mask在训练集中
                        data = d # 去除时间列 TODO: 保留时间列用于还原
                        flag = False # 所有记录连续
                    else: # mask在验证集中
                        self.validset.append(d[:,1:])
                    self.full_counter+=1
            self.file_counter +=1

        # if self.transforms is not None:
        encoder_input, decoder_input = self.transforms(data,self.pp)
        # return shape: L x F_original (100 x  69)
        return encoder_input, decoder_input

    def __len__(self):
        return sum(self.l)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, val=False):

    #initalize encoder_hidden
    input_tensor = input_tensor.transpose(0,1)
    encoder_hidden = encoder.initHidden(input_tensor.size(1))
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(2)
    target_tensor = target_tensor.to(device)

    loss = 0.
    #input_tensor: L x B x F
    #encoder_hidden: 1 x B x H
    #encoder_output: L x B x H
    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

    decoder_input = torch.zeros_like(input_tensor[0]) # B x F
    decoder_hidden = encoder_hidden[-1] # B x H
    outputs = decoder(target_tensor,decoder_hidden)

    loss = criterion(outputs, target_tensor[:,0,:,:])

    if not val:
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / target_length, outputs

def mask_and_pp(batch, pp):
    # batch: B x L x F_ori
    mask_len = para.sequence_length//10
    shape = batch.shape
    mask = torch.cat([torch.ones(shape[0],int(shape[1]*0.9),shape[2]), #前90条
            torch.cat([torch.ones(shape[0],int(shape[1]*0.1),1), #后10条，机器号
            torch.ones(shape[0],int(shape[1]*0.1),shape[2]-1)*np.nan],2)],1) #后10条，其他字段
    masked_batch = torch.mul(batch, mask)
    masked_batch = pp.transform(masked_batch).float().transpose(0, 1)
    masked_batch[torch.isnan(masked_batch)] = 0
    batch = pp.transform(batch).float().transpose(0, 1)
    # masked_batch,batch: L x B x F
    return masked_batch, batch

def train_transform(data, pp):
    if len(data.shape) == 3:
        '''
        用于dataset中transform，batch段数据
        data: Bx100x(1+1+68)
        return: encode_data     Bx90x141
                decode_input    Bx4x10x141:
                    decode_data Bx10x141
                    mask        Bx10x141
                    last_obs    Bx10x141
                    delta       Bx10x141
        '''
        transformed = pp.transform(torch.tensor(data[:,:,1:].float()))
        batch_size = transformed.shape[0]
        encode_data = transformed[:,:90] # BxL_90xF
        decode_data = transformed[:,90:] # BxL_10xF
        decode_len = decode_data.shape[1] # L_10
        mask_ori = torch.rand(batch_size,68) < 0.9 # mask 10%左右的字段 注: 不含机器号
        mask = torch.ones(batch_size,10,sum(pp.transformed_features)) # BxL_10xF
        for j in range(1,69):
            start = sum(pp.transformed_features[:j])
            end = sum(pp.transformed_features[:j+1])
            # BxL_10xF(start~end)        B     -> Bx1        -> Bx1x1      -> BxL_10xF(start~end)
            mask[:,:,start:end] = mask_ori[:,j-1].unsqueeze(-1).unsqueeze(-1).expand(-1,decode_len,pp.transformed_features[j])
        last_obs = transformed[:,89:-1] #89~98 shape: BxL_10xF
        for b in range(batch_size):
            for i in range(1,10):
                # 缺失处的last_obs向上寻找
                last_obs[b,i,mask[b,i]==0] = last_obs[b,i-1,mask[b,i]==0]
        
        func = lambda x: x.total_seconds()
        delta = (data[:,90:,0] - data[:,89:-1:,0]).reshape(-1)# B*L_10
        for i, v in enumerate(delta):
            delta[i] = func(v)
        #        B*L_10                     ->   BxL_10                       ->BxL_10x1
        delta = torch.tensor(delta.astype('f4')).reshape(batch_size,decode_len).unsqueeze(-1)
        #BxL_10xF               BxL_10x1         Bx1xF
        delta = torch.mul(delta,torch.ones(batch_size,sum(pp.transformed_features)).unsqueeze(1))
        for b in range(batch_size):
            for i in range(1,10):
                # 缺失处的delta向上加和
                delta[b,i,mask[b,i]==0] = delta[b,i,mask[b,i]==0] + delta[b,i-1,mask[b,i]==0]

        return encode_data.float(),torch.cat([\
                    decode_data.unsqueeze(1).float(),\
                    last_obs.unsqueeze(1).float(),\
                    mask.unsqueeze(1).float(),\
                    delta.unsqueeze(1).float()],dim = 1)
    else:
        '''
        用于dataset中transform，batch段数据
        data: 100x(1+1+68)
        return: encode_data     90x141
                decode_input    4x10x141:
                    decode_data 10x141
                    mask        10x141
                    last_obs    10x141
                    delta       10x141
        '''
        transformed = pp.transform(torch.tensor(data[:,1:].astype('f4')))
        encode_data = transformed[:90] # L_90xF
        decode_data = transformed[90:] # L_10xF
        decode_len = decode_data.shape[0] # L_10
        mask_ori = torch.rand(68) < 0.9 # mask 10%左右的字段 注: 不含机器号
        mask = torch.ones(10,sum(pp.transformed_features)) # BxL_10xF
        for j in range(1,69):
            start = sum(pp.transformed_features[:j])
            end = sum(pp.transformed_features[:j+1])
            # L_10xF(start~end)        1     -> 1x1            -> L_10xF(start~end)
            mask[:,start:end] = mask_ori[j-1].unsqueeze(-1).expand(decode_len,pp.transformed_features[j])
        last_obs = transformed[89:-1] #89~98 shape: L_10xF
        for i in range(1,10):
            # 缺失处的last_obs向上寻找
            last_obs[i,mask[i]==0] = last_obs[i-1,mask[i]==0]
        
        func = lambda x: x.total_seconds() # TimeDelta -> float
        delta = (data[90:,0] - data[89:-1:,0])# L_10
        for i, v in enumerate(delta):
            delta[i] = func(v)
        #                  L_10                 ->L_10x1
        delta = torch.tensor(delta.astype('f4')).unsqueeze(-1)
        #L_10xF               L_10x1         1xF
        delta = torch.mul(delta,torch.ones(sum(pp.transformed_features)).unsqueeze(0))
        for i in range(1,10):
            # 缺失处的delta向上加和
            delta[i,mask[i]==0] = delta[i,mask[i]==0] + delta[i-1,mask[i]==0]

        return encode_data.float(),torch.cat([decode_data.unsqueeze(0).float(), mask.unsqueeze(0).float(), last_obs.unsqueeze(0).float(),delta.unsqueeze(0).float()],dim = 0)



def main():
    # 1.加载预处理
    with open("meta.pkl",'rb') as file:
        pp = pickle.loads(file.read())
    print('加载预处理meta完成。',flush=True)

    # 2.加载数据
    dataset = MyDataset(para.train_data,pp,transforms=train_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=para.batch_size,
                                                 shuffle=False)

    # 3.模型
    encoder = EncoderRNN(141,para.hidden_size).to(device)
    decoder = DecoderRNN(141,para.hidden_size,141,[0]*sum(pp.transformed_features)).to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=para.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=para.learning_rate)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience = 2,verbose=True)
    
    # 4.训练
    for epoch in range(30):
        print('epoch {0} start'.format(epoch), flush=True)
        pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
        for step, batch in pbar:
            encoder.zero_grad()
            decoder.zero_grad()
            encoder_batch, decoder_input = batch[0], batch[1]
            loss,_ = train(encoder_batch.to(device), decoder_input.to(device), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            # if (step+1)%100==0:
            #     print(batch_x[-1,0],pred[-1,0],flush=True)
            # 进度条中展示loss
            pbar.set_description("Loss {0:.4f}".format(loss))

        # val_data = torch.tensor(dataset.getValidSet().astype('f4'))
        # masked_val_data,val_data = mask_and_pp(val_data,pp)
        # val_loss,_ = train(masked_val_data.to(device), val_data.to(device), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, val=True)
        
        # 5.保存模型
        # es(val_loss,[encoder,decoder])
        torch.save(encoder,'encoder{0}.pkl'.format(epoch+1))
        torch.save(decoder,'decoder{0}.pkl'.format(epoch+1))


if __name__ == '__main__':
    main()