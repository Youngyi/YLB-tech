import numpy as np
import sys
sys.path.append("..")
from utilities import DataLoader, PreProc
import para
import os 
import pickle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from model import EncoderRNN, DecoderRNN

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
    def __init__(self, file_path, transforms=None):
        self.dl = DataLoader()
        self.file_path = file_path
        # 未处理不连续L序列数
        # self.l = [3464, 3446, 3476, 3449, 3429, 3468, 3463, 3463, 3444, 3440, 3456, 3458, 3504, 3478, 3511, 3407, 3395, 3414, 3453, 3478, 3448, 3458, 3455, 3479, 3469, 3502, 3475, 3464, 3488, 3461, 3473, 3397, 3468]
        # 处理后连续L序列数
        self.l = [3362,3325,3357,3340,3326,3365,3364,3363,3337,3336,3353,3354,3406,3370,3403,3301,3299,3299,3342,3369,3335,3349,3347,3360,3367,3395,3363,3339,3397,3370,3366,3281,3348]
        self.current_mn = None
        self.data = None
        self.counter = 0
        self.transforms = transforms


    def __getitem__(self, index):
        if index == 0: #新epoch重置counter
            self.counter = 0
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
            self.counter = 0
            self.data = self.dl(para.train_data,machine_num)

        flag = True # True: 尚未得到新序列， False: 得到新序列
        while flag:
            d = self.data[self.counter*para.sequence_length:self.counter*para.sequence_length+para.sequence_length]
            if not d.isna().any().any(): # 不存在缺失
                d = d.values # to numpy
                diff = d[1:,0] - d[:-1,0]
                diff = np.array([di.total_seconds() for di in diff])
                if all(diff<14): # 所有记录连续
                    data = d[:,1:] # 去除时间列 TODO: 保留时间列用于还原
                    flag = False # 所有记录连续
            self.counter +=1


        if self.transforms is not None:
            data = self.transforms.transform(self.data[index])
        # return shape: L x F_original (100 x  69)
        return torch.tensor(data.astype('f4'))

    def __len__(self):
        return sum(self.l)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    #drop out part
    #mask last 10 data from length 100 as 0
    input = input_tensor.numpy()
    input = input[:90]
    padding = np.zeros((10,input.shape[1],input.shape[2]))
    input = np.concatenate([input,padding])
    input_tensor = torch.tensor(input.astype('f4')).to(device)
    #initalize encoder_hidden
    encoder_hidden = encoder.initHidden(input_tensor.size(1))
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    target_tensor = target_tensor.to(device)
    encoder_outputs = torch.zeros(para.sequence_length, encoder.hidden_size, device=device)

    loss = 0.
    #input_tensor: L x B x F
    #encoder_hidden: 1 x B x H
    #encoder_output: L x B x H
    # print(input_tensor.shape)
    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

    decoder_input = torch.zeros_like(input_tensor[0]) # B x F

    decoder_hidden = encoder_hidden.view(-1,para.hidden_size) # B x H

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #decoder_input: B x F
            #decoder_output: B x F
            #decoder_hidden: B x H
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
    # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            decoder_input = decoder_output.detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].float())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def main():
    # 1.加载数据
    dataset = MyDataset(para.train_data)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=para.batch_size,
                                                 shuffle=False)

    # 2.预处理
    with open("meta.pkl",'rb') as file:
        pp = pickle.loads(file.read())
    print('加载预处理meta完成。',flush=True)

    # 3.模型
    encoder = EncoderRNN(141,para.hidden_size).to(device)
    decoder = DecoderRNN(141,para.hidden_size,141).to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=para.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=para.learning_rate)
    criterion = nn.MSELoss()
    
    # 4.训练
    for epoch in range(para.num_epoch):
        print('epoch {0} start'.format(epoch), flush=True)
        pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
        for step, batch_x in pbar:
            encoder.zero_grad()
            decoder.zero_grad()
            batch_x = pp.transform(batch_x)
            batch_x = batch_x.view(para.sequence_length, -1, 141).float()
            loss = train(batch_x, batch_x, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            # 进度条中展示loss
            pbar.set_description("Loss {0:.4f}".format(loss))
        # 5.保存模型
        torch.save(encoder,'encoder{0}.pkl'.format(epoch+1))
        torch.save(decoder,'decoder{0}.pkl'.format(epoch+1))


if __name__ == '__main__':
    main()