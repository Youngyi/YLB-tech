import numpy as np
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
        self.l = [3464, 3446, 3476, 3449, 3429, 3468, 3463, 3463, 3444, 3440, 3456, 3458, 3504, 3478, 3511, 3407, 3395, 3414, 3453, 3478, 3448, 3458, 3455, 3479, 3469, 3502, 3475, 3464, 3488, 3461, 3473, 3397, 3468]
        self.current_mn = None
        self.data = None
        self.transforms = transforms


    def __getitem__(self, index):
        # 寻找index对应的machine_num和index
        machine_num = 0
        for i in range(len(self.l)):
            if index>=self.l[i]:
                index-=self.l[i]
            else:
                machine_num = i
                break
        # Load data
        if self.current_mn != machine_num:
            self.current_mn = machine_num
            self.data = self.dl(para.train_data,machine_num)[:,1:]

        if self.transforms is not None:
            data = self.transforms.transform(self.data[index])
        else:
            # data = self.data[index]
            data = self.data[index*100:index*100+100,:]
        return torch.tensor(data.astype('f4'))

    def __len__(self):
        return sum(self.l)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        '''
        input: L x B x F
        hidden: 1 x B x H
        output: L x B x H
        '''
        output, hidden = self.gru(input,hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,para.batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        '''
        input: B x F
        hidden: B x H
        output: B x F
        '''
        output = input.view(para.batch_size, -1)
        output = F.relu(output)
        output = self.gru(output, hidden) # B x H
        hidden = output
        output = self.out(output)  # B x F
        return output, hidden

    def initHidden(self):
        return torch.zeros(para.batch_size, self.hidden_size, device=device)


def forecast(input_tensor, encoder, decoder):
    encoder_outputs = torch.zeros(100, encoder.hidden_size, device=device)
    encoder_hidden = encoder.initHidden()

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.zeros_like(input_tensor[0])  # B x F

    decoder_hidden = encoder_hidden.view(para.batch_size, -1)  # B x H
    if True:
        # Teacher forcing: Feed the target as the next input
        for di in range(100):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            decoder_input = input_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(100):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            decoder_input = decoder_output.detach()  # detach from history as input
    return decoder_output


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(100, encoder.hidden_size, device=device)

    loss = 0.
    #input_tensor: L x B x F
    #encoder_hidden: 1 x B x H
    #encoder_output: L x B x H
    # print(input_tensor.shape)
    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

    decoder_input = torch.zeros_like(input_tensor[0]) # B x F

    decoder_hidden = encoder_hidden.view(para.batch_size,-1) # B x H

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
    # print(loss.data)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def main():
    # 1.加载数据
    data = np.loadtxt('small.csv',dtype=np.str,delimiter=',',skiprows=1)
    data = torch.tensor(data[:,1:].astype('f4'))

    # 2.预处理
    with open("meta.pkl",'rb') as file:
        pp = pickle.loads(file.read())
    print('加载预处理meta完成。',flush=True)
    # inputs = pp.transform(data)
    # feature_num = inputs.shape[-1]
    # print(inputs.shape)
    # 2.1 batch化
    # inputs = inputs.view(para.sequence_length,-1,feature_num).float() # L x B x F

    # 3.模型
    encoder = EncoderRNN(141,para.hidden_size).to(device)
    decoder = DecoderRNN(141,para.hidden_size,141).to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=para.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=para.learning_rate)
    criterion = nn.MSELoss()
    dataset = MyDataset(para.train_data)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=para.batch_size,
                                                 shuffle=False)
    # 4.训练
    for epoch in range(10):
        print('epoch {0} start'.format(epoch), flush=True)
        for step, batch_x in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            encoder.zero_grad()
            decoder.zero_grad()
            batch_x = pp.transform(batch_x)
            # print(pp.recover(batch_x).shape)
            batch_x = batch_x.view(para.sequence_length, -1, 141).float()
            loss = train(batch_x, batch_x, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print(loss)
    torch.save(encoder,'encoder0.pkl')
    torch.save(decoder,'decoder0.pkl')
    # loss = train(inputs,inputs,encoder,decoder,encoder_optimizer, decoder_optimizer, criterion)
    # print(loss)


if __name__ == '__main__':
    main()