import numpy as np
from utilities import DataLoader, PreProc
import para
import os 
import pickle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# hyper parameters TODO 放到para.py中
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCH = 3

class MyDataset(Dataset):
    def __init__(self, file_path, transforms=None):
        self.dl = DataLoader()
        self.file_path = file_path
        self.l = [346449, 344698, 347650, 344969, 342916, 346811, 346365, 346392, 344410, 344082, 345631, 345813, 350407, 347874, 351165, 340722, 339552, 341436, 345346, 347898, 344836, 345878, 345580, 347927, 346940, 350253, 347506, 346421, 348828, 346186, 347325, 339731, 346873]
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
            data = self.data[index]

        return torch.tensor(data.astype('f4'))

    def __len__(self):
        return sum(self.l)


def main():
    #1. 预处理meta相关
    dl = DataLoader()
    pp = None
    if not os.path.exists("meta.pkl"): # 预处理meta不存在
        pp = PreProc(dl.t[1:])
        for machine_num in range(33):
            raw_data = dl(para.train_data,machine_num) #加载数据
            data = raw_data[:,1:] #移除时间列
            pp.fit(data)
            print('机器 {0} 处理完毕。'.format(str(machine_num+1)),flush=True)
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

    #2. 数据batch化
    dataset = MyDataset(para.train_data)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

    #3. 模型相关
    net2 = torch.nn.Sequential(
    torch.nn.Dropout(0.03),
    torch.nn.Linear(141, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 141)
    )
    net2.double()
    optimizer = torch.optim.Adam(net2.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.MSELoss()
    loss_list = []

    #4. 训练相关
    for epoch in range(EPOCH):
        print('epoch {0} start'.format(epoch),flush=True)
        for step, batch_x in tqdm(enumerate(dataset_loader),total = len(dataset_loader)):
            net2.zero_grad()
            batch_x = pp.transform(batch_x)
            prediction = net2(batch_x)
            loss = loss_func(prediction, batch_x)
            loss.backward()

            optimizer.step()
            if loss>10:
                print(raw_data[step*BATCH_SIZE])
                break
            loss_list.append(loss)
    plt.plot(loss_list)
    plt.show()



if __name__ == '__main__':
    main()