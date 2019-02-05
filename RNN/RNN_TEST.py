import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from utilities import DataLoader, PreProc
import para
from tensorboardX import SummaryWriter
import torch.nn as nn
torch.manual_seed(1)    # reproducible
import tqdm


# Hyper Parameters
TIME_STEP = 10      # rnn time step / image height
INPUT_SIZE = 1      # rnn input size / image width
LR = 0.02           # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data
EPOCH = 3

class RNN(nn.Module):
    def __init__(self, input_size, hiden_size1, hidden_size2):
        super(RNN, self).__init__()

        self.rnn = nn.RNNCell(  # 这回一个普通的 RNN 就能胜任
            input_size=input_size,
            hidden_size=hiden_size1,     # rnn hidden unit
        )
        self.rnn2 = nn.RNNCell(  # 这回一个普通的 RNN 就能胜任
            input_size=hidden_size1,
            hidden_size=hidden_size2,     # rnn hidden unit
        )
        self.out = nn.Linear(hidden_size2, input_size)
        self.total_train_times = 0

    def forward(self, x, h_state1, h_state2):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        # print(x)
        self.total_train_times += 1
        # print(self.total_train_times)
        outs = []    # 保存所有时间点的预测值
        # print(x.shape)
        for time_step in range(7):    # 对每一个时间点计算 output
            h_state1 = self.rnn(x[:, time_step, :], h_state1)
            h_state2 = self.rnn2(h_state1, h_state2)
            out = self.out(h_state2)
            print(out)
            outs.append(out)
        for time_step in range(3):  # 对每一个时间点计算 output
            h_state1 = self.rnn(out, h_state1)
            h_state2 = self.rnn2(h_state1, h_state2)
            out = self.out(h_state2)
            outs.append(out)
        return torch.stack(outs, dim=1), h_state1 , h_state2


rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
loss_func = nn.MSELoss()

h_state1 = None   # 要使用初始 hidden state, 可以设成 None
h_state2 = None

# for step in range(10000):
#     start, end = step * np.pi, (step+1)*np.pi   # time steps
#     # sin 预测 cos
#     steps = np.linspace(start, end, 10, dtype=np.float32)
#     x_np = np.sin(steps)    # float32 for converting torch FloatTensor
#     y_np = np.cos(steps)
#
#     x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
#     y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
#     # print(x)
#     prediction, h_state1, h_state2 = rnn(x, h_state1, h_state2)   # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
#     loss = loss_func(prediction, y)     # cross entropy loss
#     optimizer.zero_grad()               # clear gradients for this training step
#     loss.backward(retain_graph=True) # backpropagation, compute gradients
#     optimizer.step()                    # apply gradients
#     plt.plot(steps, y_np.flatten(), 'r-')
#     plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
#     plt.draw()
#     plt.pause(0.05)
#
# plt.ioff()
# plt.show()


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
            raw_data = dl(para.train_data, machine_num) #加载数据
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
    rnn = RNN(input_size=141, hiden_size1=100, hiden_size2=100)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()
    loss_list = []
    h_state1 = None
    h_state2 = None
    #4. 训练相关
    for epoch in range(EPOCH):
        print('epoch {0} start'.format(epoch),flush=True)
        for step, batch_x in tqdm(enumerate(dataset_loader),total = len(dataset_loader)):
            net2.zero_grad()
            batch_x = pp.transform(batch_x)
            prediction, h_state1, h_state2 = rnn(batch_x, h_state1, h_state2)
            loss = loss_func(prediction, batch_x)
            loss.backward()

            optimizer.step()
            # if loss>10:
            #     print(raw_data[step*BATCH_SIZE])
            #     break
            loss_list.append(loss)
    plt.plot(loss_list)
    plt.show()



if __name__ == '__main__':
    main()