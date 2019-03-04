import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import para
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
import para
import pickle
from tqdm import tqdm


# Hyper Parameters
TIME_STEP = 10      # rnn time step / image height
INPUT_SIZE = 1      # rnn input size / image width
LR = 0.02           # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data
EPOCH = 10

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(RNN, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.rnn = nn.GRUCell(  # 这回一个普通的 RNN 就能胜任
            input_size=input_size,
            hidden_size=hidden_size1,     # rnn hidden unit
        )
        self.rnn2 = nn.GRUCell(  # 这回一个普通的 RNN 就能胜任
            input_size=hidden_size1,
            hidden_size=hidden_size2,     # rnn hidden unit
        )
        self.out = nn.Linear(hidden_size2, input_size)
        self.total_train_times = 0

    def forward(self, x, h_state1, h_state2):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state

        output = nn.functional.relu(x)
        output = self.rnn(output, h_state1)
        output = nn.functional.relu(output)
        output = self.rnn2(output, h_state2)
        output = nn.functional.relu(output)
        output = self.out(output)
        return output, h_state1 , h_state2

    def initHidden1(self):
        return torch.zeros(para.batch_size, self.hidden_size1)

    def initHidden2(self):
        return torch.zeros(para.batch_size, self.hidden_size2)

def train(input_tensor, target_tensor, net, optimizer, criterion):
    hidden1 = net.initHidden1().cuda()
    hidden2 = net.initHidden2().cuda()
    optimizer.zero_grad()
    input_length = input_tensor.size(0)
    input_tensor = input_tensor.cuda()
    target_tensor = target_tensor.cuda()
    loss = 0.
    output_tensor = torch.zeros(para.batch_size,141).cuda()
    print(input_tensor[0].device)
    for di in range(90):
        output_tensor, hidden1, hidden2 = net(input_tensor[di], hidden1, hidden2)
        # loss += criterion(output_tensor, target_tensor[di].float())
    for di in range(10):
        output_tensor, hidden1, hidden2 = net(output_tensor, hidden1, hidden2)
        loss += criterion(output_tensor, target_tensor[di+90].float())
    loss.backward()
    print(loss.data)
    optimizer.step()
    return loss.item()


rnn = RNN(141,500,500)
rnn = rnn.cuda(0)
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
        return sum(self.l)-28



def main():
    #1. 预处理meta相关
    dl = DataLoader()
    pp = None
    with open("meta.pkl", 'rb') as file:
        pp = pickle.loads(file.read())



    #2. 数据batch化
    dataset = MyDataset(para.train_data)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=32,
                                                    shuffle=False)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()
    loss_list = []
    h_state1 = None
    h_state2 = None
    #4. 训练相关
    for epoch in range(EPOCH):
        print('epoch {0} start'.format(epoch),flush=True)
        pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
        for step, batch_x in pbar:

            rnn.zero_grad()
            batch_x = pp.transform(batch_x)
            batch_x = batch_x.view(100, -1, 141).float()
            print(batch_x.shape)
            target = batch_x
            loss = train(batch_x, target, rnn, optimizer, loss_func)
            pbar.set_description("Loss {0:.4f}".format(loss))
            optimizer.step()
            # if loss>10:
            #     print(raw_data[step*BATCH_SIZE])
            #     break
        torch.save(rnn, 'unsupervised_rnn{0}.pkl'.format(epoch + 1))
    plt.plot(loss_list)
    plt.show()



if __name__ == '__main__':
    main()