from data_preprocess.standardizing import preprocess_df
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms

LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCH = 3


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path,transforms=None):

        self.data = preprocess_df(pd.read_csv(csv_path, index_col=0))
        self.labels = np.asarray(self.data).astype(np.float32)
        self.transforms = transforms
        self.size = self.labels.shape[0] * self.labels.shape[1]
        # self.zeros_num = int(self.size * percent)
        # self.ones_num = self.size -  self.zeros_num
        # self.matrix = np.hstack((np.zeros(self.zeros_num)))
        self.train_data = self.labels

    def __getitem__(self, index):
        if self.transforms is not None:
            data = self.transforms(self.data)
        label = self.labels[index]
        return (label, label)

    def __len__(self):
        return len(self.data.index)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        torch.nn.Dropout(0.1)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x



if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_data_from_csv = CustomDatasetFromCSV(csv_path='/Users/yangyucheng/Desktop/SCADA/train/201807_1.csv')
    dataset_loader = torch.utils.data.DataLoader(dataset=custom_data_from_csv,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)
    net = Net(n_feature=68, n_hidden=1000, n_output=68)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.MSELoss()

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(dataset_loader):
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, '| Step: ', step, '| Loss ', loss)