from RNN import unsupervised_rnn as nn
import torch
rnn = nn.RNNCell(10, 10)
input = torch.randn(3, 3, 10)
print(input.size(1))
hx = torch.randn(3, 10)
output = []
for i in range(3):
    hx = rnn(input[i], hx)
    output.append(hx)
for i in range(3):
    hx = rnn(hx, hx)
    output.append(hx)
print(output)