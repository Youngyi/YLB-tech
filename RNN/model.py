import para
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def initHidden(self,batch_size):
        return torch.zeros(1,batch_size, self.hidden_size, device=device)

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
        output = input.view(-1, 141)
        # output = F.relu(output)
        output = self.gru(output, hidden) # B x H
        hidden = output
        output = self.out(output)  # B x F
        return output, hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=device)
