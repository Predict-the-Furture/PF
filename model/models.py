import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Test_Model(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Test_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2

        self.device = device

        self.lstm_0 = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size, self.hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.hidden_size // 2, self.output_size)

    def forward(self, x):


        out, _ = self.lstm_0(x)
        out = self.dropout(out)
        out, _ = self.lstm_1(x)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2

        self.device = device

        self.lstm_0 = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.lstm_1 = nn.LSTM(self.hidden_size, self.hidden_size // 4, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear_0 = nn.Linear(self.hidden_size // 4, 16)
        self.linear_1 = nn.Linear(16, 8)
        self.linear_2 = nn.Linear(8, self.output_size)

    def forward(self, x):


        out, _ = self.lstm_0(x)
        out = self.dropout(out)
        out, _ = self.lstm_1(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.linear_0(out)
        out = self.dropout(out)
        out = self.linear_1(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out