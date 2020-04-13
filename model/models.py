import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Test_Model(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Test_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device

        self.lstm_0 = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size, self.hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.hidden_size // 2, self.input_size)

    def forward(self, x):


        out, _ = self.lstm_0(x)
        out = self.dropout(out)
        out, _ = self.lstm_1(x)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True,batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def init_hidden(self, x):
        return (Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)),
                Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)))

    def forward(self, x):
        hidden, cell = self.init_hidden(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        print(output.size())
        output = F.softmax(self.linear(output), 1)
        output = output[:, -1, :]
        return output
