import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Test_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(Test_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.device = device

        self.lstm_LtoR = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm_RtoL = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def init_hidden(self, x):
        return (torch.zeros(4, x.size(0), self.hidden_size).to(self.device),
                torch.zeros(4, x.size(0), self.hidden_size).to(self.device))

    def forward(self, x):
        self.hidden_LtoR = self.init_hidden(x)
        self.hidden_RtoL = self.init_hidden(x)
        lstm_out_LtoR, self.hidden_LtoR = self.lstm_LtoR(x, self.hidden_LtoR)
        reversed_embeds = x
        lstm_out_RtoL, self.hidden_RtoL = self.lstm_RtoL(reversed_embeds, self.hidden_RtoL)

        output = torch.cat((lstm_out_LtoR, lstm_out_RtoL), 2)

        output = F.softmax(self.linear(output), 1)
        output = output[:, -1, :]
        return output


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

        output = F.softmax(self.linear(output), 1)
        output = output[:, -1, :]
        return output
