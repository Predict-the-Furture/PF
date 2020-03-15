import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


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
