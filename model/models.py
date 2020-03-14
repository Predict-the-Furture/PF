import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True,batch_first=True)


    def init_hidden(self, x):
        return (Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)),
                Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)))

    def forward(self, x):
        hidden, cell = self.init_hidden(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        linear = nn.Linear(self.hidden_size, self.input_size)
        output = F.softmax(linear(output), 1)
        output = output[:, -1, :]
        return output
