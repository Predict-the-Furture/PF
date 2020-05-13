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
        self.output_size = 4

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
        out =  F.softmax(out, dim=1)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        modules = []
        parameters = [[3, 32, 50, False],
                      [32, 48, 25, True],
                      [48, 64, 12, False],
                      [64, 96, 6, True]]
        for layer in parameters:
            modules.append(self.conv_layer(*layer))
        self.conv_sequential = nn.ModuleList(modules)

        modules = []
        parameters = [[96 * 3 * 3, 256, 'relu', True],
                      [256, 2, 'softmax', False]]
        for layer in parameters:
            modules.append(self.dense_layer(*layer))
        self.dense_sequential = nn.ModuleList(modules)

    def conv_layer(self, conv_ic, conv_oc, conv_pd, dropout=False):
        if dropout:
            return nn.Sequential(nn.Conv2d(conv_ic, conv_oc, 3, 3, padding=conv_pd), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(p=0.25))
        else:
            return nn.Sequential(nn.Conv2d(conv_ic, conv_oc, 3, 3, padding=conv_pd), nn.ReLU(), nn.MaxPool2d(2))

    def dense_layer(self, linear_i, linear_o, activation='relu', dropout=False):
        if activation == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Softmax()
        if dropout:
            return nn.Sequential(nn.Linear(linear_i, linear_o), activation, nn.Dropout(0.5))
        else:
            return nn.Sequential(nn.Linear(linear_i, linear_o), activation, nn.Dropout(0.5))

    def forward(self, x):
        for layer in self.conv_sequential:
          x = layer(x)
        x = x.view(-1, 96 * 3 * 3)

        for layer in self.dense_sequential:
          x = layer(x)
        return x