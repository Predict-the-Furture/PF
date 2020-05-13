import os
import time
import importlib
import argparse
import torch
import numpy as np

from datetime import datetime
from pytz import timezone
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from utils import folder
from model import CNN, Test_Model
from data_loader import DiabetesDataset_CNN

tpu_check = importlib.find_loader('torch_xla')
if tpu_check is not None:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

timezone = timezone('Asia/Seoul')

class Trainer():
    def __init__(self, args):
        self.tpu = False
        if args.device == 'cpu':
            self.device = 'cpu'
        elif args.device == 'gpu':
            self.device = 'cuda:0'
        elif args.device == 'tpu':
            self.device = xm.xla_device()
            self.tpu = True
        else:
            exit(0)
        
        #torch.manual_seed(777)

        self.save_dir = args.save_dir
        self.start_epoch = 0
        self.summary_write = args.summary_write
        self.save_model = args.save_model
        self.end_epoch = args.end_epoch

        folder([self.save_dir, self.save_dir + '/models', self.save_dir + '/runs'])

        self.summary = SummaryWriter(self.save_dir + '/runs/' + datetime.now(timezone).strftime("%Y-%m-%d-%H%M"))

        self.dataset_train = DiabetesDataset_CNN(train=True)
        self.dataset_test = DiabetesDataset_CNN(train=False)
        self.train_loader = DataLoader(dataset=self.dataset_train, batch_size=32, shuffle=True, num_workers=0)
        self.evaluate_loader = DataLoader(dataset=self.dataset_test, batch_size=32, shuffle=False, num_workers=0)

        self.model = CNN()

        self.model = self.model.to(self.device)

        if args.resume is not None:
            self.load_checkpoint()

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        bar_total = tqdm(range(self.start_epoch, self.end_epoch), desc='Training', leave=False)
        n_samples = len(self.train_loader.sampler)
        for self.epoch in bar_total:
            total_loss = 0
            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #inputs = inputs.transpose(1, 3)
                y_pred = self.model(inputs)
                loss = self.criterion(y_pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                if self.tpu:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(self.train_loader)
            bar_total.set_description("Loss: {}".format(train_loss))
            bar_total.refresh()

            if self.epoch % self.summary_write == 0:
                accuracy = self.evaluate()
                self.summary.add_scalar('Train loss', train_loss, self.epoch)
                self.summary.add_scalar('Validation accuracy', accuracy, self.epoch)
                self.summary.close()

            if self.epoch % self.save_model == 0:
                self.save_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(args.resume)
        self.start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']

        self.model.load_state_dict(state_dict)
        print("Loading checkpoint: {} ...".format(args.resume))

    def save_checkpoint(self):
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        filename = str(self.save_dir + '/models/checkpoint-epoch{}.pth'.format(self.epoch))
        if self.tpu:
            xm.save(state, filename)
        else:
            torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def evaluate(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for data in self.evaluate_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #inputs = inputs.transpose(1, 3)
                predicted = self.model(inputs)
                _, predicted = torch.max(predicted, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Trainer')
    args.add_argument('-d', '--device', default='gpu', type=str)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('-s', '--save_dir', default='.', type=str)
    args.add_argument('-w', '--summary_write', default=10, type=int)
    args.add_argument('-m', '--save_model', default=1000, type=int)
    args.add_argument('-e', '--end_epoch', default=100000, type=int)

    args = args.parse_args()
    trainer = Trainer(args)

    if args.device == 'tpu':
        xmp.spawn(trainer.train(), args=())
    else:
        trainer.train()
