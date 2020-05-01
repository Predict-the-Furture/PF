import os
import time
import importlib
import argparse
import torch

from datetime import datetime
from pytz import timezone
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import folder
from model import Model, Test_Model
from data_loader import DiabetesDataset

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
        
        torch.manual_seed(777)

        self.save_dir = args.save_dir
        self.start_epoch = 0
        self.summary_write = args.summary_write
        self.save_model = args.save_model
        self.end_epoch = args.end_epoch

        folder([self.save_dir, self.save_dir + '/models', self.save_dir + '/runs'])

        self.summary = SummaryWriter(self.save_dir + '/runs/' + datetime.now(timezone).strftime("%Y-%m-%d-%H%M"))

        self.dataset_train = DiabetesDataset(train=True)
        self.dataset_test = DiabetesDataset(train=False)
        self.train_loader = DataLoader(dataset=self.dataset_train, batch_size=128, shuffle=True, num_workers=0)
        self.evaluate_loader = DataLoader(dataset=self.dataset_test, batch_size=128, shuffle=False, num_workers=0)

        self.model = Model(8, 256, self.device)

        self.model = self.model.to(self.device)

        if args.resume is not None:
            self.load_checkpoint()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)

    def train(self):
        bar_total = tqdm(range(self.start_epoch, self.end_epoch), desc='Training', leave=False)
        n_samples = len(self.train_loader.sampler)
        for self.epoch in bar_total:
            total_loss = 0
            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

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
                self.summary.add_scalar('Validation loss', accuracy, self.epoch)
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
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.
            for data in self.evaluate_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted = self.model(inputs)
                loss = self.criterion(predicted, labels)

                batch_size = inputs.shape[0]
                total_loss += loss.item() * batch_size

        n_samples = len(self.evaluate_loader.sampler)
        self.model.train()
        return total_loss / n_samples


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Trainer')
    args.add_argument('-d', '--device', default='cpu', type=str)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('-s', '--save_dir', default='.', type=str)
    args.add_argument('-w', '--summary_write', default=100, type=int)
    args.add_argument('-m', '--save_model', default=1000, type=int)
    args.add_argument('-e', '--end_epoch', default=100000, type=int)

    args = args.parse_args()
    trainer = Trainer(args)

    if args.device == 'tpu':
        xmp.spawn(trainer.train(), args=())
    else:
        trainer.train()
