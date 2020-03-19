import time
import importlib
import argparse
import torch

from datetime import datetime
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Model
from data_loader import DiabetesDataset

tpu_check = importlib.find_loader('torch_xla')
if tpu_check is not None:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

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

        self.checkpoint_dir = 'saved'

        self.summary = SummaryWriter('runs/' + datetime.today().strftime("%Y-%m-%d-%H%M%S"))

        self.dataset = DiabetesDataset()
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=64, shuffle=True, num_workers=0)
        self.evaluate_loader = DataLoader(dataset=self.dataset, batch_size=64, shuffle=True, num_workers=0)

        self.model = Model(6, 60, 4, self.device)
        self.model = self.model.to(self.device)

        self.criterion =nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def train(self):
        start = time.time()
        for self.epoch in range(1000):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                y_pred = self.model(inputs)

                loss = self.criterion(y_pred, labels)
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                if self.tpu == False:
                    self.optimizer.step()
                else:
                    xm.optimizer_step(self.optimizer)
                    xm.mark_step()

            if (self.epoch + 1) % 50 == 0:
                accuracy = self.evaluate()
                self.summary.add_scalar('loss', accuracy, self.epoch + 1)
                print("{} epoch, accuracy: {} in {}s".format(self.epoch + 1, accuracy, time.time() - start))
                start = time.time()
                self.summary.close()

            if (self.epoch + 1) % 100 == 0:
                self.save_checkpoint()


    def save_checkpoint(self):
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.to('cpu').state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(self.epoch + 1))
        torch.save(state, filename)
        self.model.to(self.device)
        print("Saving checkpoint: {} ...".format(filename))

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.
            for i, data in enumerate(self.evaluate_loader):
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

    args = args.parse_args()
    trainer = Trainer(args)

    trainer.train()