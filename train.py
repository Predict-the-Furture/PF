import torch

from datetime import datetime
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Model
from data_loader import DiabetesDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self):
        self.checkpoint_dir = './saved'

        self.summary = SummaryWriter('./runs/' + datetime.today().strftime("%Y-%m-%d-%H%M%S"))

        self.dataset = DiabetesDataset()
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=64, shuffle=True, num_workers=0)
        self.evaluate_loader = DataLoader(dataset=self.dataset, batch_size=64, shuffle=True, num_workers=0)

        self.model = Model(6, 60, 4)
        self.model = self.model.to(device)

        self.criterion =nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def train(self):

        for self.epoch in range(200):
            print('-------------------------------------------------------')
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                y_pred = self.model(inputs)

                loss = self.criterion(y_pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            accuracy = self.evaluate()
            self.summary.add_scalar('loss', accuracy, self.epoch * len(self.train_loader) + i)
            print("Epoch: {}, Accuracy: {}".format(self.epoch, accuracy))
            self.summary.close()

            if self.epoch % 100 == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(self.epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.
            for i, data in enumerate(self.evaluate_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                predicted = self.model(inputs)
                loss = self.criterion(predicted, labels)

                batch_size = inputs.shape[0]
                total_loss += loss.item() * batch_size

        n_samples = len(self.evaluate_loader.sampler)
        self.model.train()
        return total_loss / n_samples

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()