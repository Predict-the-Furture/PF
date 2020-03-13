import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import Model
from data_loader import DiabetesDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    summary = SummaryWriter()
    dataset = DiabetesDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)
    evaluate_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)

    model = Model(6, 60, 4).to(device)

    criterion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(200):
        print('-------------------------------------------------------')
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, i, loss.item())

        summary.add_scalar('loss', evaluate(model, evaluate_loader), epoch * len(train_loader) + i)
        summary.close()

def evaluate(model, validation_loader, use_cuda=False):
    model.eval()
    with torch.no_grad():
        acc = .0
        for i, data in enumerate(validation_loader):
            X = data[0]
            y = data[1]
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            predicted = model(X)
            acc+=(predicted.round() == y).sum()/float(predicted.shape[0])
    model.train()
    return (acc/(i+1)).detach().item()

if __name__ == '__main__':
    train()