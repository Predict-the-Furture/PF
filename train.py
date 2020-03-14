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

    model = Model(6, 60, 4)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

        accuracy = evaluate(model, evaluate_loader, criterion)
        summary.add_scalar('loss', accuracy, epoch * len(train_loader) + i)
        print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
        summary.close()

def evaluate(model, validation_loader, criterion):
    with torch.no_grad():
        acc = .0
        iterations = 0
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            predicted = model(inputs)
            loss = criterion(predicted, labels)
            acc += loss
            iterations += 1
    return acc / iterations

if __name__ == '__main__':
    train()