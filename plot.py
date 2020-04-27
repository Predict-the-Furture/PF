import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm
from model import Test_Model, Model
from data_loader import DiabetesDataset, load_test_stocks
from utils import vstack

from mpl_finance import candlestick_ohlc

matplotlib.use('TkAgg')


dataset = DiabetesDataset(train=False)
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('models/checkpoint-epoch3000.pth', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']

model = Test_Model(4, 512, device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

total_loss = 0.0
criterion = nn.MSELoss()

batch_real_data = []
batch_predicted = []

with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(data_loader)):

        output = model(data)

        loss = criterion(output, target)

        batch_real_data.append(target.numpy())
        batch_predicted.append(output.numpy())

        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size

    n_samples = len(data_loader.sampler)
    loss = total_loss / n_samples
    print('{} Loss: {}'.format(total_loss, loss))

real_data = vstack(batch_real_data)
predicted = vstack(batch_predicted)

#real_data = dataset.min_max_scaler.inverse_transform(real_data)
#predicted = dataset.min_max_scaler.inverse_transform(predicted)

real_data = np.exp(real_data)
predicted = np.exp(predicted)

print(np.shape(real_data), np.shape(predicted))

#real_data = list(i[0] for i in real_data)
#predicted = list(i[0] for i in predicted)

fig = plt.figure(figsize=(8, 5))
fig.set_facecolor('w')
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
axes = []
axes.append(plt.subplot(gs[0]))
axes.append(plt.subplot(gs[1], sharex=axes[0]))
axes[0].get_xaxis().set_visible(False)

arange = list([i] for i in range(np.shape(real_data)[0]))
candlestick_ohlc(axes[0], np.hstack([arange, real_data]), colorup='r', colordown='b', alpha=.5)
candlestick_ohlc(axes[0], np.hstack([arange, predicted]), colorup='forestgreen', colordown='navy', alpha=.2)

#axes[1].bar(np.arange(real_data.shape[0]), real_data[:, 4], color='r', align='center', alpha=.5)
#axes[1].bar(np.arange(predicted.shape[0]), predicted[:, 4], color='g', align='center', alpha=.5)
plt.tight_layout()
plt.legend()
plt.show()