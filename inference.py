
import argparse
import torch
import importlib

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch import nn

import pandas as pd
from model import Test_Model, Model

from tqdm import tqdm, trange

import pymysql

tpu_check = importlib.find_loader('torch_xla')

if tpu_check is not None:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    from tqdm._tqdm_notebook  import tqdm, trange
else:
    matplotlib.use('TkAgg')

config = {

    "user": "root",
    "password": "5123",
    "host": "15.164.49.81",
    "port": 3306
}


class inference:

    def __init__(self, args):
        self.maria = pymysql.connect(**config)
        self.cursor = self.maria.cursor()
        self.cursor.execute("USE daily_stock")
        self.cursor.execute("show tables")
        self.maria.commit()

        self.db_tables = list(i[1:] for (i, ) in self.cursor.fetchall())
        self.db_tables = self.db_tables[:50]
        self.db_tables = ['005930']
        from_date = '20191001'
        to_date = '20200420'

        self.cursor.execute("SELECT date FROM _005930 WHERE " + from_date + " < date and date < " + to_date + " ORDER BY date ASC")
        self.db_dates = list(i.strftime('%Y%m%d') for (i, ) in self.cursor.fetchall())

        if args.device == 'tpu':
            self.device = xm.xla_device()
            dir = 'drive/My Drive/PF/'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dir = ''

        checkpoint = torch.load(dir + 'models/checkpoint-epoch3000.pth', map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']

        self.model = Model(8, 256, self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.money = 1
        self.mean_money = 1
        self.value_list = [self.money]
        self.mean_value_list = [self.mean_money]
        self.real_yield_list = []
        self.pred_yield_list = []
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test(self):
        print('Data Length: {}'.format(len(self.db_dates) - 62))
        print('Date: from {} to {}'.format(self.db_dates[62], self.db_dates[-1]))

        with torch.no_grad():
            for i in trange(len(self.db_dates) - 63, leave=False):
                data = []
                real = []
                stocks_list = []

                for item in tqdm(self.db_tables, leave=False):

                    self.cursor.execute("SELECT * FROM _" + item + " WHERE " + self.db_dates[i] + " < date and date < " +
                                   self.db_dates[i + 63] + " ORDER BY date ASC")
                    self.maria.commit()
                    _data = list([open, high, low, close, volume, listed_stocks, credit_of_stocks, credit_of_volume, foreign_ratio]
                                                                        for (date, open, high, low, close, volume,
                                                                       credit_of_stocks, credit_of_volume, foreign_ratio,
                                                                       listed_stocks) in self.cursor.fetchall())
                    _data = np.array(_data)

                    if np.shape(_data)[0] == 62 and not 0 in _data[:, -1]:
                        #_data = _data[:, :-1]
                        _data = pd.DataFrame(_data)
                        volume = _data[4] / _data[5]
                        volume = volume[1:]
                        credit = _data[6]
                        credit = credit[1:]
                        credit_of_volume = _data[7][1:]
                        foreign_ratio = _data[8][1:]
                        _data = _data.drop([4, 5, 6, 7, 8], axis='columns')
                        _data = _data.pct_change()
                        _data = _data[1:]
                        _data = pd.concat([_data, volume, credit, credit_of_volume, foreign_ratio],  axis=1)
                        _data = _data.to_numpy()
                        #_data = np.log(_data)

                        data.append(_data[:-1])
                        real.append(_data[-1])
                        stocks_list.append(item)

                data = torch.FloatTensor(data).to(self.device)
                output = self.model(data)
                output = np.array(output)
                real_close = (real[output[:, 0].argmax()][3])

                self.mean_money *= (np.mean(np.array(real)[:, 3]) + 1) #1
                aa = 0
                bb = output[output[:, 0].argmax(), 1]
                if bb >= 0.9:
                    aa = 0
                elif bb >= 0.8:
                    aa = 1
                elif bb >= 0.7:
                    aa = 2
                elif bb >= 0.6:
                    aa = 3
                elif bb >= 0.5:
                    aa = 4
                elif bb >= 0.4:
                    aa = 5
                elif bb >= 0.3:
                    aa = 6
                elif bb >= 0.2:
                    aa = 7
                elif bb >= 0.1:
                    aa = 8
                elif bb > 0.:
                    aa = 9
                if real_close > 0:
                    self.t[aa] += 1
                else:
                    self.t[aa] -= 1

                # and output[output[:, 0].argmax(), 1] > 0.5
                print( output[:, 0].max())
                if output[:, 0].max() > 0 : #data[output.argmax(), -1, -1]
                        self.money *= (real_close + 1) #1

                        self.pred_yield_list.append(output[:, 0].max())
                        self.real_yield_list.append(real_close)

                        if real_close > 0:
                            self.TP += 1
                        else:
                            self.TN += 1
                else:
                    if real_close > 0:
                        self.FP += 1
                    else:
                        self.FN += 1

                self.mean_value_list.append(self.mean_money)
                self.value_list.append(self.money)

        print(self.TP, self.TN, self.FP, self.FN)

        plt.figure(1)
        plt.title("Money")
        plt.xticks(rotation=90)
        plt.plot(self.db_dates[62:], self.mean_value_list, color='g')
        plt.plot(self.db_dates[62:], self.value_list, color='b')
        plt.legend()

        plt.figure(2)
        plt.title("Yield")
        plt.plot(self.real_yield_list, color='g')
        plt.plot(self.pred_yield_list, color='b')
        plt.legend()
        plt.show()

        plt.figure(3)
        plt.title("ㅅㄱㄱㄱ")
        plt.plot(self.t)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Trainer')
    args.add_argument('-d', '--device', default='cpu', type=str)
    args = args.parse_args()

    instance = inference(args)

    if args.device == 'tpu':
        xmp.spawn(instance.test(), args=())
    else:
        instance.test()
