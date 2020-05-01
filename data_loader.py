import sys

import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import pymysql

config = {

    "user": "root",
    "password": "5123",
    "host": "15.164.49.81",
    "port": 3306
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DiabetesDataset(Dataset):
    def __init__(self, train=False):

        self.maria = pymysql.connect(**config)
        self.cursor = self.maria.cursor()
        self.cursor.execute("USE daily_stock")
        self.cursor.execute("show tables")
        self.maria.commit()
        self.len = 0

        if train == True:
            self.db_tables = list(i[1:] for (i, ) in self.cursor.fetchall())
            self.db_tables = self.db_tables[:15]
            #self.db_tables = ['005930', '035720', '015760']


            _sql = " WHERE date < 20190101"
        else:
            self.db_tables = ['005930']
            _sql = " WHERE date > 20190101"

        len_list = []

        self.data = []
        self.target = []

        _data = []
        for i, item in tqdm(enumerate(self.db_tables), desc='Retrieve all stock data'):
            self.cursor.execute("SELECT * FROM _" + item + _sql + " ORDER BY date ASC")
            self.maria.commit()
            _data = list([open, high, low, close, volume, listed_stocks, credit_of_stocks, credit_of_volume, foreign_ratio]
                                                        for (_, open, high, low, close, volume,
                                                       credit_of_stocks, credit_of_volume, foreign_ratio,
                                                       listed_stocks) in self.cursor.fetchall())

            #self.data = np.log(self.data)
            #self.min_max_scaler = MinMaxScaler()
            #self.data = self.min_max_scaler.fit_transform(self.data)
            _data = pd.DataFrame(_data)

            _data = _data[_data[4] != 0]
            volume = _data[4] / _data[5]
            volume = volume[1:]
            credit = _data[6]
            credit = credit[1:]
            credit_of_volume = _data[7][1:]
            foreign_ratio = _data[8][1:]
            _data = _data.drop([4, 5, 6, 7, 8], axis='columns')
            _data = np.log(_data.pct_change() + 1)
            _data = _data[1:]
            _data = pd.concat([_data, volume, credit, credit_of_volume, foreign_ratio],  axis=1)
            _data = _data.to_numpy()
            len_list.append(len(_data) - 61)
            x = []
            y = []
            for j in range(len_list[i]):
                temp = _data[j: j + 61]
                t = temp[-1, 3]
                #print(temp[:-1])
                x.append(temp[:-1])
                if t > 0:
                    t = [t, 1]
                else:
                    t = [t, 0]

                y.append(t)
            self.data.extend(x)
            self.target.extend(y)

        self.len = sum(len_list)
        self.data = np.array(self.data)
        self.target = np.array(self.target)
        self.data = torch.from_numpy(self.data).float()
        self.target = torch.from_numpy(self.target).float()

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len

def load_test_stocks():
    maria = pymysql.connect(**config)
    cursor = maria.cursor()
    cursor.execute("USE daily_stock")

    db_tables = ['000020']
    data = []

    for item in tqdm(db_tables, desc='Retrieve all stock data'):
        cursor.execute("SELECT * FROM _" + item + " ORDER BY date ASC")
        maria.commit()
        data += list([open, high, low, close, volume, amount] for (_, open, high, low, close, volume, amount) in cursor.fetchall())

    return data