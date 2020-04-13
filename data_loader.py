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
            self.db_tables = self.db_tables[:50]
            _sql = " WHERE date < 20150101"
        else:
            self.db_tables = ['005930']
            _sql = " WHERE date > 20190101"

        for item in tqdm(self.db_tables, desc='Count num'):
            self.cursor.execute("SELECT COUNT(*) FROM _" + item + _sql)
            self.maria.commit()
            self.len += self.cursor.fetchone()[0] - 61

        self.data = []
        for item in tqdm(self.db_tables, desc='Retrieve all stock data'):
            self.cursor.execute("SELECT * FROM _" + item + _sql + " ORDER BY date ASC")
            self.maria.commit()
            self.data += list([open, high, low, close] for (_, open, high, low, close, volume) in self.cursor.fetchall())
        #self.data = np.log(self.data)
        #self.min_max_scaler = MinMaxScaler()
        #self.data = self.min_max_scaler.fit_transform(self.data)
        self.data = pd.DataFrame(self.data)
        self.data = np.log(self.data.pct_change() + 1)
        self.data = self.data[1:]

        self.data = self.data.to_numpy()


    def __getitem__(self, index):
        result = torch.FloatTensor(self.data[index: index + 61])
        return result[:-1], result[-1]

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