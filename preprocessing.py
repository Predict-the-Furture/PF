import os
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process
from tqdm import tqdm
from mpl_finance import candlestick2_ochl

config = {

    "user": "root",
    "password": "5123",
    "host": "15.164.49.81",
    "port": 3306
}

#dir = 'drive/My Drive/PF/'
dir = ''
dimension = 50
dpi = 96

maria = pymysql.connect(**config)
cursor = maria.cursor()
cursor.execute("USE daily_stock")
cursor.execute("show tables")
maria.commit()

db_tables = list(i[1:] for (i, ) in cursor.fetchall())

def main():
    procs = []

    for item in db_tables:
        proc = Process(target=work, args = (item, ))
        procs.append(proc)
        proc.start()

    for proc in tqdm(procs):
        proc.join()

    print("Converting ohlc to candlestick finished.")

def work(item):
    if not os.path.exists('{}dataset/{}'.format(dir, item)):
        os.mkdir('{}dataset/{}'.format(dir, item))

    _data = []

    cursor.execute("SELECT * FROM _" + item + " ORDER BY date ASC")
    maria.commit()
    _data = list([date, open, high, low, close, volume, listed_stocks, credit_of_stocks, credit_of_volume, foreign_ratio]
                 for (date, open, high, low, close, volume,
                      credit_of_stocks, credit_of_volume, foreign_ratio,
                      listed_stocks) in cursor.fetchall())

    #self.data = np.log(self.data)
    #self.min_max_scaler = MinMaxScaler()
    #self.data = self.min_max_scaler.fit_transform(self.data)
    _data = pd.DataFrame(_data)

    for i in range(0, len(_data) - 22):
        c = _data.iloc[i:i + 23, :]
        today = c.iloc[20, 4]
        tomorrow = c.iloc[-1, 4]
        date = c.iloc[-1, 0]
        plt.style.use('dark_background')
        if today < tomorrow:
            t = 1
        else:
            t = 0
        pngfile = '{}dataset/{}/{}_{}.png'.format(dir, item, date, t)
        if not os.path.exists(pngfile):
            c = c.iloc[:20, :]
            fig = plt.figure(figsize=(dimension / dpi, dimension / dpi), dpi=dpi)
            ax1 = fig.add_subplot(1, 1, 1)
            #candlestick_ohlc(ax1, [c[0], c[1], c[2], c[3]], width=1)
            candlestick2_ochl(ax1, c[1], c[4], c[2],
                              c[3], width=1,
                              colorup='#77d879', colordown='#db3f3f')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis('off')

            fig.savefig(pngfile, pad_inches=0, transparent=False)
            plt.close(fig)

if __name__ == '__main__':
    main()