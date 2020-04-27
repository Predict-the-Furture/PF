from pykrx import stock

df = stock.get_shorting_balance_by_ticker("20000101", "20190405", "005930")
print(df.head())