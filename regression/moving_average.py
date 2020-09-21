import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

time_series = 182

def drop_columns(hist):
    hist = hist.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
    return hist

def train_size(num, df):
    length = df.shape[0] - num
    return length


data = yf.Ticker('MSFT')

data_hist = data.history(period="max")
print(data_hist.head())
df = drop_columns(data_hist)
print(df.shape)
train = df[:train_size(time_series, df)]
valid = df[train_size(time_series, df):]

preds = []

for i in range(0, valid.shape[0]):
    a = train['Close'][len(train)-time_series+i:].sum() + sum(preds)
    b = a/time_series
    preds.append(b)
    print(f'a = {a}, b = {b}')


rms = np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds), 2)))
print(f'RMSE value: {rms}')

df_2 = pd.DataFrame(index=valid.index, data=preds, columns=['Close'])

plt.plot(train['Close'])
plt.plot(df_2)
plt.plot(valid['Close'])
plt.ylabel('Stock Price ($)')
plt.xlabel('Date')
plt.legend(['Train', 'Prediction', 'Actual'])
plt.show()






