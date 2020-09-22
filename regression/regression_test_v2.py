import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

time_series = 182

def drop_columns(hist):
    hist = hist.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
    return hist

def train_size(num, df):
    length = df.shape[0] - num
    return length

def create_sets(df):
    df_train = df.index.to_numpy().reshape(-1, 1)
    for i in df_train:
        print(i)
    df_valid = df['Close'].to_numpy().reshape(-1, 1)
    return df_train, df_valid


data = yf.Ticker('MSFT')

data_hist = data.history(period="max")
df = drop_columns(data_hist)
train = df[:train_size(time_series, df)]
valid = df[train_size(time_series, df):]

x_train, y_train = create_sets(train)
x_valid, y_valid = create_sets(valid)
print(x_train, y_train)

model = LinearRegression().fit(x_train, y_train)

preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)



