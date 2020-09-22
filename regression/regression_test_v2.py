import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

time_series = 182

def drop_columns(hist):
    return hist.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])

def train_size(num, df):
    return df.shape[0] - num

def create_sets(df):
    df_x = df['Date'].to_numpy().reshape(-1, 1)
    df_y = df['Close'].to_numpy().reshape(-1, 1)
    return df_x, df_y


data = yf.Ticker('MSFT')

data_hist = data.history(period="max")
df = drop_columns(data_hist)

new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(df)):
    new_data['Date'][i] = str(df.index[i]).split(' ')[0]
    new_data['Close'][i] = df['Close'][i]

# print(new_data)

train = new_data[:train_size(time_series, new_data)]
valid = new_data[train_size(time_series, new_data):]

x_train, y_train = create_sets(train)
x_valid, y_valid = create_sets(valid)

y_train = train.index


# print(x_train, y_train)

model = LinearRegression().fit(x_train, y_train)

preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)
