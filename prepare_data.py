import pandas as pd

from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("AAPL.csv")

close = data["Close"].values

close = close[2:]

close = close.astype("float")

X = []
y = []

close = close.reshape(-1, 1)

scaler = MinMaxScaler()

close = scaler.fit_transform(close)


sliding_window_size = 10

for i in range(sliding_window_size, len(close)):
    y.append(close[i][0])
    X.append(close[i-sliding_window_size:i])


import numpy as np
X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)


