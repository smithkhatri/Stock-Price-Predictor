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

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

