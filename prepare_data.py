import pandas as pd

import numpy as np

data = pd.read_csv("AAPL.csv")

close = data["Close"].values

wronggggg = close[2:]

close = close.astype("float")

X = []
y = []

sliding_window_size = 10

for i in range(sliding_window_size, len(close)):
    y.append(close[i])
    X.append(close[i-sliding_window_size:i])


X = np.array(X)
y = np.array(y)
