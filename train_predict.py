import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X = np.load("X_train.npy")
y = np.load("y_train.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
y = y.view(-1, 1)

X = X.squeeze()

class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        return x

model = StockPredictor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())
    
    
    
model.eval()
with torch.no_grad():
    last_window = X[-1]
    last_window = last_window.unsqueeze(0)

    predicted_price = model(last_window)

    prediction_value = predicted_price.item()

print("Predicted next price:", prediction_value)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from prepare_data import scaler

prediction = prediction_value

prediction = np.array([[prediction]])

real_price = scaler.inverse_transform(prediction)

print("Predicted next price:", real_price[0][0])

model.eval()

X = np.load("X_test.npy")
y = np.load("y_test.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
y = y.view(-1, 1)

X = X.squeeze()

with torch.no_grad():
    predictions = model(X)

predictions = predictions.numpy()

predictions = scaler.inverse_transform(predictions)
real_y = scaler.inverse_transform(y.numpy())

import matplotlib.pyplot as plt

plt.plot(real_y, label="Real Price")
plt.plot(predictions, label="Predicted Price")

plt.legend()
plt.title("Stock Price Prediction")
plt.show()