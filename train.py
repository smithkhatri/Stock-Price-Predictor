import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

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
    
