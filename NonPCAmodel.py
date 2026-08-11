from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

data = fetch_california_housing(as_frame=True)
X = data.data

X = (X - X.mean()) / X.std() #Standardization / dont understant what its doing
# ML TIME
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #split


#The loop
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32)

y_train_t = y_train_t.reshape(-1, 1)
y_test_t  = y_test_t.reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_values = []

for epoch in range(2000):
    preds = model(X_train_t)
    loss = loss_fn(preds, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())
        loss_values.append(loss.item())

with torch.no_grad():
    test_preds = model(X_test_t)
    test_loss = loss_fn(test_preds, y_test_t)
    print(test_loss.item())


plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
