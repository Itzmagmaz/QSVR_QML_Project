
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

data = np.load("synthetic_dataset.npy")
X = data

X_std = (X - X.mean()) / X.std() #Standardization / dont understant what its doing
#print(X_std.mean().round(4)) #should be 0
#print(X_std.std().round(4)) # should be 1

cov = np.cov(X_std.T) #Covariance matrix of the standardized data/ How data moves together
#what does .T mean? (Transpose) 

eigenvalues, eigenvectors = np.linalg.eigh(cov) #Eigen decomposition of the covariance matrix

explained = eigenvalues / eigenvalues.sum() #weighed contribution of each eigenvalue to the total variance in the data

eigenvectors_sorted = eigenvectors[:, ::-1] #orders strongest first, weakest last

X_2d = X_std @ eigenvectors_sorted[:, :2]   # keep first 2 components
X_9d = X_std @ eigenvectors_sorted[:, :9]   # keep first 9
print(eigenvalues[::-1])
print(explained[::-1])
# ML TIME
y = data.sum(axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_9d, y, test_size=0.2, random_state=0) #split


#The loop
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

y_train_t = y_train_t.reshape(-1, 1)
y_test_t  = y_test_t.reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    preds = model(X_train_t)
    loss = loss_fn(preds, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    test_preds = model(X_test_t)
    test_loss = loss_fn(test_preds, y_test_t)
    print(test_loss.item())

#PCA print
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.3)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("California Housing — PCA to 2D")
plt.show()

#PCA MSE thing
losses = []
for g in range(2,10):
    X_g = X_std @ eigenvectors_sorted[:, :g]   # keep first g components
    X_train, X_test, y_train, y_test = train_test_split(X_g, y, test_size=0.2, random_state=0) #split

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    y_train_t = y_train_t.reshape(-1, 1)
    y_test_t  = y_test_t.reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(g, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())

    with torch.no_grad():
        test_preds = model(X_test_t)
        test_loss = loss_fn(test_preds, y_test_t)
        losses.append(test_loss.item())
        print(test_loss.item())

plt.figure()
plt.plot(range(2,10), losses, marker='o')
plt.title("MSE vs PCA components")
plt.xlabel("Number of PCA components")
plt.ylabel("Mean Squared Error")
plt.savefig("ML_SYNTH_PCA_MSE.png")