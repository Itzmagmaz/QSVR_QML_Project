from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)

eigenvalues_sorted = [.5]#orders strongest first, weakest last
eigenvectors = [1] #does not do anything currently 
powerlaw_exponent = 1.5 #controls the decay of the eigenvalues
for i in range(1, 9):
    eigenvalues_sorted.append(.5 / (i + 1) ** powerlaw_exponent)#orders strongest first, weakest last
    eigenvectors.append(1) 

eigenvalues_sorted = np.array(eigenvalues_sorted) #make it easier later for the matrix multiplication

eigenvalues_sorted = np.diag(eigenvalues_sorted)
eigenvectors = np.diag(eigenvectors)

dataset = np.empty((1000, 9))#initialize empty dataset with 9 features 


for i in range(1000):
    datapoint = eigenvectors @ np.sqrt(eigenvalues_sorted) @ np.random.normal(0, 1, 9) # x = Q*sqrt(lambda)*z
    dataset[i]= datapoint.reshape(1, 9) #append the new datapoint to the dataset


np.save("synthetic_powerlaw_dataset.npy", dataset)
print("Synthetic dataset saved as synthetic_powerlaw_dataset.npy")