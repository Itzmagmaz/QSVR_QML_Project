from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split


np.random.seed(0)

eigenvalues_sorted = [.5]#orders strongest first, weakest last
eigenvectors = [1] #does not do anything currently 
for i in range(1, 9):
    eigenvalues_sorted.append(eigenvalues_sorted[i-1]/2)
    eigenvectors.append(1) 

eigenvalues_sorted = np.array(eigenvalues_sorted) #make it easier later for the matrix multiplication

eigenvalues_sorted = np.diag(eigenvalues_sorted)
eigenvectors = np.diag(eigenvectors)

dataset = np.empty((1000, 9))#initialize empty dataset with 9 features 


for i in range(1000):
    datapoint = eigenvectors @ np.sqrt(eigenvalues_sorted) @ np.random.normal(0, 1, 9) # x = Q*sqrt(lambda)*z
    dataset[i]= datapoint.reshape(1, 9) #append the new datapoint to the dataset


np.save(Path(__file__).resolve().with_name("synthetic_dataset.npy"), dataset)
print("Synthetic dataset saved as synthetic_dataset.npy")
