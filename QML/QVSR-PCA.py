from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from classiq import *
from classiq.applications.qsvm.qsvm import QSVM, QuantumKernelEvaluator
from classiq.applications.qsvm.quantum_feature_maps import pauli_feature_map
from classiq.open_library.functions.variational import inplace_encode_on_bloch
from sklearn.svm import SVR
import random
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True)
X = data.data
X_std = (X - X.mean()) / X.std()#Standardization / dont understant what its doing




# ML TIME
y = data.target

def QSVR_PCA(model, target, k):
    X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=0) #split



    X_train = np.array(X_train, float)
    X_test  = np.array(X_test, float)
    y_train = np.array(y_train.values, float)
    y_test  = np.array(y_test.values, float)


    cov = np.cov(X_train.T) #Covariance matrix of the standardized data/ How data moves together
    #what does .T mean? (Transpose) 

    eigenvalues, eigenvectors = np.linalg.eigh(cov) #Eigen decomposition of the covariance matrix

    vecs = eigenvectors[:, ::-1] #orders strongest first, weakest last

    X_train = X_train @ vecs[:, :k] #apperently creates a leak when done to the whole dataset
    X_test = X_test @ vecs[:, :k]


    scaler = MinMaxScaler(feature_range=(0, 2*np.pi))

    X_train = scaler.fit_transform(X_train)   # learn min/max FROM TRAIN, then scale
    X_test  = scaler.transform(X_test)        # apply TRAIN's min/max to test

    #mini split due to hardware limitations
    X_train, y_train = X_train[:60], y_train[:60]
    X_test,  y_test  = X_test[:20],  y_test[:20]

    N_DIM = X_train.shape[1]                 # 8 features -> 8 qubits
    PAULIS = [[Pauli.Z], [Pauli.Z, Pauli.Z]] # single-qubit Z rotations + two-qubit ZZ rotations
    AFFINES = [[1, 0], [1, np.pi]]           # how data values map into rotation angles
    CONNECTIVITY = 2                         # which qubit pairs get entangled (2 = full: all pairs)
    REPS = 2                                 # how many times to repeat the encoding block

    feature_map = lambda data, qba: pauli_feature_map(  #still dont understand lambda fucntions
        data, PAULIS, AFFINES, CONNECTIVITY, REPS, qba
    )

    N_DIM = X_train.shape[1]
    kernel = QuantumKernelEvaluator(feature_map=feature_map, num_qubits=N_DIM)

    K_train = kernel.kernel(X_train)            # (60, 60) — train vs train
    K_test  = kernel.kernel(X_test, X_train)    # (20, 60) — test vs train

    qsvr = SVR(kernel="precomputed")   # "precomputed" = "I'll hand you the kernel myself"
    qsvr.fit(K_train, y_train)         # learn from the quantum train kernel + real prices
    preds = qsvr.predict(K_test)       # predict on the test-vs-train kernel

    print("QSVR test MSE:", mean_squared_error(y_test, preds))

    print("K_train mean:", K_train.mean())
    print("K_train std :", K_train.std())
    print("K_train min :", K_train.min(), "max:", K_train.max())

    return mean_squared_error(y_test, preds)

mse= []
components = []

for g in range(2, 9):
    print("PCA with", g, "components")
    mse.append(QSVR_PCA(X_std, y, g))
    components.append(g)

plt.plot( components,mse, marker='o')
plt.title("QSVR with PCA")
plt.xlabel("Number of PCA components")
plt.ylabel("Mean Squared Error")
plt.savefig("qsvr_pca_sweep.png")