from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from classiq import *
from classiq.applications.qsvm.qsvm import QSVM, QuantumKernelEvaluator
from classiq.applications.qsvm.quantum_feature_maps import pauli_feature_map
from classiq.open_library.functions.variational import inplace_encode_on_bloch
from sklearn.svm import SVR
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "Images"
IMAGES_DIR.mkdir(exist_ok=True)

data_path = Path(__file__).resolve().parent.parent / "ML" / "synthetic_dataset.npy"
data = np.load(data_path)
#for some reason, sklearn is not installed in the classiq environment, so we have to install it manually


X = data
X_std = (X - X.mean()) / X.std()#Standardization / dont understant what its doing



# ML TIME
y = data.sum(axis=1)

def QSVR_PCA(model, target, k):
    X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=0) #split



    X_train = np.array(X_train, float)
    X_test  = np.array(X_test, float)
    y_train = np.array(y_train, float)
    y_test  = np.array(y_test, float)


    train_mean = X_train.mean(axis=0)
    
    X_train_centered = X_train - train_mean
    X_test_centered  = X_test - train_mean

    cov = np.cov(X_train_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov) #Eigen decomposition of the covariance matrix

    vecs = eigenvectors[:, ::-1] #orders strongest first, weakest last

    X_train = X_train_centered @ vecs[:, :k] #I Dont fully undertand the centering
    X_test  = X_test_centered @ vecs[:, :k] 


    scaler = MinMaxScaler(feature_range=(0, 2*np.pi))

    X_train = scaler.fit_transform(X_train)   # learn min/max FROM TRAIN, then scale
    X_test  = scaler.transform(X_test)        # apply TRAIN's min/max to test

    #mini split due to hardware limitations
    X_train, y_train = X_train[:60], y_train[:60]
    X_test,  y_test  = X_test[:20],  y_test[:20]

    N_DIM = X_train.shape[1]                 # 8 features -> 8 qubits
    PAULIS = [[Pauli.Z]] # single-qubit Z rotations                               THIS CAUSED LOTS OF PROBLEMS
    AFFINES = [[1, 0], [1, np.pi]]           # how data values map into rotation angles
    CONNECTIVITY = 2                          # which qubit pairs get entangled (2 = full: all pairs)
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

    return mean_squared_error(y_test, preds), r2_score(y_test, preds)

mse= []
components = []
R = []
for g in range(2, 9):
    print("PCA with", g, "components")
    l = QSVR_PCA(X_std, y, g)
    R.append(l[1])
    mse.append(l[0])
    components.append(g)

plt.figure()
plt.plot( components,mse, marker='o')
plt.title("QSVR with PCA")
plt.xlabel("Number of PCA components")
plt.ylabel("Mean Squared Error")
plt.savefig(IMAGES_DIR / "qsvr_pca_sweep_SynthZ.png")


