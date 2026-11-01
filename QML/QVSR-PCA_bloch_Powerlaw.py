from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from classiq import *
from classiq.applications.qsvm.qsvm import QSVM, QuantumKernelEvaluator
from classiq.open_library.functions.variational import encode_on_bloch
from classiq.open_library.functions.variational import inplace_encode_on_bloch
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "Images"
IMAGES_DIR.mkdir(exist_ok=True)

#for some reason, sklearn is not installed in the classiq environment, so we have to install it manually

data = fetch_california_housing(as_frame=True)

data_path = Path(__file__).resolve().parent.parent / "ML" / "synthetic_powerlaw3_dataset.npy"
data = np.load(data_path)


X = data
X_std = (X - X.mean()) / X.std()#Standardization / dont understant what its doing

@qfunc
def Map(data: CArray[CReal], x: QArray[QBit]):
    inplace_encode_on_bloch(data, x)
# ML TIME
y = data.sum(axis=1)

def QSVR_PCA(model, target,k ):
    X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=0) #split



    X_train = np.array(X_train, float)
    X_test  = np.array(X_test, float)
    y_train = np.array(y_train, float)
    y_test  = np.array(y_test, float)


    cov = np.cov(X_train.T) #Covariance matrix of the standardized data/ How data moves together
    #what does .T mean? (Transpose)


    eigenvalues, eigenvectors = np.linalg.eigh(cov) #Eigen decomposition of the covariance matrix

    explained = eigenvalues / eigenvalues.sum()

    vecs = eigenvectors[:, ::-1] #orders strongest first, weakest last

    X_train = X_train @ vecs[:, :k] #apperently creates a leak when done to the whole dataset
    X_test = X_test @ vecs[:, :k]


    scaler = MinMaxScaler(feature_range=(0, 2*np.pi))

    X_train = scaler.fit_transform(X_train)   # learn min/max FROM TRAIN, then scale
    X_test  = scaler.transform(X_test)        # apply TRAIN's min/max to test

    PCA_weights = []
    powerlaw_exponent = 2  # controls the decay of the eigenvalues
    for i in range(X_train.shape[1]):
        PCA_weights.append(round((X_train.shape[1] / (i + 1)) ** powerlaw_exponent))

    X_train =np.repeat(X_train, PCA_weights, axis=1)
    X_test =np.repeat(X_test, PCA_weights, axis=1)

    if X_train.shape[1] % 2:
        X_train = np.pad(X_train, ((0,0),(0,1)))
        X_test  = np.pad(X_test,  ((0,0),(0,1)))


    #mini split due to hardware limitations
    X_train, y_train = X_train[:60], y_train[:60]
    X_test,  y_test  = X_test[:20],  y_test[:20]

    feature_map = Map
    N_DIM = X_train.shape[1]
    
    kernel = QuantumKernelEvaluator(feature_map=feature_map, num_qubits= X_train.shape[1] // 2)

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
plt.title("powerlaw weights Synthetic data Bloch encoding with weights : QSVR with PCA")
plt.xlabel("Number of PCA components")
plt.ylabel("Mean Squared Error")
plt.savefig(IMAGES_DIR / "qsvr_pca_powerlawweights_bloch.png")

