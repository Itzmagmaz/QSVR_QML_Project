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
import csv

IMAGES_DIR = Path(__file__).resolve().parent / "Scaled QML Images" / "Bloch PCA Baseline"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = IMAGES_DIR / "QSVR_Bloch_PCA_Baseline_Results.csv"

#for some reason, sklearn is not installed in the classiq environment, so we have to install it manually

data = fetch_california_housing(as_frame=True)
data2 = load_diabetes(as_frame=True)

X = data.data
X_std = (X - X.mean()) / X.std()#Standardization / dont understant what its doing


X2 = data2.data
X_std2 = (X2 - X2.mean()) / X2.std()




@qfunc
def Map(data: CArray[CReal], x: QArray[QBit]):
    inplace_encode_on_bloch(data, x)
# ML TIME
y = data.target
y2 = data2.target
def QSVR_PCA(model, target, k, seed):
    X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=seed) #split



    X_train = np.array(X_train, float)
    X_test  = np.array(X_test, float)
    y_train = np.array(y_train.values, float)
    y_test  = np.array(y_test.values, float)


    cov = np.cov(X_train.T) #Covariance matrix of the standardized data/ How data moves together
    #what does .T mean? (Transpose)


    eigenvalues, eigenvectors = np.linalg.eigh(cov) #Eigen decomposition of the covariance matrix

    explained = eigenvalues / eigenvalues.sum()

    vecs = eigenvectors[:, ::-1] #orders strongest first, weakest last

    X_train = X_train @ vecs[:, :k] #apperently creates a leak when done to the whole dataset
    X_test = X_test @ vecs[:, :k]


    scaler = MinMaxScaler(feature_range=(-1, 1))  # inplace_encode_on_bloch expects [-1, 1] and multiplies by pi itself

    X_train = scaler.fit_transform(X_train)   # learn min/max FROM TRAIN, then scale
    X_test  = scaler.transform(X_test)        # apply TRAIN's min/max to test

    PCA_weights = []

    for i in range(X_train.shape[1]):
        PCA_weights.append(1)  # unweighted baseline: one copy of each PCA component
        # PCA_weights.append(2 ** (X_train.shape[1] - i - 1))  # old 2^n weighting

    X_train =np.repeat(X_train, PCA_weights, axis=1)
    X_test =np.repeat(X_test, PCA_weights, axis=1)

    total_qubits = (X_train.shape[1] + 1) // 2

    if X_train.shape[1] % 2: # Bloch needs an even number of features (2 per qubit)
        X_train = np.pad(X_train, ((0,0),(0,1)))
        X_test  = np.pad(X_test, ((0,0),(0,1)))

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

    return mean_squared_error(y_test, preds), r2_score(y_test, preds), total_qubits

with RESULTS_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Encoding",
        "Seed",
        "PCA Components",
        "Physical Qubits",
        "MSE",
        "R2"
    ])

for seed in range(5):
    mse= []
    components = []
    R = []
    for g in range(2, 9):
        print("PCA with", g, "components")
        l = QSVR_PCA(X_std, y, g, seed)
        R.append(l[1])
        mse.append(l[0])
        components.append(g)

        with RESULTS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Bloch",
                seed,
                g,
                l[2],
                l[0],
                l[1]
            ])

    plt.figure()
    plt.plot( components,mse, marker='o')
    plt.title(f"Bloch encoding unweighted : QSVR with PCA seed({seed})")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean Squared Error")
    plt.savefig(IMAGES_DIR / f"qsvr_pca_bloch_sweep_seed_{seed}.png")
    plt.close()

    plt.figure()
    plt.plot( components,R, marker='o')
    plt.title(f"Bloch encoding unweighted : QSVR PCA R^2 score seed({seed})")
    plt.xlabel("Number of PCA components")
    plt.ylabel("R^2 score")
    plt.savefig(IMAGES_DIR / f"qsvr_pca_bloch_R2_seed_{seed}.png")
    plt.close()

