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
import csv

IMAGES_DIR = Path(__file__).resolve().parent / "Scaled QML Images" / "Synth Bloch PCA Baseline"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = IMAGES_DIR / "QSVR_Synth_Bloch_PCA_Baseline_Results.csv"

data_path = Path(__file__).resolve().parent.parent / "ML" / "synthetic_powerlaw3_dataset.npy"
data = np.load(data_path)
#for some reason, sklearn is not installed in the classiq environment, so we have to install it manually


X = data
X_std = (X - X.mean()) / X.std()#Standardization / dont understant what its doing



@qfunc
def Map(data: CArray[CReal], x: QArray[QBit]):
    inplace_encode_on_bloch(data, x)

# ML TIME
y = data.sum(axis=1)

def QSVR_PCA(model, target, k, seed):
    X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=seed) #split



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


    scaler = MinMaxScaler(feature_range=(-1, 1))  # inplace_encode_on_bloch expects [-1, 1] and multiplies by pi itself

    X_train = scaler.fit_transform(X_train)   # learn min/max FROM TRAIN, then scale
    X_test  = scaler.transform(X_test)        # apply TRAIN's min/max to test

    total_qubits = (X_train.shape[1] + 1) // 2  # unweighted: one copy of each PCA component, 2 per qubit

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
    plt.title(f"Synthetic data Bloch encoding unweighted : QSVR with PCA seed({seed})")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean Squared Error")
    plt.savefig(IMAGES_DIR / f"qsvr_pca_bloch_sweep_Synth_seed_{seed}.png")
    plt.close()


    plt.figure()
    plt.plot( components,R, marker='o')
    plt.title(f"Synthetic data Bloch encoding unweighted : QSVR PCA R^2 score seed({seed})")
    plt.xlabel("Number of PCA components")
    plt.ylabel("R^2 score")
    plt.savefig(IMAGES_DIR / f"qsvr_pca_bloch_R2_Synth_seed_{seed}.png")
    plt.close()
