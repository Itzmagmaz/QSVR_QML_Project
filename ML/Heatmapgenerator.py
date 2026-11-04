from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pathlib import Path
import csv


IMAGES_DIR = Path(__file__).resolve().parent.parent / "Images" / "Powerlaw images"
IMAGES_DIR.mkdir(exist_ok=True)

RESULTS_CSV = IMAGES_DIR / "QSVR_Powerlaw_Bloch_NoRotation_Results.csv"

# lowest MSE for each (data exponent, qubit exponent) combination, across all PCA component counts
best = {}
with RESULTS_CSV.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (float(row["Data Exponent"]), float(row["Qubit Exponent"]))
        mse = float(row["MSE"])
        if key not in best or mse < best[key][0]:
            best[key] = (mse, int(row["PCA Components"]))

data_exponents = sorted({d for d, q in best})
qubit_exponents = sorted({q for d, q in best})

grid = np.full((len(data_exponents), len(qubit_exponents)), np.nan)
for (d, q), (mse, k) in best.items():
    grid[data_exponents.index(d), qubit_exponents.index(q)] = mse

plt.figure()
plt.imshow(grid, cmap="viridis_r", origin="lower")  # reversed so yellow = lower MSE
plt.colorbar(label="Lowest MSE")
plt.xticks(range(len(qubit_exponents)), qubit_exponents)
plt.yticks(range(len(data_exponents)), data_exponents)
plt.xlabel("Qubit Exponent")
plt.ylabel("Data Exponent")
plt.title("Bloch powerlaw: lowest MSE per combination")

middle = (np.nanmin(grid) + np.nanmax(grid)) / 2
for (d, q), (mse, k) in best.items():
    plt.text(qubit_exponents.index(q), data_exponents.index(d), f"{mse:.3f}\nk={k}",
             ha="center", va="center", color="black" if mse < middle else "white", fontsize=8)  # dark text on light cells

plt.savefig(IMAGES_DIR / "qsvr_powerlaw_bloch_lowest_mse_heatmap.png")
plt.close()

