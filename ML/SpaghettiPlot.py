import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent

# --- Bloch PCA Baseline ---


# group the rows by seed: {seed: [(PCA components, MSE), ...]}
def makegraph(RESULTS_CSV, FOLDER, weight, dataset):
    runs = {}
    with RESULTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            seed = int(row["Seed"])
            runs.setdefault(seed, []).append((int(row["PCA Components"]), float(row["MSE"])))

    plt.figure()
    for seed, points in sorted(runs.items()):
        points.sort()
        components = [p[0] for p in points]
        mse = [p[1] for p in points]
        plt.plot(components, mse, marker='o', alpha=0.5, label=f"seed {seed}")  # one thin line per seed

    # average across seeds, drawn on top
    all_components = sorted({p[0] for points in runs.values() for p in points})
    mean_mse = [np.mean([m for points in runs.values() for c, m in points if c == k]) for k in all_components]
    plt.plot(all_components, mean_mse, color="black", linewidth=3, marker='o', label="mean")
    plt.title(f"Bloch encoding {weight}, {dataset} : QSVR with PCA\nall seeds", fontsize=10)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.savefig(FOLDER / f"qsvr_pca_bloch_spaghetti_{weight}_{dataset}.png")
    plt.close()

"""
FOLDER = ROOT / "Images" / "Bloch PCA Baseline"
RESULTS_CSV = FOLDER / "QSVR_Bloch_PCA_Baseline_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "unweighted", "California Housing Dataset")

FOLDER = ROOT / "Images" / "California Bloch Powerlaw Seeds"
RESULTS_CSV = FOLDER / "QSVR_California_Bloch_Powerlaw_Seeds_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "weighted(1.0)", "California Housing Dataset")

FOLDER = ROOT / "Images" / "Synth Bloch PCA Baseline"
RESULTS_CSV = FOLDER / "QSVR_Synth_Bloch_PCA_Baseline_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "unweighted", "Synthetic dataset power law(3.0)")
FOLDER = ROOT / "Images" / "Powerlaw Seeds"
RESULTS_CSV = FOLDER / "QSVR_Powerlaw_Bloch_NoRotation_Seeds_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "weighted(1.0)", "Synthetic dataset power law(3.0)")

FOLDER = ROOT / "Images" / "Powerlaw Seeds Scaled"
RESULTS_CSV = FOLDER / "QSVR_Powerlaw_Bloch_NoRotation_Seeds_Scaled_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "weighted(1.0)", "Synthetic dataset power law(3.0)")
"""
SCALED = ROOT / "Images" / "Scaled QML Images"

FOLDER = SCALED / "Bloch PCA Baseline"
RESULTS_CSV = FOLDER / "QSVR_Bloch_PCA_Baseline_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "unweighted", "California Housing Dataset")
"""
FOLDER = SCALED / "California Bloch Powerlaw Seeds"
RESULTS_CSV = FOLDER / "QSVR_California_Bloch_Powerlaw_Seeds_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "weighted(1.0) ", "California Housing Dataset")

FOLDER = SCALED / "Synth Bloch PCA Baseline"
RESULTS_CSV = FOLDER / "QSVR_Synth_Bloch_PCA_Baseline_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "unweighted ", "Synthetic dataset power law(3.0)")

FOLDER = SCALED / "Powerlaw Seeds"
RESULTS_CSV = FOLDER / "QSVR_Powerlaw_Bloch_NoRotation_Seeds_Results.csv"

makegraph(RESULTS_CSV, FOLDER, "weighted(1.0)", "Synthetic dataset power law(3.0)")
"""