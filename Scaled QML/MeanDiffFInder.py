import pandas as pd
from scipy import stats
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "Scaled QML Images"
"""
w = pd.read_csv(IMAGES_DIR / "Powerlaw Seeds" / "QSVR_Powerlaw_Bloch_NoRotation_Seeds_Results.csv")
u = pd.read_csv(IMAGES_DIR / "Synth Bloch PCA Baseline" / "QSVR_Synth_Bloch_PCA_Baseline_Results.csv")
"""
w = pd.read_csv(IMAGES_DIR / "California Bloch Powerlaw Seeds" / "QSVR_California_Bloch_Powerlaw_Seeds_Results.csv")
u = pd.read_csv(IMAGES_DIR / "Bloch PCA Baseline" / "QSVR_Bloch_PCA_Baseline_Results.csv")
w_seed = w.groupby("Seed")["MSE"].mean()
u_seed = u.groupby("Seed")["MSE"].mean()

pct = (u_seed.mean() - w_seed.mean()) / u_seed.mean() * 100

p = stats.ttest_rel(w_seed, u_seed).pvalue
print(f"weighted mean MSE = {w_seed.mean():.3f}, unweighted mean MSE = {u_seed.mean():.3f}")
print(f"{pct:.1f}% lower, p = {p:.3f}")
