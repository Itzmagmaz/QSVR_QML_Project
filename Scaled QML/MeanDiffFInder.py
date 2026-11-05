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
weighted_mean= w.groupby("Seed")["MSE"].mean()
unweighted_mean = u.groupby("Seed")["MSE"].mean()

pchange= (unweighted_mean.mean() - weighted_mean.mean()) / unweighted_mean.mean() * 100

pval = stats.ttest_rel(weighted_mean, unweighted_mean).pvalue
print(f"weighted mean MSE = {weighted_mean.mean():.3f}, unweighted mean MSE = {unweighted_mean.mean():.3f}")
print(f"{pchange:.1f}% lower, p = {pval:.3f}")
