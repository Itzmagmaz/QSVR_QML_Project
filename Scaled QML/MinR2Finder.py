import pandas as pd
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "Scaled QML Images"

for csv_file in sorted(IMAGES_DIR.glob("*/*.csv")):
    df = pd.read_csv(csv_file)
    print(f"{csv_file.parent.name}: mean R2 = {df['R2'].mean():.3f}")
