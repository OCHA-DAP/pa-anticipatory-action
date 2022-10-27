"""
Script to turn csv of GloFAS stations into config format
for aa toolbox. Input is country ISO2 code.
"""
import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
FILENAME = (
    DATA_DIR / "private/exploration/glb/glofas/Qgis_World_outlet_20221004.csv"
)

country_code = sys.argv[1].upper()

df = pd.read_csv(
    FILENAME,
    usecols=[
        "station_id",
        "StationName",
        "Country code",
        "CountryName",
        "LisfloodX",
        "LisfloodY",
    ],
)
df = df.loc[df["Country code"] == country_code].sort_values(
    by=["StationName", "station_id"]
)

# Add suffix
g = df.groupby("StationName")
df["StationName"] += (
    g.cumcount()
    .add(1)
    .astype(str)
    .radd(" ")
    .mask(g["StationName"].transform("count") == 1, "")
)

print("glofas:")
print("  reporting_points:")
for _, row in df.iterrows():
    # print(f"  - id: {row['station_id']}")
    print(f"  - name: {row['StationName']}")
    print(f"    lon: {row['LisfloodX']}")
    print(f"    lat: {row['LisfloodY']}")
