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
    DATA_DIR
    / "private/exploration/glb/glofas/Qgis_World_outlet_202104_20210421.csv"
)

country_code = sys.argv[1]

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
df = df.loc[df["Country code"] == country_code].sort_values(by="station_id")

print("glofas:")
print("  reporting_points:")
for _, row in df.iterrows():
    print(f"  - id: {row['station_id']}")
    print(f"    name: {row['StationName']}")
    print(f"    lon: {row['LisfloodX']}")
    print(f"    lat: {row['LisfloodY']}")
