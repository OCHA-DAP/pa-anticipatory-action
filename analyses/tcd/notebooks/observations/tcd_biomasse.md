```python
%load_ext jupyter_black
```

```python
#### Load libraries and set global constants

%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import sys
import os

CHD_GREEN = "#1bb580"

# quick fix for changing env variables from AA to OAP
os.environ["AA_DATA_DIR"] = os.environ["OAP_DATA_DIR"]
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

iso3 = "tcd"
config = Config()
parameters = config.parameters(iso3)
country_data_processed_dir = (
    Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3
)
adm2_bound_path = (
    country_data_processed_dir
    / config.SHAPEFILE_DIR
    / "tcd_adm2_area_of_interest.gpkg"
)

#### Set variables

_save_processed_path = os.path.join(
    path_mod,
    Path(config.DATA_DIR),
    config.PUBLIC_DIR,
    config.PROCESSED_DIR,
    iso3,
    "wrsi",
)

import src.indicators.drought.biomasse as bm
```

This is just the code for downloading and procesing the Biomasse data, then aggregating to a specific set of admin codes (our region of interest in Chad. Additional analysis and exploration is done within `biomasse_exploration.R`.

```python
bm.download_dmp()
```

```python
dmp = bm.calculate_biomasse(admin_level="ADM2")
```

```python
gdf_adm2 = gpd.read_file(adm2_bound_path)
gdf_reg = gdf_adm2[gdf_adm2.area_of_interest == True]
bm_df = bm.aggregate_biomasse(admin_pcodes=gdf_reg.admin2Pcod, iso3="tcd")
```

We are activating if the `biomasse_anomaly` is below 80 in the 24th dekad of 2023. We can quickly check that below.

```python
bm_df[(bm_df["year"] == 2023) & (bm_df["dekad"] == 24)]
```
```python
bm_df[(bm_df["year"] == 2024) & (bm_df["dekad"] == 24)]
```

```python
df_plot = bm_df[bm_df["dekad"] == 24].copy()
df_plot["biomasse_anomaly"] /= 100

fig, ax = plt.subplots(dpi=300)
df_plot.plot(
    x="year", y="biomasse_anomaly", ax=ax, legend=False, color="dodgerblue"
)

thresh = 0.8

ax.axhline(y=thresh, color="grey", linestyle="--", alpha=0.5)
ax.annotate(
    f" seuil = {thresh:.0%}",
    xy=(2025, thresh),
    color="grey",
    ha="left",
    va="center",
)

for year, row in df_plot.set_index("year").iterrows():
    tp = row["biomasse_anomaly"]
    if tp <= thresh:
        ax.annotate(
            year,
            xy=(year, tp),
            color="crimson",
            ha="center",
            va="top",
        )

current_year = 2024
current_val = df_plot.set_index("year").loc[current_year, "biomasse_anomaly"]

ax.annotate(
    f" {current_year}\n {current_val:.0%}",
    xy=(current_year, current_val),
    color=CHD_GREEN,
    ha="left",
    va="center",
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Année")
ax.set_ylabel("Anomalie de biomasse")
ax.set_title("Anomalie de biomasse à 3e décade d'août")
```
```python
df_plot
```




