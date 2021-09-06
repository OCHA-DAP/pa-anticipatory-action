## Plot ARC2 and CHIRPS resolution

Simple notebook to investigate the resolution of ARC2 and CHIRPS data when compared against various admin levels in Malawi.

```python
from pathlib import Path
import sys
import os

import rioxarray
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
```

Set the config values and parameters.

```python
path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

config = Config()
parameters = config.parameters('malawi')
COUNTRY_ISO3 = parameters["iso3_code"]
DATA_DIR = Path(config.DATA_DIR)

RAW_DIR =  DATA_DIR / config.PUBLIC_DIR / config.RAW_DIR / COUNTRY_ISO3
ARC2_DIR = DATA_DIR / config.PUBLIC_DIR / 'exploration' / COUNTRY_ISO3 / 'arc2'
ARC2_FILEPATH = ARC2_DIR / "arc2_20002020_approxmwi.nc"

ADM1_SHP = RAW_DIR / config.SHAPEFILE_DIR / parameters['path_admin1_shp']
ADM2_SHP = RAW_DIR / config.SHAPEFILE_DIR / parameters['path_admin2_shp']
ADM3_SHP = RAW_DIR / config.SHAPEFILE_DIR / parameters['path_admin3_shp']

PLOT_DIR = DATA_DIR / 'public' / 'processed' / 'mwi' / 'plots' / 'dry_spells' / 'arc2'
```

Read in data.

```python
# Admin boundaries
df_adm3 = gpd.read_file(ADM1_SHP)
df_adm2 = gpd.read_file(ADM2_SHP)
df_adm1 = gpd.read_file(ADM3_SHP)

# ARC2 precipitation
ds = rioxarray.open_rasterio(ARC2_FILEPATH,masked=True).squeeze().to_dataset()
ds.attrs["units"]='mm/day'
ds_clip=ds.rio.write_crs("EPSG:4326").rio.clip(df_adm1.geometry.apply(mapping), df_adm1.crs, all_touched=True)
```

Create the plots.

```python
f, ax = plt.subplots(figsize=(11, 10))
ds_clip.sel(T='2000-02-01').to_array().plot(cmap=plt.cm.Blues, ax=ax) # Just select a single date, doesn't really matter which one
df_adm3.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.35)
ax.axis('off')
ax.set(title="ADM3 boundary with ARC2")
plt.savefig(PLOT_DIR / 'arc2_adm3.png')
```

```python
f, ax = plt.subplots(figsize=(11, 10))
ds_clip.sel(T='2000-02-01').to_array().plot(cmap=plt.cm.Blues, ax=ax)
df_adm2.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.35)
ax.axis('off')
ax.set(title="ADM2 boundary with ARC2")
plt.savefig(PLOT_DIR / 'arc2_adm2.png')
```
