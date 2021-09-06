## Plot ARC2 and CHIRPS resolution

Simple notebook to investigate the resolution of ARC2 and CHIRPS data when compared against various admin levels in Malawi.

```python
import rioxarray
from shapely.geometry import mapping
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
```

Set the config values and parameters.

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3)

arc2_dir = os.path.join(country_data_exploration_dir,"arc2")
arc2_filepath = os.path.join(arc2_dir, "arc2_20002020_approxmwi.nc")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
adm3_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin3_shp"])

plot_path = Path(config.DATA_DIR) / 'public' / 'processed' / 'mwi' / 'plots' / 'dry_spells' / 'arc2'
```

Read in data.

```python
# Admin boundaries
df_adm3 = gpd.read_file(adm3_bound_path)
df_adm2 = gpd.read_file(adm2_bound_path)
df_adm1 = gpd.read_file(adm1_bound_path)
```

```python
# ARC2 precipitation
ds=rioxarray.open_rasterio(arc2_filepath,masked=True).squeeze()
ds=ds.to_dataset()
ds.attrs["units"]='mm/day'
ds_clip=ds.rio.write_crs("EPSG:4326").rio.clip(df_adm1.geometry.apply(mapping), df_adm1.crs, all_touched=True)
```

Create the plots.

```python
f, ax = plt.subplots(figsize=(11, 10))
ds_clip.sel(T='2000-02-01').to_array().plot(cmap=plt.cm.Blues, ax=ax)
df_adm3.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.35)
ax.axis('off')
ax.set(title="ADM3 boundary with ARC2")
plt.savefig(plot_path / 'arc2_adm3.png')
```

```python
f, ax = plt.subplots(figsize=(11, 10))
ds_clip.sel(T='2000-02-01').to_array().plot(cmap=plt.cm.Blues, ax=ax)
df_adm2.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.35)
ax.axis('off')
ax.set(title="ADM2 boundary with ARC2")
plt.savefig(plot_path / 'arc2_adm2.png')
```
