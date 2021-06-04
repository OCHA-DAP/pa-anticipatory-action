---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: antact
    language: python
    name: antact
---

### Compute the bias of CHIRPS-GEFS's 5 day forecast

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.enums import Resampling
import matplotlib
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl
import datetime
from datetime import timedelta
import re
import seaborn as sns
import calendar
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
# print(path_mod)
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp,download_url
from src.utils_general.raster_manipulation import fix_calendar, invert_latlon, change_longitude_range
from src.utils_general.plotting import plot_raster_boundaries_clip,plot_spatial_columns
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
country_data_raw_dir = os.path.join(config.DATA_DIR,config.RAW_DIR,country)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PROCESSED_DIR,country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)
dry_spells_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")
chirpsgefs_processed_dir = os.path.join(dry_spells_processed_dir,"chirpsgefs")

#we have different methodologies of computing dryspells and rainy season
#this notebook chooses one, which is indicated by the files being used
chirpsgefs_stats_path=os.path.join(chirpsgefs_processed_dir,"mwi_chirpsgefs_rainyseas_stats_mean_back.csv")
chirps_rolling_sum_path=os.path.join(dry_spells_processed_dir,"data_mean_values_long.csv")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```python
days_ahead=5
```

```python
if days_ahead==5:
    chirpsgefs_path=os.path.join(chirpsgefs_processed_dir,"mwi_chirpsgefs_rainyseas_stats_mean_back_adm2_5day.csv")
    chirps_rolling_sum_path=os.path.join(dry_spells_processed_dir,"data_mean_values_long_5day.csv")
elif days_ahead==15:
    chirpsgefs_path=os.path.join(chirpsgefs_processed_dir,"mwi_chirpsgefs_rainyseas_stats_mean_back.csv")
    chirps_rolling_sum_path=os.path.join(dry_spells_processed_dir,"data_mean_values_long.csv")
```

```python
#ccontains several statistics per adm2-date combination since 2000
df_cg_fd=pd.read_csv(chirpsgefs_path)
df_cg_fd["date"]=pd.to_datetime(df_cg_fd["date"])
#date_forec_end is not correct!! didn't adjust correctly in computation script
df_cg_fd["date_forec_end"]=df_cg_fd.date+timedelta(days=days_ahead-1)
# df_cg_fd["date_forec_end"]=pd.to_datetime(df_cg_fd["date_forec_end"])
```

```python
# df_bound_adm2=gpd.read_file(adm2_bound_path)
```

```python
#read historically observed 15 day rolling sum for all dates (so not only those with dry spells), derived from CHIRPS
#this sometimes gives a not permitted error --> move the chirps_rolling_sum_path file out of the folder and back in to get it to work (dont ask me why)
df_histobs=pd.read_csv(chirps_rolling_sum_path)
df_histobs.date=pd.to_datetime(df_histobs.date)

#add start of the rolling sum 
df_histobs["date_start"]=df_histobs.date-timedelta(days=days_ahead-1)

#add adm2 and adm1 name
# df_histobs=df_histobs.merge(df_bound_adm2[["ADM1_EN","ADM2_EN","ADM2_PCODE"]],left_on="pcode",right_on="ADM2_PCODE")
```

```python
df_histobs
```

```python
#merge forecast and observed
#only include dates that have a forecast, i.e. merge on right
#date in df_chirpsgefs is the first date of the forecast
#so the values are the rolling sum for the date+timedelta(t=days_ahead) #where days_aheas is 5 in this case
df_histformerg=df_histobs.merge(df_cg_fd,how="right",left_on=["date_start","pcode"],right_on=["date","ADM2_PCODE"],suffixes=("obs","forec"))
```

```python
df_histformerg["diff_forecobs"]=df_histformerg["mean_cell"]-df_histformerg[f"rollsum_{days_ahead}d"]
```

```python
df_histformerg.diff_forecobs.median()
```

```python

df_histformerg[["dateforec","dateobs","diff_forecobs","mean_cell","rollsum_15d"]] #,"rollsum_5d"
```

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_histformerg,y="diff_forecobs",x=f"rollsum_{days_ahead}d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_histformerg[f"rollsum_{days_ahead}d"].max()+20,10)
group = df_histformerg.groupby(pd.cut(df_histformerg[f"rollsum_{days_ahead}d"], bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels(f"Observed {days_ahead} day sum (mm)", f"Forecasted {days_ahead} day sum - Observed {days_ahead} day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells",f"plot_MWI_chirpsgefs_{days_ahead}days_density.png"))
```

```python
#plot the observed vs forecast-observed for obs<=2mm
df_sel=df_histformerg[df_histformerg[f"rollsum_{days_ahead}d"]<=2].sort_values(f"rollsum_{days_ahead}d")
g=sns.jointplot(data=df_sel,y="diff_forecobs",x=f"rollsum_{days_ahead}d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel[f"rollsum_{days_ahead}d"].max()+20,0.1)
group = df_sel.groupby(pd.cut(df_sel[f"rollsum_{days_ahead}d"], bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels(f"Observed {days_ahead} day sum (mm)", f"Forecasted {days_ahead} day sum - Observed {days_ahead} day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density.png"))
```

```python
#plot the observed vs forecast-observed for obs<=2mm
df_sel=df_histformerg[df_histformerg[f"rollsum_{days_ahead}d"]<=30].sort_values(f"rollsum_{days_ahead}d")
g=sns.jointplot(data=df_sel,y="diff_forecobs",x=f"rollsum_{days_ahead}d", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_sel[f"rollsum_{days_ahead}d"].max()+20,1)
group = df_sel.groupby(pd.cut(df_sel[f"rollsum_{days_ahead}d"], bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forecobs.mean()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="mean")
g.set_axis_labels(f"Observed {days_ahead} day sum (mm)", f"Forecasted {days_ahead} day sum - Observed {days_ahead} day sum", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
# plt.savefig(os.path.join(country_data_exploration_dir,"dryspells","plot_MWI_chirpsgefs_density.png"))
```

```python
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates
fig,ax=plt.subplots(figsize=(30,12))
df_histobs[(df_histobs.date.dt.year==2018)].groupby("date_start",as_index=False).mean().sort_values(by="date_start").plot.bar(x="date_start",y=f"rollsum_{days_ahead}d",ax=ax)
df_cg_fd[(df_cg_fd.date_forec_end.dt.year==2018)].groupby("date",as_index=False).mean().sort_values(by="date").plot.bar(x="date",y="mean_cell",ax=ax,alpha=0.5,color="red")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=12))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
```

```python

```
