### Evaluating the forecast skill of ECMWF seasonal forecast in Malawi
This notebook is to compare the forecast skill of ECMWF's seasonal forecast for various lead times. We use the monthly total precipitation that is forecasted by this forecast. As ground truth, we use CHIRPS observations. 

This notebook solely looks at the CRPS across all leadtimes, and makes a distinction for the rainy season and for months with low precipitation.   

`mwi_ecmwf_monthly_skill_dryspells.md` assesses the skill of the forecast for dry spells specifically. 

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import rioxarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xskillscore as xs
import numpy as np

import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
import glob
import itertools

import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(country_data_processed_dir,"dry_spells","full_list_dry_spells.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
```

# Determine general skill per leadtime


### Read in forecast and observational data

```python
# #only needed if stats files haven't been computed before and are not on the gdrive
# #takes several hours
# processing.compute_stats_per_admin(country)
```

```python
da = processing.get_ecmwf_forecast("mwi")
```

```python
da_lt=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_obs=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
#only select the years for which we also identified dry spells
da_obs=da_obs.sel(time=da_obs.time.dt.year.isin(range(2000,2021)))
```

```python
#interpolate forecast data such that it has the same resolution as the observed values
#using "nearest" as interpolation method and not "linear" because the forecasts are designed to have sharp edged and not be smoothed
da_lt_interp=da_lt.interp(latitude=da_obs["y"],longitude=da_obs["x"],method="nearest")
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-06-01'
# end = '2020-10-31'

rf_list_slice = da_lt.sel(time=start,latitude=da_lt.latitude.values[5],longitude=da_lt.longitude.values[5])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts.

```python
df_crps=pd.DataFrame(columns=['leadtime', 'crps'])

#rainy includes the months to select
#thresh the thresholds
subset_dict={"rainy":[12,1,2],"thresh":[170]}

for leadtime in da_lt_interp.leadtime:
    forecast = da_lt_interp.sel(
    leadtime=leadtime.values).dropna(dim='time')
    observations = da_obs.reindex({'time': forecast.time}).precip
    # For all dates
    crps = xs.crps_ensemble(observations, forecast,member_dim='number')
    append_dict = {'leadtime': leadtime.values,
                          'crps': crps.values,
                           'std': observations.std().values,
                           'mean': observations.mean().values,
              }

    if "rainy" in subset_dict:
        # For rainy season only
        observations_rainy = observations.where(observations.time.dt.month.isin(subset_dict['rainy']), drop=True)
        crps_rainy = xs.crps_ensemble(
            observations_rainy,
            forecast.where(forecast.time.dt.month.isin(subset_dict['rainy']), drop=True),
            member_dim='number')
        append_dict.update({
                f'crps_rainy': crps_rainy.values,
                f'std_rainy': observations_rainy.std().values,
                f'mean_rainy': observations_rainy.mean().values
            })
    
    if "thresh" in subset_dict:
        for thresh in subset_dict["thresh"]:
            crps_thresh = xs.crps_ensemble(observations.where(observations<=thresh), forecast.where(observations<=thresh), member_dim='number')
            append_dict.update({
                f'crps_{thresh}': crps_thresh.values,
                f'std_{thresh}': observations.where(observations<=thresh).std().values,
                f'mean_{thresh}': observations.where(observations<=thresh).mean().values
            })
        df_crps= df_crps.append([append_dict], ignore_index=True)
```

```python
def plot_skill(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([k for k in df_crps.keys() if "crps" in k]):
        y = df[subset]
        if division_key is not None:
            dkey = f'{division_key}_{subset.split("_")[-1]}' if subset!="crps" else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')
    # Add colours to legend
    for i, subset in enumerate([k for k in append_dict.keys() if "crps" in k]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend()
```

```python
#performance pretty bad.. especially looking at the mean values, it is about 20% off on average for decjanfeb..
# Plot absolute skill
plot_skill(df_crps)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

```
