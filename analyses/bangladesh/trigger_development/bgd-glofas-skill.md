### Evaluating the forecast skill of GloFAS in Bangladesh

This notebook is to compare the forecast skill of GloFAS for various lead times. We are comparing the reforecast product (lead time 5-30 days) against the reanalysis product. This is an improvement on the ```process-glofas``` notebook and takes the processed GloFAS data created by ```get_glofas_data.py```. 

We're specifically interested in the forecast skill during times of potential flooding, here estimated to be between June - Oct. 

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import pandas as pd
import xskillscore as xs

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas
from src.bangladesh import get_glofas_data as ggd

DATA_DIR = os.environ['AA_DATA_DIR']

# I've copied the processed data to my local aa directory because of this weird xarray file opening problem,
# but others would probably want to uncomment the line below to read from the GDrive
#GLOFAS_DIR = os.path.join(path_mod, 'analyses', 'bangladesh', 'trigger_development', 'processed_data') 
GLOFAS_DIR = os.path.join(DATA_DIR, 'processed', 'bangladesh', 'GLOFAS_Data')
SKILL_DIR = Path(os.path.join(DATA_DIR, 'exploration', 'bangladesh', 'GLOFAS_Data'))

STATION = 'Bahadurabad'
LEADTIME_HOURS = [120, 240, 360, 480, 600, 720]### Create GloFAS objects### Read in and interpolate data for station
```

### Create GloFAS objects

```python
glofas_reanalysis = glofas.GlofasReanalysis(
    stations_lon_lat=ggd.FFWC_STATIONS
)
glofas_forecast = glofas.GlofasForecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=ggd.LEADTIME_HOURS
)
glofas_reforecast = glofas.GlofasReforecast(
    stations_lon_lat=ggd.FFWC_STATIONS, leadtime_hours=ggd.LEADTIME_HOURS
)
```

### Read in and interpolate data for station

```python
da_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3
    )[STATION]

def shift_dates(da_dict):
    return{leadtime_hour:
        da.assign_coords(time=da.time.values + np.timedelta64(
            int(leadtime_hour/24), 'D'))
        for leadtime_hour, da in da_dict.items()
        }

def interp_dates(da_dict):
    return {
        leadtime_hour:
    da.interp(
        time=pd.date_range(
           da.time.min().values, 
          da.time.max().values), 
          method='linear')
    for leadtime_hour, da
    in da_dict.items()
    }

da_glofas_forecast_dict = {leadtime_hour:
    glofas_forecast.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, 
        country_iso3=ggd.COUNTRY_ISO3, 
        leadtime_hour=leadtime_hour
    )[STATION]
    for leadtime_hour in ggd.LEADTIME_HOURS}
da_glofas_forecast_dict = shift_dates(da_glofas_forecast_dict)

da_glofas_reforecast_dict = {leadtime_hour:
    glofas_reforecast.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, 
        country_iso3=ggd.COUNTRY_ISO3, 
        leadtime_hour=leadtime_hour
    )[STATION]
    for leadtime_hour in ggd.LEADTIME_HOURS}
da_glofas_reforecast_dict = interp_dates(
    shift_dates(da_glofas_reforecast_dict))
da_glofas_reforecast_dict
```

#### Read in and clean up the reforecast and reanalysis

The function below reads in the reforecast data based on an input lead time and station. This is read in directly from the processed ```.nc``` file, but in the future should make use of the existing ```get_glofas_data.py``` script. The data is also interpolated to a daily granularity and temporally shifted according to the forecast lead time. 

We'll read in the reforecast data according to the station and leadtime hours specified above. We'll also read in the reanalysis data to use as our proxy for historical observations. 

```python
def process_rf(lt, station):
    rf = xr.open_dataset(os.path.join(GLOFAS_DIR, f'bgd_cems-glofas-reforecast_0{lt}.nc'))[STATION]
    rf = (
        rf.interp(time=pd.date_range(rf.time.min().values, rf.time.max().values), method='linear')
            .shift(time=int(lt/24))
            .rename({'number': 'member'}) # Rename to fit with xskillscore
    )
    return rf

rf_list = dict((hour, process_rf(hour, STATION)) for hour in LEADTIME_HOURS)
#rf_list = [process_rf(hour, STATION) for hour in LEADTIME_HOURS]
ra = xr.open_dataset(os.path.join(GLOFAS_DIR, 'bgd_cems-glofas-historical.nc'))[STATION]
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2000-06-01'
end = '2000-10-31'
rf_list_slice = [rf_list[lt].sel(time=slice(start, end)) for lt in rf_list]
rf_list_slice_mean = [da.mean(dim='member', keep_attrs=True) for da in rf_list_slice]
ra_slice = ra.sel(time=slice(start, end))

# Basic line plot to check that the data looks like we'd expect
for rf in rf_list_slice_mean:
    lt = rf[1].step.values/np.timedelta64(1, 'D')
    rf.plot.line(label=f'LT {int(lt)} days')
ra_slice.plot.line(label='Historical')
plt.legend()
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. This is also what GloFAS uses in evaluating their own forecast skill.

```python
df_crps = pd.DataFrame(columns=['leadtime', 'crps', 'year'])
years = list(range(1999,2019))

for year in years:
    for lt in rf_list:
        start = str(year) + '-06-01'
        end = str(year) + '-10-30'
        rf_slice = rf_list[lt].sel(time=slice(start, end))
        ra_slice = ra.sel(time=slice(start, end))
        crps_ensemble = xs.crps_ensemble(ra_slice, rf_slice)
        df_crps = df_crps.append([{'leadtime':lt, 'crps':crps_ensemble.values.round(2), 'year': year}], ignore_index=True)

# Save results disaggregated by year
df_crps.to_csv(os.path.join(GLOFAS_DIR, f'{STATION}_crps.csv'))
```

```python
# Quick plot to summarize the results across all years
known_skill = SKILL_DIR / 'forecast_skill.csv'
df_skill = pd.read_csv(known_skill, header=None)

df_crps_avg = df_crps.groupby('leadtime').mean().reset_index()

plt.plot(df_crps_avg.leadtime/24, df_crps_avg.crps)
plt.plot(df_skill[0], df_skill[1])
plt.title("GloFAS forecast skill at Bahadurabad:\n 1999-2018 reforecast, June-Oct average")
plt.xlabel("Lead time (days)")
plt.ylabel("CRPS")
plt.show()
```

### Bias: rank histogram

```python

```
