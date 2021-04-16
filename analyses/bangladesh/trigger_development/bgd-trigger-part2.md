```python
import sys
import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
import numpy as np
import pandas as pd
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding import glofas
from src.bangladesh import get_glofas_data as ggd

DATA_DIR = Path(os.environ['AA_DATA_DIR'])
GLOFAS_DIR = DATA_DIR / 'processed/bangladesh/GLOFAS_Data' 
STATION = 'Bahadurabad_glofas'
mpl.rcParams['figure.dpi'] = 300
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
```

### For forecast and reforecast, create summary data array
Contains median and 1,2,3 sigma centiles

```python
nsig_max = 3
percentile_dict = {
    **{'median': 50.},
    **{f'{n}sig+': norm.cdf(n) * 100 for n in range(1,nsig_max+1)},
    **{f'{n}sig-': (1-norm.cdf(n)) * 100 for n in range(1,nsig_max+1)},
}
coord_names = ["leadtime_hour", "time"]

def get_da_glofas_summary(da_glofas_dict):

    data_vars_dict = {var_name:
        (coord_names,
        np.array([
            np.percentile(da_glofas, percentile_value, axis=0)
            for da_glofas in da_glofas_dict.values()
        ]))
        for var_name, percentile_value in percentile_dict.items()}

    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(
            time=da_glofas_dict[120].time,
            leadtime_hour=ggd.LEADTIME_HOURS
        )
    )

da_glofas_forecast = get_da_glofas_summary(da_glofas_forecast_dict)
da_glofas_reforecast = get_da_glofas_summary(da_glofas_reforecast_dict)
```

 ### Read in FFWC data

```python
ffwc_dir = DATA_DIR / 'exploration/bangladesh/FFWC_Data'
```

```python
# Read in data from Sazzad that has forecasts
ffwc_wl_filename = 'Bahadurabad_WL_forecast20172019.xlsx'
ffwc_leadtime_hours = [24, 48, 72, 96, 120]

# Need to combine the three sheets
df_ffwc_wl_dict = pd.read_excel(
        ffwc_dir / ffwc_wl_filename,
                                sheet_name=None,
                                header=[1], index_col='Date')
df_ffwc_wl = (df_ffwc_wl_dict['2017']
              .append(df_ffwc_wl_dict['2018'])
                        .append(df_ffwc_wl_dict['2019'])
                        .rename(columns={
                            f'{leadtime_hour} hrs': f'ffwc_{int(leadtime_hour/24)}day'
                            for leadtime_hour in ffwc_leadtime_hours
                        })).drop(columns=['Observed WL']) # drop observed because we will use the mean later
# Convert date time to just date
df_ffwc_wl.index = df_ffwc_wl.index.floor('d')
```

```python
# Then read in the older data (goes back much futher)
FFWC_RL_HIS_FILENAME='2020-06-07 Water level data Bahadurabad Upper danger level.xlsx'
ffwc_rl_name='{}/{}'.format(ffwc_dir,FFWC_RL_HIS_FILENAME)
df_ffwc_wl_old=pd.read_excel(ffwc_rl_name,index_col=0,header=0)
df_ffwc_wl_old.index=pd.to_datetime(df_ffwc_wl_old.index,format='%d/%m/%y')
df_ffwc_wl_old
df_ffwc_wl_old=df_ffwc_wl_old[['WL']].rename(columns={'WL': 
                                                    'observed'})[df_ffwc_wl_old.index < df_ffwc_wl.index[0]]
df_ffwc_wl = pd.concat([df_ffwc_wl_old, df_ffwc_wl])
```

```python
# Read in the more recent file from Hassan
ffwc_full_data_filename = 'SW46.9L_19-11-2020.xls'
df_ffwc_wl_full = (pd.read_excel(ffwc_dir / ffwc_full_data_filename, 
                                 index_col='DateTime')
                   .rename(columns={'WL(m)': 'observed'}))[['observed']]

# Mutliple observations per day. Find mean and std
df_ffwc_wl_full['date'] = df_ffwc_wl_full.index.date
df_ffwc_wl_full = (df_ffwc_wl_full.groupby('date').agg(['mean', 'std'])
)['observed'].rename(columns={'mean': 'observed', 'std': 'obs_std'})
df_ffwc_wl_full.index = pd.to_datetime(df_ffwc_wl_full.index)


# Combine with first DF

df_ffwc_wl = pd.merge(df_ffwc_wl_full[['obs_std']], df_ffwc_wl, left_index=True, right_index=True, how='outer')
df_ffwc_wl.update(df_ffwc_wl_full, overwrite=False)
```

### Add GloFAS to FFWC

```python
# Create final df
df_final = df_ffwc_wl.copy()

# Add glofas obs
df_glofas = da_glofas_reanalysis.to_dataframe()[[STATION]].rename(columns={STATION: 'glofas_observed'})
df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)

# Add glofas forecasts
glofas_columns = ['median', 
'1sig-', '2sig-', '3sig-', 
'1sig+', '2sig+', '3sig+']
for leadtime_hour in ggd.LEADTIME_HOURS:
    df_glofas_reforecast = da_glofas_reforecast.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas_forecast = da_glofas_forecast.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas = (pd.concat([df_glofas_reforecast, df_glofas_forecast])
                 .rename(columns={cname: f'glofas_{int(leadtime_hour/24)}day_{cname}' for cname in glofas_columns}))
    df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)
    
## Any event elements that are NA should be False    
#df_final['event'] = df_final['event'].fillna(False)

# Don't bother starting before FFWC observations do
df_final = df_final[df_final.index >= df_ffwc_wl.index[0]]

```

### Read in FFWC discharge data

```python
ffwc_discharge_filename = 'bahadurabad_discharge_01.xlsx'
df_ffwc_discharge=pd.read_excel(ffwc_dir / ffwc_discharge_filename,
                index_col='Date').rename(
    columns={'Discharge (m3/s)': 'ffwc_discharge'})
df_discharge = pd.merge(
                            df_final, df_ffwc_discharge,
                            how='inner',
                             left_index=True, right_index=True)
```

### Plot GloFAS vs FFWC discharge

```python
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(df_discharge['ffwc_discharge'], df_discharge['glofas_observed'], '.', alpha=0.5)
x = np.arange(-10000, 200000, 10000)
ax.plot(x, x, c='r')
ax.set_xlim(0, 150000)
ax.set_ylim(0, 150000)
ax.set_xlabel('FFWC water discharge (m^3 s^-1)')
ax.set_ylabel('GloFAS water discharge (m^3 s^-1)')

```

```python
x = df_discharge['ffwc_discharge']
y = df_discharge['glofas_observed']
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, (y-x)/x, '.', alpha=0.5)
ax.axhline(0, c='k')
ax.set_xlabel('FFWC water discharge (m$^3$ s$^{-1}$)')
ax.set_ylabel('Fractional difference [(G-F)/F]')

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.plot(x, (y-x), '.', alpha=0.5)
ax2.axhline(0, c='k')
ax2.set_xlabel('FFWC water discharge (m$^3$ s$^{-1}$)')
ax2.set_ylabel('GloFAS - FFWC (m$^3$ s$^{-1}$)')
ax2.set_ylim(-70000, 70000)
```

```python
# For GloFAS discharge values > 50000:
thresh = 50000
df_discharge_high = df_discharge[df_discharge['ffwc_discharge'] > thresh]

bins = np.arange(-2, 2, 0.1)
y = df_discharge_high['glofas_observed'] 
x = df_discharge_high['ffwc_discharge']
z = (y-x)/x
plt.hist(z, bins=bins)
```

```python
xvar = 'observed'
yvar = 'ffwc_discharge'
df = df_discharge[[xvar, yvar]].dropna()
x = df[xvar]
y = df[yvar]

fig, ax = plt.subplots()
ax.plot(x, y, '.', alpha=0.5)
ax.set_xlabel('FFWC water level [m]')
ylabel='FFWC river discharge [m$^3$ s$^{-1}$]'
ax.set_ylabel(ylabel)

split_val = 19.5
idx = x < split_val
print("Pearson's for above and below 19.5", pearsonr(x[idx], y[idx]),
pearsonr(x[~idx], y[~idx]))

ax.axvline(19.5, lw=0.3, c='k')
m3, m2, m1, b = np.polyfit(x[idx], y[idx], 3)
xplot = np.arange(12, 22)
plt.plot(xplot, m3 * xplot**3 + m2 * xplot**2 +  m1*xplot + b)
```

```python
y_reduced = y /( m3 * x **3 + m2 * x **2 + m1 * x + b)
plt.hist(y_reduced, bins=np.arange(0.4, 1.4, 0.05))
```

```python
(975 - 799) / 975
```

```python
df
```

```python

```
