```python
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import norm, pearsonr


import read_in_data as rd
reload(rd)

mpl.rcParams['figure.dpi'] = 300
```

### Create GloFAS objects

```python
da_glofas_reanalysis = rd.get_glofas_reanalysis()
da_glofas_forecast = rd.get_glofas_forecast()
da_glofas_forecast_summary = rd.get_da_glofas_summary(da_glofas_forecast)
da_glofas_reforecast = rd.get_glofas_reforecast()
da_glofas_reforecast_summary = rd.get_da_glofas_summary(da_glofas_reforecast)

```

 ### Read in FFWC data

```python
df_ffwc_wl = rd.read_in_ffwc()
```

### Add GloFAS to FFWC

```python
# Create final df
df_final = df_ffwc_wl.copy()

# Add glofas obs
df_glofas = da_glofas_reanalysis.to_dataframe()[[rd.STATION]].rename(columns={rd.STATION: 'glofas_observed'})
df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)

# Add glofas forecasts
glofas_columns = ['median', 
'1sig-', '2sig-', '3sig-', 
'1sig+', '2sig+', '3sig+']
for leadtime_hour in da_glofas_reforecast_summary.leadtime_hour:
    df_glofas_reforecast = da_glofas_reforecast_summary.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas_forecast = da_glofas_forecast_summary.sel(leadtime_hour=leadtime_hour).to_dataframe()[glofas_columns]
    df_glofas = (pd.concat([df_glofas_reforecast, df_glofas_forecast])
                 .rename(columns={cname: f'glofas_{int(leadtime_hour/24)}day_{cname}' for cname in glofas_columns}))
    df_final = pd.merge(df_final, df_glofas, how='outer', left_index=True, right_index=True)

# Don't bother starting before FFWC observations do
df_final = df_final[df_final.index >= df_ffwc_wl.index[0]]

```

### Read in FFWC discharge data

```python
ffwc_discharge_filename = 'bahadurabad_discharge_01.xlsx'
df_ffwc_discharge=pd.read_excel(rd.ffwc_dir / ffwc_discharge_filename,
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
