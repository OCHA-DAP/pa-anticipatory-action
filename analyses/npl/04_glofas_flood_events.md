We want to know the forecast performance against GloFAS events, i.e. when an
event is defined as a GloFAS return period exceedance

```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from matplotlib.ticker import MaxNLocator

import npl_parameters as parameters
from src.indicators.flooding.glofas import utils
from src.utils_general import math
```

```python
RP_LIST = [1.5, 2, 5]
MAIN_RP = 1.5
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=parameters.COUNTRY_ISO3, use_incorrect_area_coords=parameters.USE_INCORRECT_AREA_COORDS)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = parameters.COUNTRY_ISO3, leadtimes=parameters.LEADTIMES,
    interp=True,
    use_incorrect_area_coords=parameters.USE_INCORRECT_AREA_COORDS
)
ds_glofas_reforecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)

#df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
df_return_period =  pd.read_excel(parameters.GLOFAS_RP_FILENAME, index_col='rp')

```

```python
df_station_stats = pd.DataFrame(columns=['station', 'rp', 'leadtime', 'TP', 'FP', 'FN'])

for station in df_return_period.columns:
    #for rp in df_return_period.index:
    model = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station + parameters.VERSION_LOC]
    forecast = ds_glofas_reforecast_summary[station + parameters.VERSION_LOC].sel(percentile=parameters.MAIN_FORECAST_PROB)
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, station]
        model_dates = utils.get_dates_list_from_data_array(model,  rp_val, min_duration=parameters.DURATION)
        for leadtime in parameters.LEADTIMES:
            forecast_dates = utils.get_dates_list_from_data_array(
                forecast.sel(leadtime=leadtime), rp_val, min_duration=parameters.DURATION)
            detection_stats = utils.get_detection_stats(true_event_dates=model_dates,
                                                       forecasted_event_dates=forecast_dates,
                                                       days_before_buffer=parameters.DAYS_BEFORE_BUFFER,
                                                       days_after_buffer=parameters.DAYS_AFTER_BUFFER)
            df_station_stats = df_station_stats.append({
                **{'station': station,
                'leadtime': leadtime,
                'rp': rp},
                **detection_stats
            }, ignore_index=True)

df_station_stats = utils.get_more_detection_stats(df_station_stats)
df_station_stats
```

```python
#rp_dict = {
#    1.5: '-', 
#    2: '--',
#    5: ':'
#}
rp_dict = {MAIN_RP: '-'}
plot_numbers = True
plot_precision_recall = True
leadtime_range = (1, 10)

for istation, station in enumerate(parameters.FINAL_STATIONS):
    if plot_numbers:
        qs = ['TP', 'FP', 'FN']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == station) & 
                                    (df_station_stats['rp'] == rp)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Number')
        # Make legend
        for iq, q in enumerate(qs):
            ax.plot([], [], c=f'C{iq}', label=q)
        if len(rp_dict) > 1:
            for rp, ls in rp_dict.items():
                ax.plot([], [], c='k', ls=ls, label=rp)
        ax.legend()
        ax.set_xlim(leadtime_range)
    
    if plot_precision_recall:
        qs = ['precision', 'recall']
        fig, ax = plt.subplots()
        for rp, ls in rp_dict.items():
            data = df_station_stats[(df_station_stats['station'] == station) & 
                                    (df_station_stats['rp'] == rp)]
            for iq, q in enumerate(qs):
                ax.plot(data['leadtime'], data[q], ls=ls, c=f'C{iq}')
        ax.set_title(station)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Leadtime [days]')
        ax.set_ylabel('Fraction')
        ax.set_ylim(-0.1, 1.1)
        # Make legend
        for iq, q in enumerate(qs):
            ax.plot([], [], c=f'C{iq}', label=q)
        if len(rp_dict) > 1:
            for rp, ls in rp_dict.items():
                ax.plot([], [], c='k', ls=ls, label=rp)
        ax.legend()
        ax.set_xlim(leadtime_range)
```

```python
# Print out some stats
# Round to the nearest 5 for a presentation (10 seems a bit too coarse)
rp = MAIN_RP
for station in parameters.FINAL_STATIONS:
    data = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp)]
    for q in ["POD", "FAR"]:
        data.loc[:, q + "_rounded"] = math.round_to_n(data["POD"].fillna(-1) * 100, 5)
    print(station)
    print(data[["leadtime", "POD_rounded", "FAR_rounded"]])
```

### Make plot showing event comparison

```python
rp = MAIN_RP
leadtimes = [7, 3] # Longer first

for station in parameters.FINAL_STATIONS:

    thresh = df_return_period.loc[rp, station]
    fig, axs = plt.subplots(3, figsize=(8,6))
    observations = ds_glofas_reanalysis.reindex(time=ds_glofas_reforecast.time)[station + parameters.VERSION_LOC]
    forecast = utils.get_glofas_forecast_summary(ds_glofas_reforecast)[station + parameters.VERSION_LOC].sel(percentile=MAIN_FORECAST_PROB)
    forecast_1 = forecast.sel(leadtime=leadtimes[0])
    forecast_2 = forecast.sel(leadtime=leadtimes[1])
    fig.supylabel('River dischange [m$^3$ s$^{-1}$]')
    fig.supxlabel('Year')
    fig.suptitle(station)
    for i, q in enumerate([forecast_1, forecast_2, observations]):
        ax = axs[i]
        if i == 2:
            c1 = 'k'
            c2 = 'C3'
            title = 'Modelled'
        else:
            c1 = 'C0'
            c2 = 'C1'
            title = f'{leadtimes[i]}-day lead time'
            ax.set_xticklabels([])
        x = q.time
        y = q.values
        ax.plot(x, y, '-', c=c1, lw=0.5)
        ax.set_ylim(0, 8000)
        #ax.set_xlim(0, 120000)
        ax.minorticks_on()
        ax.axhline(thresh, c=c2, lw=0.5)
        ax.set_title(title)
        for detection in utils.get_groups_above_threshold(y, thresh, parameters.DURATION):
            ax.plot(x[detection[0]], y[detection[0]], 'o', c=c2, lw=2, mfc='none')
```

### Compare Chisapani to Asaraghat

```python
# Want to compare Chisapani to Asaraghat
ls_list = ['-', '--']
stations = ['Asaraghat', 'Chisapani']
fig, ax = plt.subplots()
for istation, station in enumerate(stations):
    qs = ['precision', 'recall']
    data = df_station_stats[(df_station_stats['station'] == station) & 
                            (df_station_stats['rp'] == parameters.MAIN_RP)]
    for iq, q in enumerate(qs):
        ax.plot(data['leadtime'], data[q], ls=ls_list[istation], c=f'C{iq}')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Leadtime [days]')
ax.set_ylabel('Fraction')
ax.set_ylim(-0.1, 1.1)
# Make legend
for iq, q in enumerate(qs):
    ax.plot([], [], c=f'C{iq}', label=q)
for istation, station in enumerate(stations):
    ax.plot([], [], c='k', ls=ls_list[istation], label=station)
ax.legend()
```

```python
rp = MAIN_RP
leadtimes = [1, 3, 5, 7, 10]
for station in ['Asaraghat', 'Chisapani']:
    df = df_station_stats[(df_station_stats['station'] == station) & (df_station_stats['rp'] == rp) & (df_station_stats.leadtime.isin(leadtimes))]
    print(np.round(100 * (1-df['recall'])))
```
