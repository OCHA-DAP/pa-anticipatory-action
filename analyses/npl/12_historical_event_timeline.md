# Event timeline

Go through each date and figure out when an activation would have occured. Also, do a more thorough
historical analysis, taking into account all forecast lead times.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import npl_settings as settings
from src.indicators.flooding.glofas import utils
```

```python
RP_LIST = [1.5, 2, 5]
FORECAST_PERCENTILE_LIST = [50, 30]

LEADTIMES = [x+1 for x in range(7)]
LEADTIMES_BY_TRIGGER = {
    "action": [1, 2, 3],
    "readiness": [4, 5, 6, 7]
}
```

```python
df_station_info = pd.read_excel(settings.DHM_STATION_INFO_FILENAME, index_col='station_name')
df_wl = pd.read_csv(settings.WL_OUTPUT_FILENAME, index_col='date', parse_dates=True)

ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=settings.COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = settings.COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True, shift_dates=False
)
ds_glofas_forecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)
df_return_period =  pd.read_excel(settings.GLOFAS_RP_FILENAME, index_col='rp')
pd.options.mode.chained_assignment = None  # default='warn'
```

### Create dataframe with both river discharge and water level

```python
df_station_dict = {}
for station in settings.FINAL_STATIONS:
    wl = df_wl[[station]]
    rd = (ds_glofas_reanalysis[station + settings.VERSION_LOC]
              .to_dataframe()
              .drop(columns=['step', 'surface', 'valid_time'])
              .rename(columns={f"{station+settings.VERSION_LOC}": station}))
    data = (pd.merge(wl, rd, 
                     how='right', 
                     left_index=True, 
                     right_index=True, 
                     suffixes=['_wl', '_rd'])
            #.dropna()
            .rename(columns={f"{station}_wl": "water_level",
                    f"{station}_rd": "river_discharge"})
           )
    # Fill in the gaps so that the group finding works - not sure this is needed
    data = data.reindex(pd.date_range(data.index.min(), data.index.max()))
    # Add in the forecast data
    for leadtime in LEADTIMES:
        for percentile in FORECAST_PERCENTILE_LIST:
            forecast = (ds_glofas_forecast_summary[station + settings.VERSION_LOC]
                    .sel(leadtime=leadtime, percentile=percentile)
                    .to_dataframe()
                    .drop(columns=['surface', 'leadtime', 'percentile'])
                    .rename(columns={f"{station+settings.VERSION_LOC}": station}))
            data = (pd.merge(data, forecast,
                        how='left',
                        left_index=True,
                        right_index=True,
                        )
               .rename(columns={station: f"forecast_lt{leadtime}_p{percentile}"}))
    # Keep only the forecast date range
    data = data.dropna(subset=["forecast_lt1_p50"])
    # Get the water level events
    level_type = "danger"
    level_val = df_station_info.at[station, f'{level_type}_level']
    events = utils.get_groups_above_threshold(data['water_level'], level_val, min_duration=settings.DURATION)
    event_start_indices = [event[0] for event in events]
    data[f"event_{level_type}"] = False
    data[f"event_{level_type}"].iloc[event_start_indices] = True
    # Get river discharge events
    for rp in RP_LIST:
        rp_val = df_return_period.loc[rp, station]
        events = utils.get_groups_above_threshold(data['river_discharge'], rp_val, min_duration=settings.DURATION)
        event_start_indices = [event[0] for event in events]
        data[f"event_rp{rp}"] = False
        data[f"event_rp{rp}"].iloc[event_start_indices] = True
        # Go through each date and add activations
        for percentile in FORECAST_PERCENTILE_LIST:
            for event_type, leadtimes in LEADTIMES_BY_TRIGGER.items():
                data[f'event_{event_type}_rp{rp}_p{percentile}'] = data.apply(lambda row: 
                                          (row[[f'forecast_lt{leadtime}_p{percentile}' for leadtime in leadtimes]] >= rp_val).any(),
                                          axis=1)
    df_station_dict[station] = data
data
```

### Get TP, FP, and FN

```python
# Find TP, FP and FN

def get_consecutive_groups(x: pd.Series, n=2):
    # Find indices whre False / True change. Skip the first entry of the array because it always changes there
    l = np.where(x.shift() != x)[0][1:]
    # Convert array to pairs
    return [l[i:i+n] for i in range(0, len(l), n)]


def get_station_stats(df_station_dict, event_var, rp=settings.MAIN_RP, percentiles=None):

    days_before_buffer = 0 # Event can occur at the earliest on the same day as the trigger
    df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN', 'event_type', 'percentile'])

    if percentiles is None:
        percentiles = FORECAST_PERCENTILE_LIST
    
    for station in settings.FINAL_STATIONS:
        df_station = df_station_dict[station]
        if event_var == 'event_danger':
            df_station = df_station.dropna(subset=['water_level'])
        true_event_dates = df_station[df_station[event_var]][[event_var]].index
        
        rp_val = df_return_period.loc[rp, station]
        for event_type, leadtimes in LEADTIMES_BY_TRIGGER.items():
            for percentile in percentiles:
                glofas_event_dates = [df_station.index[start_index] for start_index, end_index in 
                                        get_consecutive_groups(df_station[f"event_{event_type}_rp{rp}_p{percentile}"])]
                detection_stats = utils.get_detection_stats(true_event_dates=true_event_dates,
                                                       forecasted_event_dates=glofas_event_dates,
                                                       days_before_buffer=settings.DAYS_BEFORE_BUFFER,
                                                       days_after_buffer=settings.DAYS_AFTER_BUFFER + min(leadtimes))
                
                df_station_stats = df_station_stats.append({
                    **{'station': station,
                    'event_type': event_type,
                    'percentile': 100 - percentile},
                    **detection_stats
                }, ignore_index=True)

    df_station_stats = utils.get_more_detection_stats(df_station_stats)
    return df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_danger", rp=2)
df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_rp2", rp=2)
df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_danger", rp=5, percentiles=[50])
df_station_stats
```

```python
df_station_stats = get_station_stats(df_station_dict, "event_rp5", rp=5, percentiles=[50])
df_station_stats
```

### Get list of historical activations

```python
# Go through each date and list the activations

event_bools = {
    'readiness': False,
    'action': False,
    #'rp2': False
    'danger': False
}
rp = 2

for station in settings.FINAL_STATIONS:
    for percentile in FORECAST_PERCENTILE_LIST:
        print(station)
        print(100-percentile)
        df_station = df_station_dict[station]#.copy().dropna(subset=["water_level"])
        df_events = pd.DataFrame(columns=['event', 'leadtimes'])
        rp_val = df_return_period.loc[rp, station]
        for date, row in df_station.iterrows():
            for event_type in event_bools.keys():
                cname = f'event_{event_type}'
                if event_type in ['readiness', 'action']:
                    cname += f"_rp{rp}_p{percentile}"
                if row[cname]:
                    if not event_bools[event_type]:
                        s = 'flood' if event_type == 'danger' else event_type + ' trigger'
                        print(date, s)
                        # Find lead time
                        if event_type in ['readiness', 'action']:
                            forecast_leadtimes = []
                            for leadtime in LEADTIMES_BY_TRIGGER[event_type]:
                                if row[f'forecast_lt{leadtime}_p{percentile}'] > rp_val:
                                    forecast_leadtimes += [leadtime]
                            df_events.loc[date, 'leadtimes'] = forecast_leadtimes
                            #print(forecast_leadtimes)
                    event_bools[event_type] = True
                else:
                    event_bools[event_type] = False
```

### Plots

```python
# Try to plot the single readiness TP, and see which percentiles capture it
rp = 2
x = ds_glofas_forecast_summary['Chisapani_v3'].sel(leadtime=4, percentile=[20, 25, 30, 35, 40, 45, 50])
rp_val = df_return_period.loc[rp, 'Chisapani']


fig, ax = plt.subplots()
x.plot.line(hue='percentile', alpha=0.5)
ax.set_xlim(np.datetime64('2009-08-10'), np.datetime64('2009-08-20'))
ax.axhline(rp_val, c='k')
ax.set_ylim(4000, 8000)
```

```python
# Plot timeline of events

percentile = 50
mpl.rcParams['hatch.linewidth'] = 0.5
for station in settings.STATIONS_FINAL:
    df_station = df_station_dict[station].copy().dropna(subset=["water_level"])
    thresh = df_station_info.at[station, f'danger_level']
    rp_val = df_return_period.loc[settings.MAIN_RP, station]
    years = df_station.index.year.unique()
    fig, axs = plt.subplots(len(years), figsize=(8, 20))
    for year, ax in zip(years, axs):
        data = df_station[df_station.index.year == year]
        # Plot everything
        ax.plot(data.index, data.water_level, c='k', label='water level', alpha=0.75, lw=0.75)
        ax2 = ax.twinx()
        ax2.plot(data.index, data.river_discharge, c='C0', label='river_discharge', alpha=0.5, lw=0.5)
        ax.plot([], [], 'C0-', label='river discharge', alpha=0.5, lw=0.5)
        ax.axhline(y=thresh, c='C3', label='danger / 1 in 2 y', alpha=0.75, lw=0.5)
        
        y0 = -0.05 * thresh
        y1 = thresh*1.2
        ax.set_ylim(y0, y1)
        ax2.set_ylim(-0.05 * rp_val, rp_val*1.2)
        ax.set_xticklabels([])
        ax.text(data.index[0], thresh + 0.2, f"{year}")
        ax.grid()
        # Add the activations
        for event_type, hatch, colour, lw in zip(['readiness', 'action','danger'], 
                                             ['/////', '\\\\\\\\\\', '.....'], 
                                             ['C2', 'C1', 'C6'],
                                             [0.5, 0.5, 2]):
            cname = f'event_{event_type}'
            if event_type != "danger":
                cname += f"_rp{settings.MAIN_RP}_p{percentile}"
            ax.fill_between(data.index, y0, y1, where=data[cname], 
                            facecolor='none', hatch=hatch, edgecolor=colour, lw=lw, 
                            alpha=.75, label=event_type)
    fig.supylabel('Water level [m]')
    fig.suptitle(station)    
    ax.legend()
```
