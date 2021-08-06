# High water-level flood events

We use the water level data to define flood events as when the water level goes 
above the danger / warning level as defined by DHM. We want to know how
often these events correspond to a GloFAS RP exceedance. 

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import npl_settings as settings
from src.indicators.flooding.glofas import utils

pd.options.mode.chained_assignment = None  # default='warn'
```

```python
RP_LIST = [1.5, 2, 5]
WL_DAYS = [1, 2, 3, 4, 5] # How many days earlier the warning level is reached
```

```python
df_station_info = pd.read_excel(settings.DHM_STATION_INFO_FILENAME, index_col='station_name')
df_wl = pd.read_csv(settings.WL_OUTPUT_FILENAME, index_col='date')


ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=settings.COUNTRY_ISO3, use_incorrect_area_coords=settings.USE_INCORRECT_AREA_COORDS)

ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = settings.COUNTRY_ISO3, leadtimes=settings.LEADTIMES,
    interp=True,
    use_incorrect_area_coords=settings.USE_INCORRECT_AREA_COORDS
)
ds_glofas_forecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)

df_return_period =  pd.read_excel(settings.GLOFAS_RP_FILENAME, index_col='rp')

```

### Create dataframe with both water level and river discharge

```python
df_station_dict = {}
for station in settings.FINAL_STATIONS:
    wl = df_wl[[station]]
    rd = (ds_glofas_reanalysis[station + settings.VERSION_LOC]
              .to_dataframe()
              .drop(columns=['step', 'surface', 'valid_time'])
              .rename(columns={f"{station+settings.VERSION_LOC}": station}))
    data = (pd.merge(wl, rd, 
                     how='inner', 
                     left_index=True, 
                     right_index=True, 
                     suffixes=['_wl', '_rd'])
            .dropna()
            .rename(columns={f"{station}_wl": "water_level",
                    f"{station}_rd": "river_discharge"})
           )
    # Fill in the gaps so that the group finding works
    data = data.reindex(pd.date_range(data.index.min(), data.index.max()))
    # Add in the forecast data
    for leadtime in settings.LEADTIMES:
        forecast = (ds_glofas_forecast_summary[station + settings.VERSION_LOC]
                    .sel(leadtime=leadtime, percentile=settings.MAIN_FORECAST_PROB)
                    .to_dataframe()
                    .drop(columns=['surface', 'leadtime', 'percentile'])
                    .rename(columns={f"{station+settings.VERSION_LOC}": station}))
        data = (pd.merge(data, forecast,
                        how='left',
                        left_index=True,
                        right_index=True,
                        )
               .rename(columns={station: f"forecast_{leadtime}"}))
    # Get the water level events
    for level_type in settings.LEVEL_TYPES:
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
    # Get water at warning level n days previously events
    # Only do 2 year RP for now
    for wl_day in WL_DAYS:
        rp = settings.MAIN_RP
        rp_val = df_return_period.loc[rp, station]
        condition = data.shift(wl_day)['water_level'] >  df_station_info.at[station, f'warning_level']
        events = utils.get_groups_above_threshold(data['river_discharge'], rp_val, min_duration=settings.DURATION,
                                                 additional_condition=condition)
        event_start_indices = [event[0] for event in events]
        data[f"event_rp{rp}_wl{wl_day}"] = False
        data[f"event_rp{rp}_wl{wl_day}"].iloc[event_start_indices] = True
    # Get forecast events
    for leadtime in settings.LEADTIMES:
        rp = settings.MAIN_RP
        events = utils.get_groups_above_threshold(data[f'forecast_{leadtime}'], rp_val, min_duration=settings.DURATION)
        event_start_indices = [event[0] for event in events]
        data[f"event_forecast{leadtime}"] = False
        data[f"event_forecast{leadtime}"].iloc[event_start_indices] = True
    df_station_dict[station] = data
data
```

```python
# Plot river dischage against WL
for station in settings.FINAL_STATIONS:    
    data = df_station_dict[station]
    fig, ax = plt.subplots()
    ax.plot(data.water_level, data.river_discharge, '.')
    idx = data['event_warning'] == True
    ax.plot(data.water_level[idx], data.river_discharge[idx], 'xr')
```

### Compare water danger level events to GloFAS RP exceedanc

Want to check how many years with events for a given level type or RP
to make sure we're comparing similar types of events

```python
for station in settings.FINAL_STATIONS:
    print(station)
    df_station = df_station_dict[station]
    df_station = df_station.groupby(df_station.index.year).sum() > 0
    print(f"Total years: {len(df_station)}")
    for level_type in settings.LEVEL_TYPES:
        n = df_station[f"event_{level_type}"].sum()
        print(f"{level_type} level: {n}")
    for rp in RP_LIST:
        n = df_station[f"event_rp{rp}"].sum()
        print(f"1 in {rp} y: {n}")
    print('\n')
```

Unfortuantely there is some mismatch. 
For Chatara, it seems that warning level corresponds to 1 in 1.5 year, and danger level to 1 in 2 year.
For chisapani, warning level is closer to 1 in 2 year (probably more like 1 in 3 year) and danger level to 1 in 5 year. 

```python
# Settle on RP and level type
rp = settings.MAIN_RP
event_level_type = 'danger'

df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN', 'wl_days'])

for station in settings.FINAL_STATIONS:
    df_station = df_station_dict[station]
    for wl_days in [None] + WL_DAYS:
        wl_events = df_station[df_station[f"event_{event_level_type}"]].index
        if wl_days is None:
            glofas_cname = f"event_rp{rp}"
        else:
            glofas_cname = f"event_rp{rp}_wl{wl_days}"
        glofas_events = df_station[df_station[glofas_cname]].index
        if wl_days is None:
            print(f"{station}")
            print(f"True events: {len(df_true_events)}")
            print(f"Glofas events: {len(glofas_events)}")
        
        
        detection_stats = utils.get_detection_stats(true_event_dates=wl_events,
                                                    forecasted_event_dates=glofas_events,
                                                    days_before_buffer=settings.DAYS_BEFORE_BUFFER,
                                                    days_after_buffer=settings.DAYS_AFTER_BUFFER)
        df_station_stats = df_station_stats.append({
                **{'station': station,
                   'wl_days': wl_days},
                **detection_stats
            }, ignore_index=True)
df_station_stats[df_station_stats['wl_days'].isnull()]
```

## Make plots for presentation

```python
def plot_arrow(ax, x, y, c):
    ax.annotate(" ", 
                     xy=(x, y+y*0.3),
                     xytext=(x, y+y*0.5),
                    arrowprops=dict(facecolor=c, shrink=0.05, headlength=3,
                               width=1, headwidth=3, lw=0.5, alpha=0.5))
    
for station in settings.FINAL_STATIONS:
    df = df_station_dict[station]
    fig, (ax2, ax1) = plt.subplots(2)
    fig.suptitle(station)
    # Water level
    
    ax1.plot(df['water_level'])
    level_val = df_station_info.at[station, f'{level_type}_level']
    idx = df['water_level'] >= level_val
    #ax1.plot(df.loc[idx, 'water_level'], '.', c='C3')
    ax1.axhline(level_val, c='C3')
    for event in list(df.index[df[f'event_{event_level_type}']]):
        plot_arrow(ax1, event, level_val, 'C3')
    
    ax1.set_ylabel('DHM water level [m]')
    ax1.set_ylim(None, level_val+ 0.7 * level_val)
   
    # River discharge
    ax2.plot(df['river_discharge'], c='C2')
    rp_val = df_return_period.loc[rp, station]
    idx = df['river_discharge'] >= rp_val
    #ax2.plot(df.loc[idx, 'river_discharge'], '.', c='C1')
    ax2.axhline(rp_val, c='C1')
    for event in list(df.index[df[f'event_rp{rp}']]):
        plot_arrow(ax2, event, rp_val, 'C1')
    
    ax2.set_ylabel('GloFAS river discharge [m$^3$ s$^{-1}$]')
    ax2.set_ylim(None, rp_val + 0.7 * rp_val)
```

### Check difference when using additional warning level condition

To reduce FPs we add the condition that the water level has to be at the warning level
N days prior, and check the results for various N

```python
for istation, station in enumerate(settings.FINAL_STATIONS):
    qs = ['TP', 'FP', 'FN']
    fig, ax = plt.subplots()
    data = df_station_stats[(df_station_stats['station'] == station)]
    data.loc[data['wl_days'].isnull(), 'wl_days'] = 0
    for iq, q in enumerate(qs):
        ax.plot(data['wl_days'], data[q], label=q)
    ax.set_title(station)
    ax.set_xlabel('Warning level reached N days before')
    ax.set_ylabel('Number')
    # Make legend
    ax.legend()

```

Check time between warning and danger level, and how often warning level is reached without reaching danger level

```python

for station, df_station in df_station_dict.items():
    warning_level = df_station_info.at[station, f'warning_level']
    danger_level = df_station_info.at[station, f'danger_level']
    events = df_station[df_station['event_warning']]
    date_diff = []
    n_danger_reached = 0
    n_danger_not_reached = 0
    true_warning_dates = []
    false_warning_dates = []
    #rint('warning and danger', warning_level, danger_level)
    for date_warning, _ in events.iterrows():
        #rint('event', date_warning)
        df = df_station[df_station.index >= date_warning]
        for date_danger, row_danger in df.iterrows():
            #rint(date_danger, row_danger['water_level'])
            if row_danger['water_level'] >= danger_level:
                date_diff.append((date_danger - date_warning) / np.timedelta64(1, 'D'))
                true_warning_dates.append(date_warning)
                n_danger_reached += 1
                break
            elif row_danger['water_level'] < warning_level:
                false_warning_dates.append(date_warning)
                n_danger_not_reached +=1 
                break
    print(f'{station}')
    print(f'Number of times danger level reached: {n_danger_reached}')
    print(f'Number of times danger level not reached: {n_danger_not_reached}')
    # Plot true and false warnings
    fig, ax = plt.subplots(figsize=(10,2))
    ax.set_title(station)
    ax.plot(df_station.index, df_station['water_level'], lw=0.5, c='k', alpha=0.5)
    ax.set_ylabel('Water level [m]')
    ax.plot(true_warning_dates, len(true_warning_dates) * [warning_level], '^C2', alpha=0.5, label='True')
    ax.plot(false_warning_dates, len(false_warning_dates) * [warning_level], 'xC0', label='False')
    ax.axhline(warning_level, c='C1', alpha=0.5)
    ax.axhline(danger_level, c='C3', alpha=0.5)
    ax.legend()
    
    # Plot time between warning and danger
    fig, ax = plt.subplots()
    ax.set_title(station)
    ax.hist(date_diff, bins=np.arange(0, 6, 1)-0.5)
    ax.set_xlabel('Number of days danger level is reached after warning level')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Number of events')
```

## Compare to forecast

As previously, compute the TP, FP, and FN fore the different forecast leadtimes.
Modify the days before buffer to not exceed the forecast length.

```python
# Settle on RP and level type
rp = settings.MAIN_RP
event_level_type = 'danger'

df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN', 'leadtime'])

for station in settings.FINAL_STATIONS:
    df_station = df_station_dict[station]
    df_station = df_station.dropna(subset=[f'forecast_{leadtime}' for leadtime in settings.LEADTIMES])
    for leadtime in [0] + settings.LEADTIMES:

        wl_events = df_station[df_station[f"event_{event_level_type}"]].index
        if leadtime == 0:
            glofas_cname = f"event_rp{rp}"
        else:
            glofas_cname = f"event_forecast{leadtime}"
        glofas_events = df_station[df_station[glofas_cname]].index

        detection_stats = utils.get_detection_stats(true_event_dates=wl_events,
                                                    forecasted_event_dates=glofas_events,
                                                    days_before_buffer=settings.DAYS_BEFORE_BUFFER,
                                                    days_after_buffer=settings.DAYS_AFTER_BUFFER)
        df_station_stats = df_station_stats.append({
            **{'station': station,
            'leadtime': leadtime},
            **detection_stats
        }, ignore_index=True)

        
df_station_stats[df_station_stats['leadtime'].isin([3, 7])]
```

```python
# Plot TP, FP, FN
for istation, station in enumerate(settings.FINAL_STATIONS):
    qs = ['TP', 'FP', 'FN']
    fig, ax = plt.subplots()
    data = df_station_stats[(df_station_stats['station'] == station)]
    for iq, q in enumerate(qs):
        ax.plot(data['leadtime'], data[q], label=q)
    ax.set_title(station)
    ax.set_xlabel('Lead time [days]')
    ax.set_ylabel('Number')
    ax.set_ylim(-0.5, None)
    # Make legend
    ax.legend()

```

```python
rp = settings.MAIN_RP
leadtimes = [7, 3] # Longer first

for station in settings.FINAL_STATIONS:

    df_station = df_station_dict[station]
    df_station = df_station.dropna(subset=[f'forecast_{leadtime}' for leadtime in settings.LEADTIMES])

    observations = df_station['water_level']
    model = df_station['river_discharge']
    forecast_1 = df_station[f'forecast_{leadtimes[0]}']
    forecast_2 = df_station[f'forecast_{leadtimes[1]}']
    
    fig = plt.figure(figsize=(10,8))
    gs= fig.add_gridspec(ncols=1, nrows=4, hspace=0.05, top=0.93, bottom=0.08)

    fig.supylabel('River dischange [m$^3$ s$^{-1}$]')
    fig.supxlabel('Year')
    fig.suptitle(station)
    x = df_station.index


    for i, q in enumerate([forecast_1, forecast_2, model, observations]):
        ax =  plt.subplot(gs[i])
        if i == 3:
            c1 = 'k'
            c2 = 'C3'
            thresh = df_station_info.at[station, f'{level_type}_level']
            ax.set_ylabel('Water level [m]')
        else:
            if i == 2:
                c1 = 'C0'
            else:
                c1 = 'C4'
            c2 = 'C1'
            ax.set_xticklabels([])
            thresh = df_return_period.loc[rp, station]
            ax.set_ylim(0, thresh*1.2)
        ax.plot(x, q, '-', c=c1, lw=0.5)
        ax.minorticks_on()
        ax.axhline(thresh, c=c2, lw=0.5)
        for detection in utils.get_groups_above_threshold(q, thresh, settings.DURATION):
            ax.plot(x[detection[0]], q[detection[0]], 'o', c=c2, lw=2, mfc='none')
```
