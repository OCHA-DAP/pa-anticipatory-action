```python
from pathlib import Path
import os
import sys
from collections import Counter

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils, glofas

mpl.rcParams['figure.dpi'] = 300
```

```python
COUNTRY_ISO3 = 'npl'

DURATION = 1 # Number of days that the GloFAS threshold should be exceeded

RP_LIST = [2, 3, 5] # List of return period values to compare

STATIONS = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar', 'Kampughat'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat'],
    'Rapti': ['Kusum'],
    'Bagmati': ['Rai_goan'],
    'Babai': ['Chepang']
}
STATIONS_BY_MAJOR_BASIN = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar', 'Kampughat', 'Rai_goan'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat', 'Kusum', 'Chepang'],
}

STATIONS_SEL = {
    'Karnali': ['Chisapani', 'Asaraghat'],
    'Rapti': ['Kusum'],
    'Babai': ['Chepang']
}

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis, RP_LIST)
df_return_period_glofas = pd.read_excel(GLOFAS_RP_FILENAME)
```

## When do activations occur at the different stations

We want to know how much stations within each basin are correlated, and if activation events in the past have occured simultaneously, so that we could have a multi-station trigger

```python
cmap = mpl.cm.get_cmap('plasma_r')
clist = cmap(np.linspace(0, 1, len(RP_LIST)))
cdict = {rp: c for rp, c in zip(RP_LIST, clist)}
legend_title = 'RP'
```

```python
events_all = {}

for basin in ['Koshi', 'Karnali', 'Rapti', 'Bagmati', 'Babai']:
    
    station_events = {}   
    stations = STATIONS[basin]
    
    for istation, station in enumerate(stations):  
        
        rp_events = {}
        
        for rp in RP_LIST:
            rp_val=df_return_period.loc[rp, station]
            df_activations = utils.get_glofas_activations(ds_glofas_reanalysis[station], rp_val, DURATION)
            rp_events[rp] = df_activations
            
        station_events[station] = rp_events
        
    events_all[basin] = station_events
```

### Plot river discharge vs time for all stations

```python
rp_list = RP_LIST
for basin, stations in STATIONS_SEL.items():
    fig, axs = plt.subplots(len(stations), figsize=(10,2*len(stations)), squeeze=False)
    fig.suptitle(basin)
    #fig.supylabel('Discharge [m$^3$ s$^{-1}$]')
    for istation, station in enumerate(stations):
        observations = ds_glofas_reanalysis[station].values
        x = ds_glofas_reanalysis.time

        ax = axs[istation, 0]
        ax.plot(x, observations, c='k', lw=0.5, alpha=0.5)

        for rp in rp_list:
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val, DURATION)
            for group in groups:
                idx = range(group[0], group[1])
                ax.plot(x[idx], observations[idx], ls='-', 
                        lw=0.7, c=cdict[rp])
            ax.axhline(y=rp_val, c=cdict[rp], lw=0.5, alpha=0.5)

        ax.text(x[10], 300, station)
        if istation == 0:
            for rp in rp_list:
                ax.plot([], [], c=cdict[rp], label=rp)
            ax.legend(title=legend_title)
```

### Plot RP exceedance

```python
# TODO: Something seems a bit buggy here, 
# as some stations don't have any events for a 1/3 RP

year_ranges = [
    [1979, 1988],
    [1989, 1998],
    [1999, 2009],
    [2010, 2021]
]

#year_ranges = [[x, x+1] for x in range(1979, 2020)]
year_ranges = [[1979, 2020]]

buffer = 5 # Buffer to make sure the lines show up
rp_list = RP_LIST
cmap = mpl.cm.get_cmap('plasma_r')
clist = cmap(np.linspace(0, 1, len(rp_list)))

all_events = {}

for basin, stations in STATIONS_SEL.items():
    
    basin_events = {}

    fig, axs = plt.subplots(len(year_ranges), figsize=(15, 10), squeeze=False)
    fig.suptitle(basin)
    for iyear, year_range in enumerate(year_ranges):
        ax = axs[iyear, 0]
        ax.axes.yaxis.set_ticks([])
        for istation, station in enumerate(stations):
            
            station_events = {}
            
            ds = (ds_glofas_reanalysis
                            .sel(time=slice(f'{year_range[0]}-01-01', 
                                            f'{year_range[1]}-01-01')))
            observations = ds[station].values
            x = ds.time

            for rp in rp_list:
                rp_val=df_return_period.loc[rp, station]
                groups = utils.get_groups_above_threshold(observations, rp_val, DURATION)
                df_activations = utils.get_glofas_activations(ds[station], rp_val, DURATION)
                #print(df_activations)
                for group in groups:
                    idx = range(group[0], group[1] + buffer)
                    ax.fill_between(x=x[idx], y1=istation, y2=istation+1, 
                                    fc=cdict[rp], alpha=1)

            ax.text(x[10], istation+0.5, station)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, len(stations))
        if iyear == 0:
            for rp in rp_list:
                ax.plot([], [], c=cdict[rp], label=rp)
            ax.legend(title=legend_title)

```

### Find events occuring at stations simultaneously

```python
def get_distance_between_ranges(r1, r2):
    if r1[0] < r2[0]:
        x, y = r1, r2
    else:
        x, y = r2, r1
    if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (r1,r2)):
        return y[0] - x[1]
    return 0


days_buffer = 30
time = ds_glofas_reanalysis.time.values

# Algorithm: Take an event date range and save it as the key.
# For each subsequent event, check if the distance beween the ranges
# is < days_buffer days. If yes, add the name of the station
# to the event, if not, make a new event in the dictionary.
# Note that this is pretty crude because it only uses the 
# date range of the initial event, and thus will depend on the order
# of the event list.
for basin, stations in STATIONS_BY_MAJOR_BASIN.items():
    event_dict = {}
    for rp in rp_list:
        event_dict[rp] = {}
        ievent = 0
        for istation, station in enumerate(stations):
            observations = ds_glofas_reanalysis[station].values
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val)
            for group in groups:
                if len(event_dict[rp]) == 0:
                    event_dict[rp][tuple(group)] = [station]
                else:
                    # Get distance between the group and all known events
                    distances = {event_date: get_distance_between_ranges(event_date, group) 
                                for event_date in event_dict[rp].keys()}
                    # Get minimum distance
                    min_distance = distances[min(distances, key=distances.get)]
                    
                    if min_distance < days_buffer:
                        matching_event = min(distances, key=distances.get)
                        event_dict[rp][matching_event].append(station)
                    else:
                        event_dict[rp][tuple(group)] = [station]
                    
                        
        event_dict[rp] = {time[event_dates[0]]: len(set(events)) for event_dates, events in event_dict[rp].items()}
    
    # Plot number of simultaneous stations vs time
    fig, ax = plt.subplots()
    for irp, rp in enumerate(rp_list):
        offset = irp * 0.005
        ax.plot(event_dict[rp].keys(), np.array(list(event_dict[rp].values())) + offset, 
                'o', label=rp, mfc='none', alpha=0.5, c=cdict[rp])
    ax.set_title(basin)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of stations')
    ax.legend(title='RP')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    
    # Plot number of events vs number of simultaneous stations
    fig, ax = plt.subplots()
    for rp in rp_list:
        incidence = Counter(list(event_dict[rp].values()))
        x, y = np.array(list(incidence.items())).T
        y_sorted = [z for _,z in sorted(zip(x,y))]
        x = np.sort(x)
        ax.plot(x, y, '-o', c=cdict[rp], label=rp)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title='RP')
    ax.set_title(basin)
    ax.set_xlabel('Number of stations')
    ax.set_ylabel('Number of events')
        
```

## Find correlations between station observations

```python

time = ds_glofas_reanalysis.time.data
day_range = int((time[-1] - time[0]) / np.timedelta64(1,'D'))
x = np.arange(-day_range, day_range + 1)
# Number of days around the offset interpolation
interp_offset = 100


#for basin, stations in STATIONS_BY_MAJOR_BASIN.items():
offset_df = pd.DataFrame(index=stations, columns=stations)
offset_df.index.name = 'station'

corr_df = pd.DataFrame(index=stations, columns=stations)
corr_df.index.name = 'station'

stations = []
for s in STATIONS_BY_MAJOR_BASIN.values():
    stations += s
for station1 in stations:
    for station2 in stations:
        # Get the discharge for each station
        discharge_s1 = ds_glofas_reanalysis[station1] 
        discharge_s2 = ds_glofas_reanalysis[station2]
        corr = correlate(discharge_s1, discharge_s2)
        imax = np.argmax(corr)
        offset_crude = x[imax]
        # Do interpolation to get offset in hours
        x_sub = x[imax - interp_offset:imax + interp_offset + 1] * 24
        corr_sub = corr[imax - interp_offset:imax + interp_offset + 1]
        corr_interp_func = interp1d(x_sub, corr_sub, kind='cubic')
        x_hours_sub = np.arange(x_sub[0], x_sub[-1] + 1)
        corr_hours = corr_interp_func(x_hours_sub)
        offset = int(x_hours_sub[np.argmax(corr_hours)])
        offset_df.at[station1, station2] = int(offset)          
        # Do an hourly interpolation and shift
        df_discharge = (pd.DataFrame({'station1': discharge_s1.values,
                                    'station2': discharge_s2.values}, 
                                    index=pd.date_range(time[0], time[-1]))
                        .asfreq('h')
                        .interpolate(method='cubic'))
        df_discharge['station2_shifted'] = df_discharge['station2'].shift(offset)
        df_discharge = (df_discharge
                        .asfreq('d')
                        .dropna()
           )
        # Then calculate pearsons correlation
        corr_df.at[station1, station2] = df_discharge.corr().loc['station1', 'station2_shifted']
corr_df = corr_df.astype('float')
```

```python
offset_df.style.background_gradient(cmap='Greens_r', low=-500, high=500)
```

```python
corr_df.style.background_gradient(cmap='Blues_r')
```

## Are activations between basins correlated? 

Checking Koshi + Bagmati and Karnali + Babai + West Rapti. For each event in the secondary basin, check to see what the shortest time is to the nearest event in the primary basin.

```python
selected_stations = {
    'Chatara': 'Koshi',
    'Asaraghat':'Karnali'
}

correlated_stations = {
    'Chatara': ['Bagmati'], 
    'Asaraghat': ['Babai', 'Rapti']}

RP = 3
```

```python
small_basin_distances = {}

for large_basin_station in correlated_stations: 
    
    for small_basin in correlated_stations[large_basin_station]:
        
        large_basin_events = events_all[selected_stations[large_basin_station]][large_basin_station][RP]
        small_basin_events = events_all[small_basin][STATIONS[small_basin][0]][RP]
      
        for index, row in small_basin_events.iterrows():
            
            small_event_start = small_basin_events.loc[index, 'start_index']
            start_distances = [small_event_start - large_event_start for large_event_start in large_basin_events['start_index']]
            min_distance = min(start_distances, key=abs)
            small_basin_events.loc[index, 'min_distance'] = min_distance
        
        plt.figure()
        plt.hist(small_basin_events.min_distance)
        
        # What percent of events occur within +-30 days?
        perc_short_distance = (small_basin_events.min_distance.abs()<30).sum() / len(small_basin_events.min_distance)
        
        # Save to new df in case wanting to do more with this
        small_basin_distances[small_basin] = small_basin_events
```
