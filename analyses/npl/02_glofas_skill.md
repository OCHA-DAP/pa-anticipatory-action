```python
from pathlib import Path
import os
from importlib import reload
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
# chdir because otherwise can't import get_glofas_data
# should only run this once
os.chdir(path_mod)

from src.indicators.flooding.glofas import utils, glofas
import src.nepal.get_glofas_data as ggd

reload(utils)

mpl.rcParams['figure.dpi'] = 200


COUNTRY_ISO3 = 'npl'
STATIONS = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat'],
    'Rapti': ['Kusum'],
    'Bagmati': ['Rai_goan'],
    'Babai': ['Chepang']
}
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=ggd.LEADTIMES,
    interp=False
)
```

## Return period

```python
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
rp_label = [str(int(x)) for x in df_return_period.index]
rp_label[0] = '1.5'
for basin, stations in STATIONS.items():
    fig, ax = plt.subplots()
    ax.set_title(basin)
    for station in stations:
        rp = df_return_period[station]
        ax.plot(rp_label, rp, 'o-', label=station)
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
    ax.legend()
```

## Skill

```python
def plot_crps(df_crps, title_suffix=None):
    for basin, stations in STATIONS.items():
        fig, ax = plt.subplots()
        for station in stations:
            crps = df_crps[station]
            ax.plot(crps.index, crps, label=station)
        ax.legend()
        title = basin
        if title_suffix is not None:
            title += title_suffix
        ax.set_title(title)
        ax.set_xlabel("Lead time [days]")
        ax.set_ylabel("Normalized CRPS [% error]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

```

```python
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                        normalization="mean")
plot_crps(df_crps * 100, title_suffix=" -- all discharge values")
```

```python
rp = 1.5
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                         normalization="mean", 
                         thresh=df_return_period.loc[rp].to_dict())
plot_crps(df_crps * 100, title_suffix=f" -- values > RP 1 in {rp} y")
```

## When do activations occur

```python
rp_list = [1.5, 2, 5, 10, 20]
cmap = mpl.cm.get_cmap('plasma_r')
clist = cmap(np.linspace(0, 1, len(rp_list)))
cdict = {rp: c for rp, c in zip(rp_list, clist)}
legend_title = 'RP'
```

### Plot river discharge vs time for all stations

```python
for basin in ['Koshi', 'Karnali']:
    stations = STATIONS[basin]
    fig, axs = plt.subplots(len(stations), figsize=(10,2*len(stations)))
    fig.suptitle(basin)
    fig.supylabel('Discharge [m$^3$ s$^{-1}$]')
    for istation, station in enumerate(stations):
        observations = ds_glofas_reanalysis[station].values
        x = ds_glofas_reanalysis.time

        ax = axs[istation]
        ax.plot(x, observations, c='k', lw=0.5, alpha=0.5)

        for rp in rp_list:
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val)
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
year_ranges = [
    [1979, 1988],
    [1989, 1998],
    [1999, 2009],
    [2010, 2021]
]
#year_ranges = [[x, x+1] for x in range(1979, 2020)]

rp_list = [1.5, 2, 5, 10, 20]
cmap = mpl.cm.get_cmap('plasma_r')
clist = cmap(np.linspace(0, 1, len(rp_list)))

for basin in ['Koshi', 'Karnali']:

    stations = STATIONS[basin]
    fig, axs = plt.subplots(len(year_ranges), figsize=(15, 10))
    fig.suptitle(basin)
    for iyear, year_range in enumerate(year_ranges):
        ax = axs[iyear]
        ax.axes.yaxis.set_ticks([])
        for istation, station in enumerate(stations):
            ds = (ds_glofas_reanalysis
                            .sel(time=slice(f'{year_range[0]}-01-01', 
                                            f'{year_range[1]}-01-01')))
            observations = ds[station].values
            x = ds.time

            for rp in rp_list:
                rp_val=df_return_period.loc[rp, station]
                groups = utils.get_groups_above_threshold(observations, rp_val)
                for group in groups:
                    idx = range(group[0], group[1])
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
for basin in ['Koshi', 'Karnali']:
    event_dict = {}
    for rp in rp_list:
        event_dict[rp] = {}
        ievent = 0
        for istation, station in enumerate(STATIONS[basin]):
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
