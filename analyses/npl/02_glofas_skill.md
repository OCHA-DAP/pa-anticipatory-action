```python
from pathlib import Path
import os
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib as mpl

#path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
#os.chdir(path_mod)

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
for basin in ['Koshi', 'Karnali']:

    stations = STATIONS[basin]
    fig, axs = plt.subplots(len(stations), figsize=(10,2*len(stations)))

    for istation, station in enumerate(stations):
        observations = ds_glofas_reanalysis[station].values
        x = ds_glofas_reanalysis.time

        ax = axs[istation]
        ax.plot(x, observations, c='k', lw=0.5, alpha=0.5)

        for irp, rp in enumerate([1.5, 2, 5, 10, 20]):
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val)
            for group in groups:
                idx = range(group[0], group[1])
                ax.plot(x[idx], observations[idx], ls='-', lw=0.7, c=f'C{irp}')
            ax.axhline(y=rp_val, c=f'C{irp}', lw=0.5, alpha=0.5)

        ax.text(x[10], 300, station)
        if i == 1:
            ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')

```

```python
mpl.rcParams['figure.dpi'] = 200
year_ranges = [
    [1979, 1988],
    [1989, 1998],
    [1999, 2009],
    [2010, 2021]
]
#year_ranges = [[x, x+1] for x in range(1979, 2020)]

rp_list = [5, 10, 20]
for basin in ['Koshi', 'Karnali']:

    stations = STATIONS[basin]
    fig, axs = plt.subplots(len(year_ranges), figsize=(15, 10))
    for iyear, year_range in enumerate(year_ranges):
        ax = axs[iyear]
        for istation, station in enumerate(stations):
            ds = (ds_glofas_reanalysis
                            .sel(time=slice(f'{year_range[0]}-01-01', 
                                            f'{year_range[1]}-01-01')))
            observations = ds[station].values
            x = ds.time

            for irp, rp in enumerate(rp_list):
                rp_val=df_return_period.loc[rp, station]
                groups = utils.get_groups_above_threshold(observations, rp_val)
                for group in groups:
                    idx = range(group[0], group[1])
                    ax.fill_between(x=x[idx], y1=istation, y2=istation+1, fc=f'C{irp}', alpha=0.5)

            #ax.text(x[10], 300, station)
            if i == 1:
                ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, len(stations))

```

```python
ds_glofas_reanalysis.sel(time=slice('1980-01-01', '2000-01-01'))
ds_glofas_reanalysis.time
```

```python
def get_distance_between_ranges(r1, r2):
    # sort the two ranges such that the range with smaller first element
    # is assigned to x and the bigger one is assigned to y
    if r1[0] < r2[0]:
        x, y = r1, r2
    else:
        x, y = r2, r1

    #now if x[1] lies between x[0] and y[0](x[1] != y[0] but can be equal to x[0])
    #then the ranges are not overlapping and return the differnce of y[0] and x[1]
    #otherwise return 0 
    if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (r1,r2)):
        return y[0] - x[1]
    return 0


x = ds_glofas_reanalysis.time.values
rp_list = [1.5, 2, 5, 10, 20]
rp_list = [5, 10, 20]


days_buffer = 5
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
                    event_dict[rp][tuple(group)] = [group]
                else:
                    # Get distance between the group and all known events
                    distances = {event_date: get_distance_between_ranges(event_date, group) 
                                for event_date in event_dict[rp].keys()}
                    # Get minimum distance
                    min_distance = distances[min(distances, key=distances.get)]
                    
                    if min_distance < days_buffer:
                        matching_event = min(distances, key=distances.get)
                        event_dict[rp][matching_event].append(group)
                    else:
                        event_dict[rp][tuple(group)] = [group]
                    
                        
        event_dict[rp] = {x[event_dates[0]]: len(events) for event_dates, events in event_dict[rp].items()}
    
    fig, ax = plt.subplots()
    for rp, offset in zip(rp_list, [0, 0.05, 0.1]):
        ax.plot(event_dict[rp].keys(), np.array(list(event_dict[rp].values())) + offset, 
                'o', label=rp, mfc='none', alpha=1)
    ax.legend()
```

```python

```

```python

```
