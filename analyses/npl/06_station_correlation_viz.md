We want to know if an event happens at a particular station, what is the river 
discharge at stations in neighbouring basins

```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr

import npl_parameters as parameters
from src.indicators.flooding.glofas import utils
```

```python
RP_LIST = [1.5, 2, 5] 
```

#### Define functions to make overlay plots

We want to understand what happens at Station B when Station A triggers according to a given threshold. 

```python
def make_max_rp_hist(ax, hist_vals, n_bins, title, xlab):
    ax.hist(hist_vals, bins=n_bins)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.axvline(1, c='red', ls='--')
    ax.set_xlim([0, 2])
    return ax

def visualize_station_overlay(
    da_primary, 
    da_secondary, 
    val_event, 
    val_secondary, 
    duration, 
    days_buffer,
    plt_title, 
    hist_title, 
    save_title_line,
    save_title_hist,
    save=False,
    make_histograms=False):

    groups = utils.get_groups_above_threshold(da_primary.values, val_event, duration)

    if len(groups) == 0:
        print('There are no events captured at this  threshold')
        return
    
    max_post = []
    max_during = []
    max_pre = []

    ngroups = len(groups)
    nrows = 2
    ncols = round(ngroups/nrows)
    igroup = 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(15,5), squeeze=False, sharey=True)
    fig.suptitle(plt_title)

    for i in range(nrows):

        for j in range(ncols):

            try:
                group = groups[igroup-1]
                
                # Values extend 15 days on either side of trigger event and normalize by the rp val
                da_event_buffer = da_secondary[group[0]-days_buffer: (group[-1]+days_buffer)] / val_secondary
                # Values just for the trigger event
                da_event = da_secondary[group[0]: group[-1]+1]
                # Values at the primary station for comparison
                da_primary_buffer = da_primary[group[0]-days_buffer: (group[-1]+days_buffer)] / val_event
                
                # Keep track of the max value for each stage of the event
                max_post.append(max(da_event_buffer[-days_buffer:-1].values))
                max_during.append(max(da_event_buffer[days_buffer:-days_buffer].values))
                max_pre.append(max(da_event_buffer[0:days_buffer].values))

                ax = axs[i,j]
                ax.axes.xaxis.set_ticks([])
                ax.set_ylim([0, 1.5])

                trigger_start_formatted = pd.to_datetime(da_event[0].time.values).date()
                trigger_end_formatted = pd.to_datetime(da_event[-1].time.values).date()
                ax.set_title(f'{trigger_start_formatted} - {trigger_end_formatted}', fontsize=8)
                ax.fill_between(x=da_event.time, y1=0, y2=1.5,alpha=0.15)

                ax.hlines(y=1, xmin=da_event_buffer.time[0], xmax=da_event_buffer.time[-1], lw=1, ls='--')
                ax.plot(da_event_buffer.time, da_event_buffer.values, c='red', lw=1, alpha=1)
                ax.plot(da_primary_buffer.time, da_primary_buffer.values, c='red',ls='--', lw=0.40, alpha=0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)                

            # In case we have too many subplots defined
            except IndexError as e:
                print(f'Deleting a subplot...{e}')
                fig.delaxes(axs[i,j])

            igroup+=1

    fig.tight_layout()

    if save:
        plt.savefig(save_title_line)

    # Make the histograms of max % of RP reached
    if make_histograms:
        n_bins = 5
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
        hist_vals = [max_pre, max_during, max_post]
        hist_titles = ['15 Days Before Trigger Period', 'During Trigger Period', '15 Days After Trigger Period']

        for iax, ax in enumerate(axs): 
            if np.isnan(hist_vals[iax]).all():
                fig.delaxes(ax)
            else:
                ax = make_max_rp_hist(ax, hist_vals[iax], n_bins, hist_titles[iax], hist_title)         

        if save:
            plt.savefig(save_title_hist)
```

#### Investigate relationship between stations with GloFAS water discharge

```python
# Choosing use_incorrect_area_coords for now because still using old GloFAS
# data downloaded non-x.x5 extent coordinates
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=parameters.COUNTRY_ISO3, use_incorrect_area_coords=True)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis, RP_LIST)
df_return_period_glofas = pd.read_excel(parameters.GLOFAS_RP_FILENAME)
```

```python
rp_event = 5 
rp_secondary = 5 
primary_stations = ['Asaraghat', 'Chisapani']
secondary_stations = ['Kusum', 'Chepang'] 
days_buffer = 15

for station in primary_stations:
    
    #rp_val_event = int(df_return_period_glofas.loc[df_return_period_glofas['rp']==rp_event, station])
    rp_val_event = df_return_period.loc[rp_event, station]
    da_primary = ds_glofas_reanalysis[station + parameters.VERSION_LOC]
        
    for station_small in secondary_stations:
        
        rp_val_secondary = df_return_period.loc[rp_secondary, station_small]  
        da_secondary = ds_glofas_reanalysis[station_small + parameters.VERSION_LOC]
        
        plt_title = f'What is water discharge at {station_small} when {station} triggers at {rp_event}-year RP?'
        save_title_line = f'line_{station_small}_{station}_{rp_secondary}_{rp_event}.png'
        save_title_hist = f'max_hist_{station_small}_{station}_{rp_secondary}_{rp_event}.png'
        hist_title = f'Max % of {rp_secondary}-year RP reached'
        
        visualize_station_overlay(
            da_primary, 
            da_secondary, 
            rp_val_event, 
            rp_val_secondary, 
            parameters.DURATION, 
            days_buffer,
            plt_title, 
            hist_title, 
            save_title_line,
            save_title_hist,
            False)
```

#### Investigate relationship between stations with water level

```python
df_wl = pd.read_csv(parameters.WL_OUTPUT_FILENAME, index_col='date')
df_wl.index = pd.to_datetime(df_wl.index).rename('time')
ds_wl = xr.Dataset.from_dataframe(df_wl)
df_station_info = pd.read_excel(parameters.DHM_STATION_INFO_FILENAME, index_col='station_name')
```

```python
thresh_event = 'danger'
days_buffer = 15

stations = ['Chisapani', 'Kusum', 'Chepang']

for station in stations:
    
    val_event = df_station_info.at[station, f'{thresh_event}_level']
    da_primary = ds_wl[station]
        
    for station_small in stations:
        if station_small == station:
            continue
        val_secondary = df_station_info.at[station_small, f'{thresh_event}_level']
        da_secondary = ds_wl[station_small]
        
        plt_title = f'What is the water level at {station_small} when {station} triggers at the {thresh_event} level?'
        save_title_line = f'line_{station_small}_{station}_{thresh_event}.png'
        save_title_hist = f'max_hist_{station_small}_{station}_{thresh_event}.png'
        hist_title = f'Max % of {thresh_event} level reached'      
        
        visualize_station_overlay(
            da_primary, 
            da_secondary, 
            val_event, 
            val_secondary, 
            parameters.DURATION, 
            days_buffer,
            plt_title, 
            hist_title, 
            save_title_line,
            save_title_hist,
            False)
```

## Added in 2022

The countrty team has proposed adding some municipalities in the 
Mahana Basin. We want to check if the flood events there could 
be picked up by the trigger in Chisapani, which is part of the Karnali
river basin. To do this we check the GloFAS discharge at 
[Kandra](https://hydrology.gov.np/#/basin/4635?_k=ep79ty) and compare
to Chispani. 

```python
rp_event = 5
rp_secondary = 5
stations = ['Chisapani_v3', 'Kandra']
secondary_stations = [
    'Kandra', 'Chisapani_v3'
] 
days_buffer = 15

for station, station_small in zip(stations, secondary_stations):
    
    rp_val_event = df_return_period.loc[rp_event, station]
    da_primary = ds_glofas_reanalysis[station]

    rp_val_secondary = df_return_period.loc[rp_secondary, station_small] 
    da_secondary = ds_glofas_reanalysis[station_small]

    plt_title = f'What is water discharge at {station_small} when {station} triggers at {rp_event}-year RP?'
    save_title_line = f'line_{station_small}_{station}_{rp_secondary}_{rp_event}.png'
    save_title_hist = f'max_hist_{station_small}_{station}_{rp_secondary}_{rp_event}.png'
    hist_title = f'Max % of {rp_secondary}-year RP reached'

    visualize_station_overlay(
        da_primary, 
        da_secondary, 
        rp_val_event, 
        rp_val_secondary, 
        parameters.DURATION, 
        days_buffer,
        plt_title, 
        hist_title, 
        save_title_line,
        save_title_hist,
        False)


```

The analysis shows that flood events at Kandra do not show a 
strong correspondence with those at Chisapani. Therefore we recommend
not to include municipalities in the Mahana basin. 

```python

```
