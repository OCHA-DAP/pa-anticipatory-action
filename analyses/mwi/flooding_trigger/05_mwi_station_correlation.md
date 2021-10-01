### Comparing trigger events between stations

This notebook explores the correlation between trigger events in Chikwawa and those at Nsanje. We're trying to understand whether a trigger at Chikwawa will likely also correspond to a trigger at Nsanje. 

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

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
from src.indicators.flooding.config import Config

config = Config()

mpl.rcParams['figure.dpi'] = 300

COUNTRY_ISO3 = 'mwi'

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

SAVE_FIG = True
LEADTIMES = [x + 1 for x in range(10)]
stations_adm2 = {
    'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}
```

Read in the GloFAS data and calculate the return periods. 

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

Define functions to make plots to compare GloFAS data between stations.

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
    plt_title, 
    hist_title, 
    save_title_line,
    save_title_hist,
    save=False):

    groups = utils.get_groups_above_threshold(da_primary.values, val_event, duration)

    if len(groups) == 0:
        print('There are no events captured at this  threshold')
        return
    
    max_post = []
    max_during = []
    max_pre = []

    ngroups = len(groups)
    nrows = 4
    ncols = round(ngroups/nrows)
    igroup = 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(15,10), squeeze=False, sharey=True)
    fig.suptitle(plt_title)

    for i in range(nrows):

        for j in range(ncols):

            try:
                group = groups[igroup-1]
                
                # Values extend 15 days on either side of trigger event and normalize by the rp val
                da_event_buffer = da_secondary[group[0]-15: (group[-1]+15)] / val_secondary
                # Values just for the trigger event
                da_event = da_secondary[group[0]: group[-1]+1]
                # Values at the primary station for comparison
                da_primary_buffer = da_primary[group[0]-15: (group[-1]+15)] / val_event
                
                # Keep track of the max value for each stage of the event
                max_post.append(max(da_event_buffer[-15:-1].values))
                max_during.append(max(da_event_buffer[15:-15].values))
                max_pre.append(max(da_event_buffer[0:15].values))

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
            except Exception as e:
                print(f'Deleting a subplot...{e}')
                fig.delaxes(axs[i,j])

            igroup+=1

    fig.tight_layout()

    if save:
        plt.savefig(PLOT_DIR / save_title_line)

    # Make the histograms of max % of RP reached
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
        plt.savefig(PLOT_DIR / save_title_hist)
```

```python
rp_event = 2 
rp_secondary = 2 
primary_stations = ['G2001']
secondary_stations = ['G1724']
primary_label = 'Chikwawa'
secondary_label = 'Nsanje'
DURATION = 3

for station in primary_stations:
    
    rp_val_event = df_return_period.loc[rp_event, station]
    da_primary = ds_glofas_reanalysis[station]
        
    for station_small in secondary_stations:
        
        rp_val_secondary = df_return_period.loc[rp_secondary, station_small]  
        da_secondary = ds_glofas_reanalysis[station_small]
        
        plt_title = f'What is water discharge at {secondary_label} when {primary_label} triggers at {rp_event}-year RP?'
        save_title_line = f'line_{secondary_label}_{primary_label}_{rp_secondary}_{rp_event}.png'
        save_title_hist = f'max_hist_{secondary_label}_{primary_label}_{rp_secondary}_{rp_event}.png'
        hist_title = f'Max % of {rp_secondary}-year RP reached'
        
        visualize_station_overlay(
            da_primary, 
            da_secondary, 
            rp_val_event, 
            rp_val_secondary, 
            DURATION, 
            plt_title, 
            hist_title, 
            save_title_line,
            save_title_hist,
            SAVE_FIG)
```

Calculate the time lag in hours between stations.

```python
time = ds_glofas_reanalysis.time.data
day_range = int((time[-1] - time[0]) / np.timedelta64(1,'D'))
x = np.arange(-day_range, day_range + 1)
# Number of days around the offset interpolation
interp_offset = 100

stations = ['G1724', 'G2001']

offset_df = pd.DataFrame(index=stations, columns=stations)
offset_df.index.name = 'station'

corr_df = pd.DataFrame(index=stations, columns=stations)
corr_df.index.name = 'station'


# Get the discharge for each station
discharge_s1 = ds_glofas_reanalysis[stations[0]] 
discharge_s2 = ds_glofas_reanalysis[stations[1]]
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
offset_df.at[stations[0], stations[1]] = int(offset) 
```

```python
offset_df
```
