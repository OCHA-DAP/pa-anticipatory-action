### Visualizing flood events with GloFAS streamflow

This notebook visualizes GloFAS streamflow (historical and forecasted) during dates when a historical flood event has been identified by an external dataset. The objective of this is to better understand the quality of the external event dataset and general agreement with GloFAS streamflow. 

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
import matplotlib.dates as mdates

from matplotlib.ticker import MaxNLocator

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.glofas import utils

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

LEADTIMES = [x + 1 for x in range(10)]
SAVE_PLOT = False
EVENT = 'RCO' # 'rco', 'floodscan', 'combined'
COUNTRY_ISO3 = 'mwi'

stations_adm2 = {
    #'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}
```

Read in the processed GloFAS data.

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES,
    interp=True
)
ds_glofas_reforecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis, method='analytical')
```

Read in the processed event dataset(s). Here, we're using the combined RCO events from both Chikwawa and Nsanje. 

```python
events = {}
for station in stations_adm2.values():
    events[station] = pd.read_csv(EXPLORE_DIR / f'all_rco_event_summary.csv')
```

Create a plot for each event. For the dates around each flood onset, we want to see what both the GloFAS reanalysis (historical) and reforecast data look like. We'll pick a specific forecast probability and include forecasts from various leadtimes. The y-axis is scaled to the percentage of a given return period threshold. Lines on the graph have a stronger weight when the values exceed the threshold.

```python
buffer_start = 15
buffer_end = 30
forecast_probability = 70
rp = 2
colors = plt.cm.Blues(np.linspace(0.4,1,len(LEADTIMES)))

for code, station, in stations_adm2.items():
    
    nevents = len(events[station].index)
    nrows = 6
    ncols = round(nevents/nrows)
    igroup = 1 
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20), squeeze=False, sharey=True)
    
    for i in range(nrows):

        for j in range(ncols):
            
            try: 
                ax = axs[i,j]
                ax.axes.xaxis.set_ticks([])
                ax.set_ylim([0.25, 1.25])

                flood_onset_date = pd.to_datetime(events[station].loc[igroup-1, 'start_date'])
                glofas_start_date = flood_onset_date - np.timedelta64(buffer_start, 'D')
                glofas_end_date = flood_onset_date + np.timedelta64(buffer_end, 'D')
                rp_val = df_return_period.loc[rp, code]

                for ix, leadtime in enumerate(LEADTIMES):
                    glofas_reforecast_sel = (
                        ds_glofas_reforecast_summary[code]
                        .sel(leadtime=leadtime)
                        .sel(percentile=forecast_probability)
                        .sel(time=slice(glofas_start_date, glofas_end_date))

                    ) / rp_val

                    reforecast_trigger = np.where(glofas_reforecast_sel.values >= 1, glofas_reforecast_sel.values, None)

                    ax.plot(glofas_reforecast_sel.time, glofas_reforecast_sel.values, color=colors[ix], lw=0.7, alpha=0.3)
                    ax.plot(glofas_reforecast_sel.time, reforecast_trigger, color=colors[ix], lw=1.5)

                glofas_reanalysis_sel = ds_glofas_reanalysis[code].sel(time=slice(glofas_start_date, glofas_end_date)) / rp_val


                reanalysis_trigger = np.where(glofas_reanalysis_sel.values >= 1, glofas_reanalysis_sel.values, None)

                ax.plot(glofas_reanalysis_sel.time, glofas_reanalysis_sel.values, color='red', lw=1, alpha=0.3)
                ax.plot(glofas_reanalysis_sel.time, reanalysis_trigger, color='red', lw=1)
                ax.plot((flood_onset_date, flood_onset_date), (0.25, 1.25), lw=0.6, color='black')
                
                ax.set_title(f'{flood_onset_date.date()}', fontsize=8)


                myFmt = mdates.DateFormatter('%d-%m-%y')
                ax.xaxis.set_major_formatter(myFmt)
                
            except Exception as e:
                print(f'Deleting a subplot...{e}')
                fig.delaxes(axs[i,j])

            igroup+=1    
            
    fig.tight_layout()
```
