We have a file from the RCO that has flood eents only in the municipalities 
of interest. Here we want to correlate these events with past GloFAS activations.

```python
from pathlib import Path
import os
import sys
from importlib import reload

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
reload(utils)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None 
```

```python
COUNTRY_ISO3 = 'npl'

STATIONS = {
    'Koshi': 'Chatara',
    'Karnali': 'Chisapani'
}
# Use "_v3" for the GloFAS model v3 locs, or empty string for the original v2 ones
VERSION_LOC = "_v3"

KARNALI_BASINS = [
    'Karnali', 'West Rapti', 'Babai'
]
KOSHI_WATERSHED = 'Saptakoshi'

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DATA_DIR_PRIVATE = DATA_DIR / 'private'
DATA_DIR_PUBLIC = DATA_DIR / 'public'
RCO_DIR = DATA_DIR_PRIVATE / 'exploration/npl/unrco'
SHAPEFILE_DIR = DATA_DIR_PUBLIC / 'raw/npl/cod_ab'


BASINS_SHAPEFILE = RCO_DIR / 'shapefiles/Major_River_Basins.shp'
WATERSHED_SHAPEFILE = RCO_DIR / 'shapefiles/Major_watershed.shp'
ADMIN_SHAPEFILE = SHAPEFILE_DIR / 'npl_admbnda_ocha_20201117/npl_admbnda_nd_20201117_shp.zip'
#ADMIN2_SHAPEFILE = 'npl_admbnda_districts_nd_20201117.shp'
ADMIN2_SHAPEFILE = 'npl_admbnda_adm2_nd_20201117.shp'

PAST_EVENTS_FILENAME = RCO_DIR / 'HistoricalReportedIncident_Municipality Level_AAPilot.xlsx'

GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"

MUNICIPALITIES_OUTPUT_GEOPACKAGE = RCO_DIR / 'shapefiles/municipalities_of_interest.gpkg'
```

### Read in and clean data

Read in GloFAS data

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
df_return_period =  pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')
```

Read in events and clean

```python
df_events = (pd.read_excel(PAST_EVENTS_FILENAME, index_col='S.No.')
             .rename(columns={'Mun_ETHOS_Code': 'pcode'})
            )
# Drop events with no date and convert date column
df_events = df_events.loc[df_events['Year'] > 0]
# For days that are 0, assume they mean the middle of the month
df_events['Day'] = np.where(df_events['Day'] == 0, 15, df_events['Day'])
df_events['Incident Date'] = pd.to_datetime(df_events[['Year', 'Month', 'Day']])

# Only select events in the time range of GloFAS
df_events = df_events.loc[df_events['Incident Date'] > ds_glofas_reanalysis.time[0].data]
```

Merge event data with admin

```python
df_admin = (gpd.read_file(f'zip://{ADMIN_SHAPEFILE}!{ADMIN2_SHAPEFILE}')
            .rename(columns={'ADM2_PCODE': 'pcode'}))

df_events = pd.merge(df_admin, df_events, how='right', left_on='pcode', right_on='pcode')
```

Read in Basin and Watershed to get areas of interest

```python
df_basins = gpd.read_file(BASINS_SHAPEFILE)
df_watershed = gpd.read_file(WATERSHED_SHAPEFILE)
```

```python
df_basins['Karnali'] = 0
idx = df_basins['Major_Basi'].isin([f'{basin} River Basin' for basin in KARNALI_BASINS])
df_basins.loc[idx, 'Karnali']= 1
karnali = df_basins[idx].dissolve(by='Karnali')
koshi = df_watershed[df_watershed['WSH_NME'] == f'{KOSHI_WATERSHED} Watershed']
```

Plot event municipalities over the areas of interest

```python
fig, ax = plt.subplots()
karnali.plot(ax=ax)
koshi.plot(ax=ax)
df_events.boundary.plot(ax=ax, color='r', lw=0.5)
```

Might be easier to just do a latitude cut. Latitude > 86 is Koshi, < 83 is Karnali.

```python
df_events['basin'] = np.nan
df_events.loc[df_events['geometry'].bounds.minx > 86,'basin'] = 'Koshi'
df_events.loc[df_events['geometry'].bounds.minx < 83,'basin'] = 'Karnali'
df_events = df_events.dropna(subset=['basin'])
```

### Combine events with GloFAS

```python
# Take only the columns with impact and group by the basin and incident date and sum
df_events_final = df_events[df_events.columns[24:43]].groupby(['basin', 'Incident Date']).sum()

```

```python
# Since there are two many events, get only those that are high impact
# Use a rolling sum
rolling_sum_size = 10
center = False
event_thresh = 0
impact_parameters =  ['Total Death', 
                         'Affected Family', 
                         'Private House Partially Damaged', 
                         'Private House Fully Damaged']
plot_results = False

df_events_high_impact = pd.DataFrame(columns=['date', 'basin', 'impact_parameter'])

for basin, station in STATIONS.items():
    # Reindex the events to the reanalysis time
    # and get the rolling sum of impact
    df = df_events_final.loc[basin]
    df = (df.reindex(ds_glofas_reanalysis.time.data, fill_value=0)
            .rolling(rolling_sum_size, center=center).sum())
    if plot_results:
        fig, axs = plt.subplots(len(impact_parameters), figsize=(5, 12))
        fig.suptitle(basin)
    for i, impact_parameter in enumerate(impact_parameters):
        # Define event threshold as n x standard deviation
        groups = utils.get_groups_above_threshold(df[impact_parameter], event_thresh)
        # Take the date of the event as the first index
        df_events_high_impact = df_events_high_impact.append(
            [{"basin": basin,
             "impact_parameter": impact_parameter,
             "date": df.index[group[0]] 
            } for group in groups],
            ignore_index=True
        )
        if plot_results:
            ax = axs[i]
            ax.plot(df.index, df[impact_parameter], label=impact_parameter, c=f'C{i}')
            ax.legend()
```

```python
# Define GloFAS events
rp = 2
ndays = 1 # Number of consecutive days above RP
days_before_buffer = 5 # Number of days before GloFAS event the flooding event can occur
days_after_buffer = 60

df_station_stats = pd.DataFrame(columns=['station', 'impact_parameter', 'TP', 'FP', 'FN'])

for basin, station in STATIONS.items():
    for impact_parameter in impact_parameters:
        df_events_sub = df_events_high_impact[
            (df_events_high_impact['basin'] == basin) & (df_events_high_impact['impact_parameter'] == impact_parameter)
        ]
        df_events_sub['detections'] = 0
        TP = 0
        FP = 0
        rp_val = df_return_period.loc[rp, station]
        observations = ds_glofas_reanalysis[station + VERSION_LOC].values
        groups = utils.get_groups_above_threshold(observations, rp_val, min_duration=ndays)
        for group in groups:
            # The GlofAS event takes place on the Nth day (since for an event)
            # you require N days in a row
            glofas_event_date = ds_glofas_reanalysis.time[group[0] + ndays - 1].data
            # Check if any events are around that date
            days_offset = (df_events_sub['date'] - glofas_event_date) /  np.timedelta64(1, 'D')
            detected = (days_offset >= -1 * days_before_buffer) & (days_offset <= days_after_buffer)
            df_events_sub.loc[detected, 'detections'] += 1
            # If there were any detections, it's  a TP. Otherwise a FP
            if sum(detected):
                TP += 1
            else:
                FP += 1
        df_station_stats = df_station_stats.append({
            'station': station,
            'impact_parameter': impact_parameter,
            'TP': TP,
            'FP': FP,
            'FN': len(df_events_sub[df_events_sub['detections'] == 0])
        }, ignore_index=True)
            
            
df_station_stats['precision'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FP'].astype(int))
df_station_stats['recall'] = df_station_stats['TP'].astype(int) / (df_station_stats['TP'].astype(int) + df_station_stats['FN'].astype(int))
```

```python
df_station_stats.groupby('station').mean()
#df_station_stats
```

```python
for basin, station in STATIONS.items():
    fig, axs = plt.subplots(len(impact_parameters), figsize=(12, 10))
    fig.suptitle(f'{basin} - {station}')
    fig.supylabel('Discharge [m$^3$ s$^{-1}$]')
    for i, impact_parameter in enumerate(impact_parameters):
        df_events_sub = df_events_high_impact[
            (df_events_high_impact['basin'] == basin) & (df_events_high_impact['impact_parameter'] == impact_parameter)
        ]
        observations = ds_glofas_reanalysis[station + VERSION_LOC].values
        x = ds_glofas_reanalysis.time
        ax = axs[i]
        ax.plot(x, observations, lw=0.5)
        # Plot when GLoFAS is above RP
        rp_val=df_return_period.loc[rp, station]
        groups = utils.get_groups_above_threshold(observations, rp_val, ndays)
        for group in groups:
            idx = range(group[0], group[1])
            ax.plot(x[idx], observations[idx], ls='-', 
                    lw=2, c='C2')
        ax.axhline(y=rp_val, c='C1')
        # Plot events
        for _, row in df_events_sub.iterrows():
            ax.plot(row['date'], rp_val, 'o', c='C3', mfc='none')
        ax.set_title(impact_parameter)
                
```

## Make a shapefile of the municipalities of interest

Not super related but I didn't want to make a new notebook for this

```python
municipalities_list = [
"Barahchhetra",
'Bhokraha Narshingh',
'Harinagar',
'Inaruwa',
'Koshi',
"Agnisair Krishna Savaran",
'Chhinnamasta',
'Hanumannagar Kankalini',
'Kanchanrup',
'Mahadeva',
'Rajbiraj',
"Shambhunath",
'Tilathi Koiladi',
'Tirahut',
'Duduwa',
'Narainapur',
'Nepalgunj',
'Rapti Sonari',
'Bansagadhi',
'Barbardiya',
'Geruwa',
'Gulariya',
'Rajapur',
'Thakurbaba',
'Lamkichuha',
"Dhangadhi",
'Tikapur',
]

```

```python
(df_admin[
    df_admin['ADM2_EN']
    .isin(municipalities_list)]
     .to_file(MUNICIPALITIES_OUTPUT_GEOPACKAGE, layer='municipalities', driver="GPKG")
)
```
