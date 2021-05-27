```python
from pathlib import Path
from pathlib import Path
import os
import sys

import pandas as pd
import geopandas as gpd
import numpy as np


path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils, glofas

COUNTRY_ISO3 = 'npl'
STATIONS = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat'],
    'West Rapti': ['Kusum'],
    'Bagmati': ['Rai_goan'],
    'Babai': ['Chepang']
}

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DATA_DIR_PUBLIC = DATA_DIR / 'public'
DATA_DIR_PRIVATE = DATA_DIR / 'private'
RCO_DIR = DATA_DIR_PRIVATE / 'exploration/npl/unrco'
SHAPEFILE_DIR = DATA_DIR_PUBLIC / 'raw/npl/cod_ab'

PAST_EVENTS_FILENAME = RCO_DIR / 'NepalHistoricalFlood1971-2020.xlsx' 
BASINS_SHAPEFILE = RCO_DIR / 'shapefiles/Major_River_Basins.shp'
WATERSHED_SHAPEFILE = RCO_DIR / 'shapefiles/Major_watershed.shp'
ADMIN_SHAPEFILE = SHAPEFILE_DIR / 'npl_admbnda_ocha_20201117/npl_admbnda_nd_20201117_shp.zip'
ADMIN2_SHAPEFILE = 'npl_admbnda_adm2_nd_20201117.shp'
```

```python
# Read in events and admin, 
# make pcode column names match for simplicity
df_events = (pd.read_excel(PAST_EVENTS_FILENAME)
             .rename(columns={'DIST_CODE_ETHOS': 'pcode'}))
df_admin = (gpd.read_file(f'zip://{ADMIN_SHAPEFILE}!{ADMIN2_SHAPEFILE}')
            .rename(columns={'DIST_PCODE': 'pcode'}))

# Drop events with no date and convert date column
df_events = df_events.loc[df_events['Year'] > 0]
# For days that are 0, assume they mean the middle of the month
df_events['Day'] = np.where(df_events['Day'] == 0, 15, df_events['Day'])
df_events['Incident Date'] = pd.to_datetime(df_events[['Year', 'Month', 'Day']])
```

```python
# Read in basin files
df_basins = gpd.read_file(BASINS_SHAPEFILE)
df_watershed = gpd.read_file(WATERSHED_SHAPEFILE)
```

```python
# Create the target region file 
df_target_regions = gpd.GeoDataFrame(columns=["name", "districts", "geometry"])
for basin in ["Karnali", "Babai", "Bagmati", "West Rapti"]:
    df_target_regions = df_target_regions.append(
    {"name": basin, 
    "geometry": df_basins.loc[df_basins['Major_Basi'] == f'{basin} River Basin', 'geometry'].iloc[0]},
    ignore_index=True
    )
# Get intersection between Saptakoshi Watershed and Koshi Basin
saptakoshi = df_watershed.loc[df_watershed['WSH_NME'] == 'Saptakoshi Watershed', 'geometry'].iloc[0]
koshi = df_basins.loc[df_basins['Major_Basi'] == 'Koshi River Basin', 'geometry'].iloc[0]

df_target_regions = df_target_regions.append(
    {"name": "Koshi", 
    "geometry": koshi.union(saptakoshi)},
    ignore_index=True
)
```

```python
# For each basin, get a list of affected regions
df_target_regions["districts"] = df_target_regions['geometry'].apply(lambda x:
                                                       [y['pcode']
                                                        for _, y in df_admin.iterrows()
                                                        if x.intersects(y['geometry'])
                                                        ])
```

```python
# Check that final list makes sense
target_districts = list(set(np.concatenate(df_target_regions['districts'])))
df_admin[df_admin['pcode'].isin(target_districts)].plot()
```

```python
# Create df of events for each basin
df_events_dict = {
    row['name']: df_events.loc[df_events['pcode'].isin(row['districts'])]
                               for _, row in df_target_regions.iterrows()
}

numerical_columns = list(df_events.columns[13:31])
for basin, df in df_events_dict.items():
    # Reduce to date and numerical columns, and combine same date
    df = (df[['Incident Date'] + numerical_columns]
          .groupby('Incident Date').sum())
    df_events_dict[basin] = df
```

## Compare historical events to GloFAS

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
# Define GloFAS events
rp = 2
ndays = 3 # Number of consecutive days above RP
days_before_buffer = 5 # Number of days before GloFAS event the flooding event can occur
days_after_buffer = 30

df_station_stats = pd.DataFrame(columns=['station', 'TP', 'FP', 'FN'])

for basin, station_list in STATIONS.items():
    df_events = df_events_dict[basin]
    for station in station_list:
        df_events['detections'] = 0
        TP = 0
        FP = 0
        rp_val = df_return_period.loc[rp, station]
        observations = ds_glofas_reanalysis[station].values
        groups = utils.get_groups_above_threshold(observations, rp_val, min_duration=ndays)
        for group in groups:
            # The GlofAS event takes place on the Nth day (since for an event)
            # you require N days in a row
            glofas_event_date = ds_glofas_reanalysis.time[group[0] + ndays - 1]
            # Check if any events are around that date
            #x = df_events.index - glofas_event_date
            days_offset = (glofas_event_date.data - df_events.index).days
            detected = (days_offset > -1 * days_before_buffer) & (days_offset < days_after_buffer)
            df_events.loc[detected, 'detections'] += 1
            # If there were any detections, it's  a TP. Otherwise a FP
            if sum(detected):
                TP += 1
            else:
                FP += 1
        df_station_stats = df_station_stats.append({
            'station': station,
            'TP': TP,
            'FP': FP,
            'FN': len(df_events[df_events['detections'] == 0])
        }, ignore_index=True)
        
```

```python
df_station_stats
```
