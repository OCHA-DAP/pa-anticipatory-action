```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import npl_settings as settings
from src.indicators.flooding.glofas import utils, glofas

pd.options.mode.chained_assignment = None 
```

```python
PAST_EVENTS_FILENAME = settings.RCO_DIR / 'NepalHistoricalFlood1971-2020.xlsx' 
BASINS_SHAPEFILE = settings.RCO_DIR / 'shapefiles/Major_River_Basins.shp'
WATERSHED_SHAPEFILE = settings.RCO_DIR / 'shapefiles/Major_watershed.shp'

EVENTS_FOR_RAGINDRA = settings.DATA_DIR_PUBLIC / 'exploration/npl/glofas/npl_glofas_events.xlsx'
```

### Read in data and clean slightly

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=settings.COUNTRY_ISO3)
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)
```

```python
# Read in events and admin, 
# make pcode column names match for simplicity
df_events = (pd.read_excel(PAST_EVENTS_FILENAME)
             .rename(columns={'DIST_CODE_ETHOS': 'pcode'}))
df_admin = (gpd.read_file(f'zip://{settings.ADMIN_SHAPEFILE}!{settings.ADMIN2_SHAPEFILE}')
            .rename(columns={'DIST_PCODE': 'pcode'}))
```

```python
# Cleaning events data
# Drop events with no date and convert date column
df_events = df_events.loc[df_events['Year'] > 0]
# For days that are 0, assume they mean the middle of the month
df_events['Day'] = np.where(df_events['Day'] == 0, 15, df_events['Day'])
df_events['Incident Date'] = pd.to_datetime(df_events[['Year', 'Month', 'Day']])

# Only select events in the time range of GloFAS
df_events = df_events.loc[(df_events['Incident Date'] > ds_glofas_reanalysis.time[0].data)
                         & (df_events['Incident Date'] < ds_glofas_reanalysis.time[-1].data)]
```

```python
# Plot the number of events per destrict
x = df_events.rename(columns={'Year': 'num_events'}).groupby('pcode').count()['num_events']
df_admin.join(x, on='pcode', how='left').plot(column='num_events', legend=True)
```

```python
# Read in basin files
df_basins = gpd.read_file(BASINS_SHAPEFILE)
df_watershed = gpd.read_file(WATERSHED_SHAPEFILE)
```

### Get sub-set of events based on basin pcodes

```python
# Create the target region file 
df_target_regions = gpd.GeoDataFrame(columns=["name", "districts", "geometry"]).set_crs('EPSG:4326')
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
# that intersect some fraction with the basin
# Chose 25% because larger leaves large gaps in river basins
intersection_thresh = 0.25
df_target_regions["districts"] = df_target_regions['geometry'].apply(lambda x:
                                                       [y['pcode']
                                                        for _, y in df_admin.iterrows()
                                                        if x.intersection(y['geometry']).area > intersection_thresh * y['geometry'].area
                                                        ])
```

```python
# Check that final list makes sense
target_districts = list(set(np.concatenate(df_target_regions['districts'])))
fig, ax = plt.subplots()
df_admin[df_admin['pcode'].isin(target_districts)].plot(ax=ax, fc='g', alpha=0.5, ec='g')
df_target_regions.plot(ax=ax, fc='b', alpha=0.5, ec='b')
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

### Compare historical events to GloFAS

```python
# Since there are two many events, get only those that are high impact
# Use a rolling sum
n_std = 5
impact_parameters =  ['Total Death', 
                         'Missing People', 
                         'Affected Family', 
                         'Private House Partially Damaged', 
                         'Private House Fully Damaged']

df_events_high_impact = pd.DataFrame(columns=['date', 'basin', 'impact_parameter'])

for basin, df in df_events_dict.items():
    # Reindex the events to the reanalysis time
     # and get the rolling sum of impact
    df = (df.reindex(ds_glofas_reanalysis.time.data, fill_value=0)
            .rolling(30, center=True).sum())
    fig, axs = plt.subplots(len(impact_parameters), figsize=(5, 12))
    fig.suptitle(basin)
    for i, impact_parameter in enumerate(impact_parameters):
        # Define event threshold as n x standard deviation
        thresh = n_std * df[impact_parameter].std()
        groups = utils.get_groups_above_threshold(df[impact_parameter], thresh)
        # Take the date of the event as the first index
        df_events_high_impact = df_events_high_impact.append(
            [{"basin": basin,
             "impact_parameter": impact_parameter,
             "date": df.index[group[0]] 
            } for group in groups],
            ignore_index=True
        )
        ax = axs[i]
        ax.plot(df.index, df[impact_parameter], label=impact_parameter, c=f'C{i}')
        ax.legend()
```

```python
for basin, df in df_events_dict.items():
    # Reindex the events to the reanalysis time
    # and get the rolling sum of impact
    df = (df.reindex(ds_glofas_reanalysis.time.data, fill_value=0)
            .rolling(5, center=True).sum()).fillna(0)

    fig, axs = plt.subplots(len(impact_parameters), figsize=(5, 12))
    fig.suptitle(basin)
    fig.supxlabel('River discharge [m$^3$ s$^{-1}$]')
    for i, impact_parameter in enumerate(impact_parameters):
        ax = axs[i]
        ax.plot(ds_glofas_reanalysis[settings.STATIONS_BY_BASIN[basin][0]], df[impact_parameter],'.', label=impact_parameter, c=f'C{i}')
        ax.legend()

```

```python
# Define GloFAS events
rp = settings.MAIN_RP
df_station_stats = pd.DataFrame(columns=['station', 'impact_parameter', 'TP', 'FP', 'FN'])

for basin, station_list in settings.STATIONS_BY_BASIN.items():
    for station in station_list:
        rp_val = df_return_period.loc[rp, station]
        glofas_dates = utils.get_dates_list_from_dataset(ds_glofas_reanalysis[station], rp_val, min_duration=settings.DURATION)
        for impact_parameter in impact_parameters:
            df_events_sub = df_events_high_impact[
                (df_events_high_impact['basin'] == basin) & (df_events_high_impact['impact_parameter'] == impact_parameter)
            ]
            detection_stats = utils.get_detection_stats(true_event_dates=df_events_sub['date'].values,
                                                       forecasted_event_dates=glofas_dates,
                                                       days_before_buffer=settings.DAYS_BEFORE_BUFFER,
                                                       days_after_buffer=settings.DAYS_AFTER_BUFFER)
   
    
            df_station_stats = df_station_stats.append({
                **{'station': station,
                'impact_parameter': impact_parameter},
                **detection_stats
            }, ignore_index=True)
df_station_stats = utils.get_more_detection_stats(df_station_stats)
df_station_stats
```

### Plot events and GloFAS together

```python
basin = 'Koshi'
impact_parameter = 'Total Death'
df_events_sub = df_events_high_impact[
                (df_events_high_impact['basin'] == basin) & (df_events_high_impact['impact_parameter'] == impact_parameter)
            ]
df_events_sub
```

```python
rp = settings.MAIN_RP
for basin, station_list in settings.STATIONS_BY_BASIN.items():
    for station in station_list:
        fig, axs = plt.subplots(len(impact_parameters), figsize=(12, 10))
        fig.suptitle(f'{basin} - {station}')
        fig.supylabel('Discharge [m$^3$ s$^{-1}$]')
        for i, impact_parameter in enumerate(impact_parameters):
            df_events_sub = df_events_high_impact[
                (df_events_high_impact['basin'] == basin) & (df_events_high_impact['impact_parameter'] == impact_parameter)
            ]
            observations = ds_glofas_reanalysis[station].values
            x = ds_glofas_reanalysis.time
            ax = axs[i]
            ax.plot(x, observations, lw=0.5)
            # Plot when GLoFAS is above RP
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val, settings.DURATION)
            for group in groups:
                idx = range(group[0], group[1])
                ax.plot(x[idx], observations[idx], ls='-', 
                        lw=2, c='C1')
            ax.axhline(y=rp_val, c='C1')
            # Plot events
            for _, row in df_events_sub.iterrows():
                ax.axvline(x=row['date'], c='r')
            ax.set_title(impact_parameter)
        axs[0].plot([], [], 'C1', label=f'1 in {rp} y')
        axs[0].plot([], [], 'C3', label='event')
        axs[0].legend()
                
```

### Get GloFAS event list for Ragindra

```python
df_glofas_events = pd.DataFrame(columns=['basin', 'station', 'rp', 'date'])
for basin, station_list in settings.STATIONS_BY_BASIN.items():
    for station in station_list:
        observations = ds_glofas_reanalysis[station]
        for rp in df_return_period.index:
            rp_val=df_return_period.loc[rp, station]
            groups = utils.get_groups_above_threshold(observations, rp_val, settings.DURATION)
            for group in groups:
                idx = group[0] + settings.DURATION - 1
                row = {
                    'date': observations.time.data[idx],
                    'basin': basin,
                    'station': station,
                    'rp': rp
                }
                df_glofas_events = df_glofas_events.append(row, ignore_index=True)
df_glofas_events = df_glofas_events.drop_duplicates(subset={'date', 'basin', 'station'}, keep='last')
df_glofas_events.to_excel(EVENTS_FOR_RAGINDRA, index=False)
```

# Appendix


### Plot all events

```python
for basin, station_list in settings.STATIONS_BY_BASIN.items():
    df_events_basin = df_events_dict[basin]
    # Only select events in the time range of GloFAS
    df_events_basin = df_events_basin.loc[df_events_basin.index > ds_glofas_reanalysis.time[0].data]
    # Only select events with deaths
    df_events_basin = df_events_basin.loc[df_events_basin['Total Death'] > 0]
    for station in station_list:
        observations = ds_glofas_reanalysis[station].values
        x = ds_glofas_reanalysis.time
        fig, ax = plt.subplots(figsize=(15,4))
        ax.plot(x, observations)
        for i, row in df_events_basin.iterrows():
            ax.axvline(x=i, c='r', alpha=0.1)
```
