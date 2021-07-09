### Combining historical flood events

This notebook combines various sources of historical flood event data: 1) Floodscan derived events, 2) events reported by the RCO, 3) DoDMA reported flooding years, and 4) past CERF allocations.

```python
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
import sys
from datetime import timedelta

mpl.rcParams['figure.dpi'] = 300

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()

SAVE_PLOT = False
SAVE_DATA = False

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
EXPLORE_DIR = PRIVATE_DIR / 'exploration' / 'mwi' / 'flooding'

stations_adm2 = {
    'G1724': 'Nsanje',
    'G2001': 'Chikwawa'
}
```

Read in the various event datasets. Both the RCO and Floodscan events are in the same format with defined start and end dates and whereby each event corresponds to a district.

```python
df_emdat = pd.read_csv(EXPLORE_DIR / 'emdat.csv')
df_dodma = pd.read_csv(PRIVATE_DIR / 'processed' / 'mwi' / 'mvac_dodma_flood_district.csv')

dict_floodscan = {}
dict_rco = {}

for station in stations_adm2.values():
    dict_rco[station] = pd.read_csv(EXPLORE_DIR / f'{station}_rco_event_summary.csv')
    dict_floodscan[station] = pd.read_csv(EXPLORE_DIR / f'{station}_floodscan_event_summary.csv')
```

The EM-DAT data still needs to be cleaned. We'll organize the dates and filter to make sure that we're getting riverine flood events within the Shire river basin.

```python
# We'll select specifically riverine floods, or events where the flood subtype isn't defined
df_emdat_sel = df_emdat[(df_emdat['Disaster Subtype']=='Riverine flood') | (df_emdat['Disaster Subtype'].isnull())]

# We'll also select events within the Shire basin, or where the basin isn't defined
# TODO: This filtering isn't working properly.. it's dropping the null ones too which we don't want...
#df_emdat_sel = df_emdat_sel[(df_emdat_sel['River Basin'].str.contains('Shire')) | (df_emdat_sel['River Basin'].isnull())]

# If no start or end day is defined we'll just set it to the first date of the month
df_emdat_sel['Start Day'] = df_emdat_sel['Start Day'].fillna(1).astype(int)
df_emdat_sel['End Day'] = df_emdat_sel['End Day'].fillna(28).astype(int)

df_emdat_sel['start_date'] = pd.to_datetime(dict(year=df_emdat_sel['Start Year'], month=df_emdat_sel['Start Month'], day=df_emdat_sel['Start Day']))
df_emdat_sel['end_date'] = pd.to_datetime(dict(year=df_emdat_sel['End Year'], month=df_emdat_sel['End Month'], day=df_emdat_sel['End Day']))

dict_emdat = {}

for station in stations_adm2.values():
    df_sel = df_emdat_sel[df_emdat_sel['Geo Locations'].str.contains(station)]
    df_sel = df_sel[['start_date', 'end_date']]
    dict_emdat[station] = df_sel
    if SAVE_DATA: df_sel.to_csv((EXPLORE_DIR / f'{station}_emdat_event_summary.csv'), index=False)
```

We'll also manually add in the 2015 CERF activation. The exact location is still unclear, but for now we'll assume that it was in both districts. The start and end dates are also somewhat unclear. I've used the date of Govt emergency declaration as the start date, and the date of the Preliminary Flood Response Plan as the end date.

```python
dict_cerf = {}
dates = [['2015-01-13', '2015-01-22']]

for station in stations_adm2.values():
    dict_cerf[station] = pd.DataFrame(dates, columns=['start_date', 'end_date'])
```

Let's tag each event by its source and combine them into a single dataframe for each station.

```python
dict_combined = {}

for station in stations_adm2.values():
    dict_floodscan[station]['source'] = 'floodscan'
    dict_rco[station]['source'] = 'rco'
    dict_emdat[station]['source'] = 'emdat'
    dict_cerf[station]['source'] = 'cerf'
    
    df1 = dict_floodscan[station].append(dict_rco[station], ignore_index=True)
    df2 = df1.append(dict_emdat[station], ignore_index=True)
    df3 = df2.append(dict_cerf[station], ignore_index=True)
    
    df3['start_date'] = pd.to_datetime(df3['start_date'])
    df3['end_date'] = pd.to_datetime(df3['end_date'])
    
    dict_combined[station] = df3[['start_date', 'end_date', 'source']].sort_values(by='start_date')
    
    # Save this out to a csv
    if SAVE_DATA: dict_combined[station].to_csv(EXPLORE_DIR / f'{station}_all_events.csv', index=False)
```

Create a new dataframe to keep track of overlapping events. 

```python
all_dates = pd.date_range('1998-01-01', '2019-12-31')
sources = dict_combined['Nsanje']['source'].unique()

buffer = 10

dict_event_daily = {}

for station in stations_adm2.values(): 
    df_event_daily = pd.DataFrame(0, index=np.arange(len(all_dates)), columns=sources)
    df_event_daily['date'] = all_dates

    for source in sources: 
        df_sel = dict_combined[station]
        df_sel = df_sel[df_sel['source']==source]

        for index, row in df_sel.iterrows():
            event_start = pd.to_datetime(df_sel.loc[index, 'start_date']) - timedelta(days=buffer)
            event_end = pd.to_datetime(df_sel.loc[index, 'end_date']) + timedelta(days=buffer)
            df_event_daily[source] = np.where((df_event_daily['date'] <= event_end) & (df_event_daily['date'] >= event_start), 1, df_event_daily[source])

    df_event_daily['source_sum'] = df_event_daily.sum(numeric_only = True, axis=1)
    
    dict_event_daily[station] = df_event_daily

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title(station)
    ax.plot(df_event_daily['date'], df_event_daily['source_sum'], '-', label=station)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of sources confirming event')
    
    if SAVE_PLOT: plt.savefig(PLOT_DIR / f'{station}_event_summary.png')
```

Extract the start and end of consecutive dates with > 1 source. There currently isn't any buffer integrated to extend the potential matching window, as this could result in overlap between two events from the same source. 

```python
dict_event_sum = {}

for station in stations_adm2.values(): 

    df_event_all = dict_event_daily[station]

    groups = np.where(np.diff(df_event_all['source_sum'] > 1, prepend=False, append=False))[
            0
        ].reshape(-1, 2)


    df_event = pd.DataFrame(groups, columns=["start_index", "end_index"])

    df_event["start_date"] = df_event["start_index"].apply(
        lambda x: df_event_all.loc[x, 'date']
    )
    df_event["end_date"] = df_event["end_index"].apply(
        lambda x: df_event_all.loc[x, 'date']
    )
    
    dict_event_sum[station] = df_event[['start_date', 'end_date']]
    if SAVE_DATA: dict_event_sum[station].to_csv(EXPLORE_DIR / f'{station}_combined_event_summary.csv')
```
