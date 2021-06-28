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

SAVE_PLOT = True
PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
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
    df_sel.to_csv((EXPLORE_DIR / f'{station}_emdat_event_summary.csv'), index=False)
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
    dict_combined[station].to_csv(EXPLORE_DIR / f'{station}_all_events.csv', index=False)
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
    dict_event_sum[station].to_csv(EXPLORE_DIR / f'{station}_combined_event_summary.csv')
```

### Archive (this isn't really working...)

Now we want to check through all the events and find those that are confirmed by at least two sources. We'll check for overlap with start and end dates between events, and add in a 30-day buffer on either side of these dates to extend the potential matching window.

```python
# # Input two dfs of flood events with start and end date columns
# # Will output a list of event start/end date pairs that have overlap between the two dfs
# # We pick the min start and max end date in the matching events
# # A buffer can also be specified to extend the potential matching window
# def check_event_match(df1, df2, buffer=0):
    
#     events_matched = []
    
#     df1_copy = df1.copy()
    
#     df1_copy['start_date_buffer'] = pd.to_datetime(df1["start_date"]) - timedelta(days=buffer)
#     df1_copy['end_date_buffer'] = pd.to_datetime(df1["end_date"]) + timedelta(days=buffer)
    
#     for df1_index, df1_row in df1_copy.iterrows():

#         df1_dates = np.array(pd.date_range(df1_row['start_date_buffer'], df1_row['end_date_buffer']))
#         df1_dates_original = np.array(pd.date_range(df1_row['start_date'], df1_row['end_date']))

#         for df2_index, df2_row in df2.iterrows():
#             df2_dates = np.array(pd.date_range(df2_row['start_date'], df2_row['end_date']))

#             if (set(df1_dates) & set(df2_dates)):
                
#                 match_start = min([df1_dates_original[0], df2_dates[0]])
#                 match_end = max([df1_dates_original[-1], df2_dates[-1]])
                
#                 events_matched.append([match_start, match_end])
                     
#     return events_matched

# def merge_events(df):
#     df['flood_id'] = 0
#     f_id = 1
    
#     # Loop through all of the events and tag the ones that are part of an overlap
#     for i in range(1, len(df.index)):        
#         start = df['start_date'].iloc[i,]
#         end = df['end_date'].iloc[i-1,]
#         if start < end:
#             df.loc[i, 'flood_id'] = f_id
#             df.loc[i-1, 'flood_id'] = f_id
#         else:           
#             df.loc[i-1, 'flood_id'] = f_id
#             f_id += 1
    
#     # Now for each event, extract the min start data and max end date
#     df_start = df.groupby('flood_id')['start_date'].min().to_frame().reset_index()
#     df_end = df.groupby('flood_id')['end_date'].max().to_frame().reset_index()
    
#     df_events = df_start.merge(df_end, on='flood_id').sort_values(by='start_date')
#     return df_events
```

```python
# # This is all a bit hard-coded but hopefully good enough for now...
# # This won't really scale up to more input sources
# buffer = 0
# dict_matched = {}

# for station in stations_adm2.values():

#     match_floodscan_emdat = check_event_match(dict_floodscan[station], dict_emdat[station], buffer)
#     match_floodscan_rco = check_event_match(dict_floodscan[station], dict_rco[station], buffer)
#     match_rco_emdat = check_event_match(dict_rco[station], dict_emdat[station], buffer)
#     match_cerf_emdat = check_event_match(dict_cerf[station], dict_emdat[station], buffer)
#     match_cerf_floodscan = check_event_match(dict_cerf[station], dict_floodscan[station], buffer)
#     match_cerf_rco = check_event_match(dict_cerf[station], dict_rco[station], buffer)
    
#     df_matched = pd.DataFrame(columns=['start_date', 'end_date', 'sources'])

#     for event in match_floodscan_emdat:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'floodscan_emdat'}, ignore_index=True)

#     for event in match_floodscan_rco:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'floodscan_rco'}, ignore_index=True)

#     for event in match_rco_emdat:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'rco_emdat'}, ignore_index=True)
    
#     for event in match_cerf_emdat:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'cerf_emdat'}, ignore_index=True)
    
#     for event in match_cerf_floodscan:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'cerf_floodscan'}, ignore_index=True)
    
#     for event in match_cerf_rco:
#         df_matched = df_matched.append({'start_date': event[0], 'end_date': event[1], 'sources': 'cerf_rco'}, ignore_index=True)
    
#     # Flag duplicates 
#     # This would happen where there is a 3-way agreement between events
#     df_matched['duplicated'] = df_matched.duplicated(subset=['start_date', 'end_date'], keep=False)
#     df_matched['sources'] = np.where(df_matched['duplicated']==True, 'floodscan_emdat_rco', df_matched['sources'])
    
#     # Now drop the duplicates
#     df_matched = df_matched.drop_duplicates(subset=['start_date', 'end_date']).drop(columns=['duplicated'])
    
#     dict_matched[station] = df_matched.sort_values(by='start_date')
```
