### Cleaning historical flood event data

This notebook cleans an input data of historical flood events in Malawi. Cleaned data is output to the 'exploration' data directory. 

```python
import datetime
import pandas as pd
import os
from pathlib import Path
import sys
from datetime import timedelta

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
```

Read the input datasets to be cleaned.

```python
df_rco = pd.read_excel(PRIVATE_DIR / 'raw' / 'mwi' / 'DISASTER PROFILE-RCO.xlsx', header=1)
```

Basic cleaning of some columns.

```python
df_rco = df_rco.rename(columns=lambda x: x.strip())
df_rco = df_rco[df_rco['TYPE OF DISASTER'].notna()]
df_rco['TYPE OF DISASTER'] = df_rco['TYPE OF DISASTER'].str.lower()
```

Filter the df to get the events that we're interested in. This processing assumes that only events with data in the ```Remark``` or ```EXTENT OF DAMAGE``` column are impactful.

```python
mask_flooding = df_rco['TYPE OF DISASTER'].str.contains('flood')
mask_district = df_rco['DISTRICT'].isin(['Nsanje', 'Chikwawa'])
mask_impact = pd.notnull(df_rco['Remark']) | pd.notnull(df_rco['EXTENT OF DAMAGE'])

df_rco_sel = df_rco[mask_flooding & mask_district & mask_impact]
```

Clean the dates and add in an ```end_date``` column. Some dates are in a ```datetime``` format without an end date, while others are in a string format such as ```24-25/12/1995```.

```python
BUFFER = 60
df_rco_sel['end_date'] = ''
df_rco_sel['start_date'] = ''
r = re.compile('.*-.*/.*/.*')

for index,row in df_rco_sel.iterrows():
    date = df_rco_sel['Date of Reported Occurrence'][index]
    
    if isinstance(date, datetime.date):
        df_rco_sel.loc[index, 'start_date'] = pd.to_datetime(date)
        df_rco_sel.loc[index, 'end_date'] = pd.to_datetime(date) + timedelta(days=60)
    
    elif isinstance(date, str): 
        if r.match(date):
            end_date = pd.to_datetime(date[9:13] + '-'+ date[6:8] +'-'+ date[:2])
            start_date = pd.to_datetime(date[9:13] + '-'+ date[6:8] +'-'+ date[3:5])
            df_rco_sel.loc[index, 'end_date'] = end_date
            df_rco_sel.loc[index, 'start_date'] = start_date
        else:
            try:
                df_rco_sel.loc[index, 'start_date'] = pd.to_datetime(date)
                df_rco_sel.loc[index, 'end_date'] = pd.to_datetime(date) + timedelta(days=BUFFER)
            except: 
                print(f'Could not parse to a date: {date}')
                df_rco_sel.drop(index, inplace=True)
    else: 
        print(f'Not a string or datetime.date: {date}')
```

Aggregate events to the district level and separate by district. We also need to merge together events that are within 2 months of each other. This aggregation and merging introduces potential inaccuracies as we are likely merging together some events within the same district but in separate TAs. 

```python
df_rco_grouped = (
    df_rco_sel
    .groupby(['start_date', 'end_date', 'DISTRICT'], as_index=False)
    .count()[['start_date', 'end_date', 'DISTRICT']]
)

for station in stations_adm2.values():
    df_district = df_rco_grouped[df_rco_grouped['DISTRICT'] == station].reset_index()
    df_district['flood_id'] = 0
    f_id = 1
    
    # Loop through all of the events and tag the ones that are part of an overlap
    for i in range(1, len(df_district.index)):        
        start = df_district['start_date'].iloc[i,]
        end = df_district['end_date'].iloc[i-1,]
        if start < end:
            df_district.loc[i, 'flood_id'] = f_id
            df_district.loc[i-1, 'flood_id'] = f_id
        else:           
            df_district.loc[i-1, 'flood_id'] = f_id
            f_id += 1
    
    # Now for each event, extract the min start data and max end date
    df_start = df_district.groupby('flood_id')['start_date'].min().to_frame().reset_index()
    df_end = df_district.groupby('flood_id')['end_date'].max().to_frame().reset_index()
    
    df_events = df_start.merge(df_end, on='flood_id')
    df_events.to_csv((EXPLORE_DIR / f'{station}_rco_event_summary.csv'), index=False)
```
