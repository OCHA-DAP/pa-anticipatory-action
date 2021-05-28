```python
import pandas as pd
import os
from pathlib import Path
import sys
import numpy as np
from datetime import timedelta

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
config = Config()

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'

ADM2 = ['Nsanje', 'Chikwawa']
```

```python
floodscan_events = {}

for adm in ADM2: 
    floodscan_events[adm] = pd.read_csv(EXPLORE_DIR / f'{adm}_floodscan_event_summary.csv')

df_emdat = pd.read_csv(EXPLORE_DIR / 'emdat.csv')
```

```python
df_emdat = df_emdat[(df_emdat['Disaster Subtype']=='Riverine flood') | (df_emdat['Disaster Subtype'].isnull())]

df_emdat['Start Day'] = df_emdat['Start Day'].fillna(1).astype(int)
df_emdat['End Day'] = df_emdat['End Day'].fillna(1).astype(int)

df_emdat['start_date'] = pd.to_datetime(dict(year=df_emdat['Start Year'], month=df_emdat['Start Month'], day=df_emdat['Start Day']))
df_emdat['end_date'] = pd.to_datetime(dict(year=df_emdat['End Year'], month=df_emdat['End Month'], day=df_emdat['End Day']))
```

```python
BUFFER = 30

for adm in ADM2:

    df_emdat_sel = df_emdat[df_emdat['Geo Locations'].str.contains(adm)]
    print(f'{len(df_emdat_sel.index)} events in {adm}')
    print(df_emdat_sel)
    df_floodscan = floodscan_events[adm]
    df_floodscan['emdat_match'] = 0
    df_floodscan['start_date_buffer'] = pd.to_datetime(df_floodscan["start_date"]) - timedelta(days=BUFFER)
    df_floodscan['end_date_buffer'] = pd.to_datetime(df_floodscan["end_date"]) + timedelta(days=BUFFER)

    for fs_index, fs_row in df_floodscan.iterrows():

        fs_dates = np.array(pd.date_range(fs_row['start_date_buffer'], fs_row['end_date_buffer']))

        for em_index, em_row in df_emdat_sel.iterrows():
            em_dates = np.array(pd.date_range(em_row['start_date'], em_row['end_date']))

            if (set(fs_dates) & set(em_dates)):
                df_floodscan.loc[fs_index, 'emdat_match'] =+1
    
    floodscan_events[adm] = df_floodscan
```

```python
floodscan_events['Nsanje']
```

```python
floodscan_events['Chikwawa']
```
