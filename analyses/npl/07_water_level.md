```python
import os
from pathlib import Path


import pandas as pd
import matplotlib.pyplot as plt
```

```python
DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DHM_DIR = DATA_DIR / 'private/exploration/npl/dhm'
WL_DIR = DHM_DIR / 'water_level'
WL_FILENAME = 'GHT_{}.txt' 
STATION_INFO_FILENAME = 'npl_dhm_station_info.xlsx'

STATIONS = [
    'Chatara',
    'Chisapani'
]
```

### Check out the DHM WL data

```python
df_station_info = pd.read_excel(DHM_DIR / STATION_INFO_FILENAME, index_col='station_name')
```

```python
df_water_level_dict = {}
for station in STATIONS:
    station_number = df_station_info.at[station, 'station_number']
    df_wl = pd.read_csv(WL_DIR / WL_FILENAME.format(int(station_number)),
                       skiprows=1, 
                        header=None,
                        comment=' ',
                        parse_dates=[0],
                        names=['date', 'time', 'water_level']
                       ).groupby('date').mean()
    df_water_level_dict[station] = df_wl
```

```python
for station, df_wl in df_water_level_dict.items():
    warning_level = df_station_info.at[station, 'warning_level']
    danger_level = df_station_info.at[station, 'danger_level']
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df_wl.index, df_wl['water_level'], '.')
    ax.axhline(y=warning_level, c='C1')
    ax.axhline(y=danger_level, c='C3')
    ax.set_title(station)
    ax.set_ylabel('Water level [m]')
```

```python
# For Chatara, remove 1980 and 1981 since there is a big gap
# For Chisapani, remove all data before 1985 becuase something weird changed with the WL
```

```python
df_wl = df_water_level_dict['Chisapani']
df_wl.groupby(df_wl.index.year).count()
```

```python

```
