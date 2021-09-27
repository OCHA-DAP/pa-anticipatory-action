Explore the DHM water level data, and create a summary CSV file

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import npl_parameters as parameters
from src.utils_general import statistics
```

```python
STATIONS = [
    'Chatara',
    'Chisapani',
    'Asaraghat',
    'Chepang',
    'Kusum'
]
```

### Check out the DHM WL data

```python
df_station_info = pd.read_excel(parameters.DHM_STATION_INFO_FILENAME, index_col='station_name', dtype={'station_number': object})
```

```python
for i, station in enumerate(STATIONS):
    station_number = df_station_info.at[station, 'station_number']
    df_wl_station = pd.read_csv(parameters.WL_RAW_DIR / parameters.WL_INPUT_FILENAME.format(station_number),
                       skiprows=1, 
                        header=None,
                        comment=' ',
                        parse_dates=[0],
                        names=['date', 'time', station]
                       ).groupby('date').mean()
    if i == 0:
        df_wl = df_wl_station.copy()
    else:
        df_wl = pd.merge(df_wl, df_wl_station, how='outer', left_index=True, right_index=True)

```

```python
def plot_wl(df_wl, df_station_info, station):
    warning_level = df_station_info.at[station, 'warning_level']
    danger_level = df_station_info.at[station, 'danger_level']
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df_wl.index, df_wl[station], '.')
    ax.axhline(y=warning_level, c='C1')
    ax.axhline(y=danger_level, c='C3')
    ax.set_title(station)
    ax.set_ylabel('Water level [m]')

for station in STATIONS:
    plot_wl(df_wl, df_station_info, station)
```

```python
# For Chatara, remove 1980 and 1981 since there is a large gap
df_wl.loc[df_wl.index.year.isin([1980, 1981]), 'Chatara'] = np.nan

# For Chisapani, remove all data before 1985 becuase something weird changed with the WL,
# and remove 1989 which also has gaps
df_wl.loc[df_wl.index.year < 1985, 'Chisapani'] = np.nan
df_wl.loc[df_wl.index.year == 1989, 'Chisapani'] = np.nan

# For Asaraghat, remove data above year 2011
df_wl.loc[df_wl.index.year > 2011, 'Asaraghat'] = np.nan

```

```python
# Plot again to take a lok at changes
for station in STATIONS:
    plot_wl(df_wl, df_station_info, station)
```

```python
# Write out the station files:, 'Chepang']:, 'Chepang']
df_wl.to_csv(WL_PROCESSED_DIR / WL_OUTPUT_FILENAME)
```

### Get return periods

```python
major_stations = ['Chatara', 'Chisapani']
rps = np.linspace(1.5, 25, 1000)

for station in major_stations:
    # Get the max value per year
    df = (df_wl[[station]]
          .dropna()
          .resample(rule='A', kind='period')
          .max()
          .dropna()
         )
    rp_analytical = statistics.get_return_period_function_analytical(df, station, show_plots=True, plot_title=station)
    rp_empirical = statistics.get_return_period_function_empirical(df, station)    
    
    # Get the RP of the warning and danger levels
    warning_level = df_station_info.at[station, 'warning_level']
    danger_level = df_station_info.at[station, 'danger_level']
    print(f"RP warning: {np.round(rps[np.argmin(np.abs(rp_analytical(rps) - warning_level))], 1)}")
    print(f"RP danger: {np.round(rps[np.argmin(np.abs(rp_analytical(rps) - danger_level))], 1)}")

    fig, ax = plt.subplots()
    ax.plot(rps, rp_analytical(rps), label='Analytical')
    ax.plot(rps, rp_empirical(rps), label='Emprical')
    ax.set_title(station)
    ax.axhline(warning_level, c='C2', label='Warning level')
    ax.axhline(danger_level, c='C3', label='Danger level')
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('Water level [m]')
    ax.legend()
    ax.grid()
```
