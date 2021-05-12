---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: Python [conda env:anact] *
    language: python
    name: conda-env-anact-py
---

```python
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.interpolate import interp1d
import os
from pathlib import Path
import sys
import seaborn as sns

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config

config = Config()
mpl.rcParams['figure.dpi'] = 300

GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']
ADM2_SEL = ['Chikwawa', 'Nsanje']
```

### Read in GloFAS data

```python
da_glofas_reanalysis = {}
da_glofas_reforecast = {}
da_glofas_forecast = {}
da_glofas_forecast_summary = {}
da_glofas_reforecast_summary = {}

for station in STATIONS: 
    da_glofas_reanalysis[station] = rd.get_glofas_reanalysis(version=GLOFAS_VERSION, station=station)
    da_glofas_reforecast[station] = rd.get_glofas_reforecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast[station] = rd.get_glofas_forecast(version=GLOFAS_VERSION, station=station)
    da_glofas_forecast_summary[station] = rd.get_da_glofas_summary(da_glofas_forecast[station])
    da_glofas_reforecast_summary[station] = rd.get_da_glofas_summary(da_glofas_reforecast[station])
```

### Explore Floodscan data

```python
df_floodscan = rd.get_floodscan_processed()
```

```python
for district in ADM2_SEL:
    fig, ax = plt.subplots()
    sns.lineplot(data=df_floodscan[df_floodscan['ADM2_EN']==district], x="date", y="mean_cell", lw=0.25)
    ax.set_ylabel('Mean flooded fraction')
    ax.set_xlabel('Date')
    ax.set_title(f'Flooding in {district}, 1998-2020')
```

### Understand relationship between GloFAS (streamflow) and Floodscan (% flooding in adm2)

```python
da_glofas_reanalysis['glofas_1'].to_dataframe().reset_index()
```

```python
df_floodscan[df_floodscan['ADM2_EN']=='Nsanje']
```
