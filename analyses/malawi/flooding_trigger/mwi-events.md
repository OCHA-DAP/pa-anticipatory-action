---
jupyter:
  jupytext:
    formats: ipynb,md
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
import geopandas as gpd
from scipy.stats import norm, pearsonr
from scipy import stats
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.interpolate import interp1d
import os
from pathlib import Path
import sys
import seaborn as sns
from functools import reduce
from datetime import timedelta

import read_in_data as rd
from importlib import reload
reload(rd)

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.indicators.flooding.glofas import utils

config = Config()
mpl.rcParams['figure.dpi'] = 300

PLOT_DIR = config.DATA_DIR / 'processed' / 'mwi' / 'plots' / 'flooding'
EXPLORE_DIR = config.DATA_DIR / 'exploration' / 'mwi' / 'flooding'
PRIVATE_DIR = config.DATA_PRIVATE_DIR
SAVE_PLOT = False
GLOFAS_VERSION = 3
STATIONS = ['glofas_1', 'glofas_2']

stations_adm2 = {
    'glofas_1': 'Nsanje',
    'glofas_2': 'Chikwawa'
}
```

```python
df_rco = pd.read_excel(PRIVATE_DIR / 'raw' / 'mwi' / 'DISASTER PROFILE-RCO.xlsx', header=1)
df_dodma = pd.read_csv(PRIVATE_DIR / 'processed' / 'mwi' / 'mvac_dodma_flood_district.csv')
```

Clean the data

```python
df_rco = df_rco.rename(columns=lambda x: x.strip())
df_rco = df_rco[df_rco['TYPE OF DISASTER'].notna()]
df_rco['TYPE OF DISASTER'] = df_rco['TYPE OF DISASTER'].str.lower()

# TODO: Clean the date column - need to convert to all the same date-time format
# Eg. datetime.datetime(1995, 3, 1, 0, 0), '14-20/01/2003', 'January 2008' 

# TODO: Add in an 'end date' column - have to just roughly guess here, maybe like 2-3 months from the start date
```

Filter the df to get the events that we're interested in

```python
mask_flooding = df_rco['TYPE OF DISASTER'].str.contains('flood')
mask_district = df_rco['DISTRICT'].isin(['Nsanje', 'Chikwawa'])

# TODO: 
# Mask out years before 1999
# Mask out events where 'EXTENT OF DAMAGE' and 'REMARK' is empty 
# ^ this is input from Kash indicating that events without this likely weren't impactful

df_rco_sel = df_rco[mask_flooding & mask_district] # Need to add in the other masks
```

Create separate dfs per district (Chikwawa and Nsajne)

```python
# TODO
```

Output to csvs to use in ```mwi-trigger.md``` and should be good to go!

```python
# TODO
```
