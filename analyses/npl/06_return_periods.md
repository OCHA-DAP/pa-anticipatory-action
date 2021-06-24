```python
from pathlib import Path
import os
import sys
from importlib import reload

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev
import xarray as xr
import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
reload(utils)
```

```python
COUNTRY_ISO3 = 'npl'

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
GLOFAS_DIR = DATA_DIR / "public/exploration/npl/glofas"
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
```

```python
years = np.arange(1.5, 30.5, 0.5)
df_rps_empirical = utils.get_return_periods(ds_glofas_reanalysis, years=years, method="empirical")
df_rps_analytical = utils.get_return_periods(ds_glofas_reanalysis, years=years, method="analytical", show_plots=True)
```

```python
glofas_rp = pd.read_excel(GLOFAS_RP_FILENAME, index_col='rp')


for i, station in enumerate(stations):
    fig, ax = plt.subplots()
    ax.plot(df_rps_empirical.index, df_rps_empirical[station], label='empirical')
    ax.plot(df_rps_empirical.index, df_rps_analytical[station], label='analytical')
    if station in glofas_rp.columns:
        ax.plot(glofas_rp[station].index, glofas_rp[station], 'o', label='GloFAS')
    ax.set_title(station)
    ax.legend()
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
```
