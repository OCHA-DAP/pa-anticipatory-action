We're finding a mismatch between our calculated return periods on those on the GloFAS web interface.
Plot them against each other to compare. (Mismatch was due to incorrect coordinate multiples
in the API calls, is now fixed)

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev
import xarray as xr
import pandas as pd

import npl_parameters as parameters
from src.indicators.flooding.glofas import utils
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=parameters.COUNTRY_ISO3)
```

```python
years = np.arange(1.5, 30.5, 0.5)
df_rps_empirical = utils.get_return_periods(ds_glofas_reanalysis, years=years, method="empirical")
df_rps_analytical = utils.get_return_periods(ds_glofas_reanalysis, years=years, method="analytical", show_plots=True)
```

```python
glofas_rp = pd.read_excel(parameters.GLOFAS_RP_FILENAME, index_col='rp')


for i, station in enumerate(ds_glofas_reanalysis.keys()):
    fig, ax = plt.subplots()
    ax.plot(df_rps_empirical.index, df_rps_empirical[station], label='empirical')
    ax.plot(df_rps_empirical.index, df_rps_analytical[station], label='analytical')
    station_name = station.split('_v3')[0]  
    if station_name in glofas_rp.columns:
        ax.plot(glofas_rp[station_name].index, glofas_rp[station_name], 'o', label='GloFAS')
    ax.set_title(station)
    ax.legend()
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
```
