```python
from pathlib import Path
import os
import sys
from importlib import reload

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import utils
reload(utils)
```

```python
COUNTRY_ISO3 = 'npl'
LEADTIMES = [x+1 for x in range(7)]
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=LEADTIMES, interp=False
)
ds_glofas_forecast_summary = utils.get_glofas_forecast_summary(ds_glofas_reforecast)
```

```python
ds_glofas_reanalysis
```

```python
years = np.arange(1.5, 30.5, 0.5)
df_rps_empirical = utils.get_return_periods(ds_glofas_reanalysis, years=years, method="empirical")
```
