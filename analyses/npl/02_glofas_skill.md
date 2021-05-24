```python
from pathlib import Path
import os
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib as mpl

#path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
#os.chdir(path_mod)

from src.indicators.flooding.glofas import utils, glofas
import src.nepal.get_glofas_data as ggd

reload(utils)

mpl.rcParams['figure.dpi'] = 200


COUNTRY_ISO3 = 'npl'
STATIONS = {
    'Koshi': ['Chatara', 'Simle', 'Majhitar'],
    'Karnali': ['Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat'],
    'Rapti': ['Kusum'],
    'Bagmati': ['Rai_goan'],
    'Babai': ['Chepang']
}
```

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
ds_glofas_reforecast = utils.get_glofas_reforecast(
    country_iso3 = COUNTRY_ISO3, leadtimes=ggd.LEADTIMES,
    interp=False
)
```

```python
df_return_period = utils.get_return_periods(ds_glofas_reanalysis)


```

```python
rp_label = [str(int(x)) for x in df_return_period.index]
rp_label[0] = '1.5'
for basin, stations in STATIONS.items():
    fig, ax = plt.subplots()
    ax.set_title(basin)
    for station in stations:
        rp = df_return_period[station]
        ax.plot(rp_label, rp, 'o-', label=station)
    ax.set_xlabel('Return period [years]')
    ax.set_ylabel('River discharge [m$^3$ s$^{-1}$]')
    ax.legend()
```

```python
def plot_crps(df_crps, title_suffix=None):
    for basin, stations in STATIONS.items():
        fig, ax = plt.subplots()
        for station in stations:
            crps = df_crps[station]
            ax.plot(crps.index, crps, label=station)
        ax.legend()
        title = basin
        if title_suffix is not None:
            title += title_suffix
        ax.set_title(title)
        ax.set_xlabel("Lead time [days]")
        ax.set_ylabel("Normalized CRPS [% error]")
```

```python
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                        normalization="mean")
plot_crps(df_crps * 100, title_suffix=" -- all discharge values")
```

```python
rp = 1.5
df_crps = utils.get_crps(ds_glofas_reanalysis, 
                         ds_glofas_reforecast,
                         normalization="mean", 
                         thresh=df_return_period.loc[rp].to_dict())
plot_crps(df_crps * 100, title_suffix=f" -- values > RP 1 in {rp} y")
```

```python

```

```python
test = df_crps['Chatara']
test.index
```
