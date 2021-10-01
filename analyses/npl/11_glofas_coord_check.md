Check that the positions of the GloFAS reporting points make sense on the raster

```python
import glob

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import npl_parameters as parameters
from src.utils_general.utils import parse_yaml
```

```python

GLOFAS_NEW_STATIONS = parameters.DATA_DIR_PRIVATE / 'exploration/glb/glofas/Qgis_World_outlet_202104_20210421.csv'
NEPAL_RAW_DATA_DIR = parameters.DATA_DIR_PUBLIC / 'raw/npl/glofas/version_3/cems-glofas-historical/'
STATIONS_YML = '../../src/npl/config.yml'

STATIONS = [
    'Chatara', 'Simle', 'Majhitar', 'Kampughat', 'Rai_goan',
    'Chisapani', 'Asaraghat', 'Dipayal', 'Samajhighat', 'Kusum', 'Chepang',
]
```

```python
df_stations = pd.DataFrame(index=STATIONS, columns=['v2', 'v3', 'ervin'])
df_stations_ervin = pd.read_csv(GLOFAS_NEW_STATIONS)
stations_dict = parse_yaml(STATIONS_YML)['glofas']['stations']
```

```python
for station in STATIONS:
    # Get v2 and v3 stations
    df_stations.loc[station, 'v2'] = (stations_dict[station]['lon'], stations_dict[station]['lat'])
    df_stations.loc[station, 'v3'] = (stations_dict[station + '_v3']['lon'], stations_dict[station + '_v3']['lat'])
    # Get lon lat from Ervin's list
    station_ervin = station
    if station_ervin == 'Rai_goan':
        station_ervin = 'Rai Goan'
    row_ervin = df_stations_ervin[df_stations_ervin['StationName'] == station_ervin].iloc[-1]
    df_stations.loc[station, 'ervin'] = (row_ervin['LisfloodX'], row_ervin['LisfloodY'])
```

## Read in raw GloFAS data

```python
# This takes a minute
da = xr.open_mfdataset([str(filename) for filename in NEPAL_RAW_DATA_DIR.glob('*.grib')],
                                        engine='cfgrib')['dis24']
```

```python
# As a start, take the average as a proxy for local 'best'
rd = np.log10(da.mean(axis=0))
z = rd.values
```

```python
res = 0.05
extent=[da.longitude[0]-res, da.longitude[-1]+res, da.latitude[-1]-res, da.latitude[0]+res]

fig, ax = plt.subplots(figsize=(25, 10))
im = ax.imshow(z, extent=extent, cmap='Greys_r', vmin=0)
cb = plt.colorbar(im)
cb.set_label('Log mean river discharge from 1979 to present')
ax.grid()

for station, row in df_stations.iterrows():
    x, y = row['v3']
    ax.plot(x, y, 'o', label=station)
ax.legend()
ax.set_xlabel('lon')
ax.set_ylabel('lat')

```
