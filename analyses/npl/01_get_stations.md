```python
import os
from pathlib import Path
import requests

import tabula
import fiona
import pandas as pd
import geopandas as gpd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options


DATA_DIR = Path(os.environ["AA_DATA_DIR"]) / 'public/exploration'

STATION_FILE = DATA_DIR / 'glb/glofas/station_list.pdf'

# Be careful if you want to run this, it takes awhile
# and they may blacklist your IP!
SCRAPE_GOV_STATIONS = False
BASE_URL = f"http://www.hydrology.gov.np/#/basin"
GOV_DIR =  DATA_DIR / 'npl/gov'
GOV_STATION_FILENAME = GOV_DIR / 'npl_gov_station_info.gpkg'
GOV_BASIN_FILENAME = GOV_DIR / 'npl_gov_basins.csv'
GOV_BASIN_SHAPEFILE = GOV_DIR / 'Nepal_Basin_final.kml'

    
opts = Options()
opts.set_headless()
browser = webdriver.Firefox(options=opts)

```

### Read in Nepal GloFAS stations

```python
tables = tabula.read_pdf(STATION_FILE, pages=[51,52], 
                         pandas_options=dict(
                             header=None))

```

```python
cnames = [ 
          'station_name', 
          'river_name', 
          'river_basin', 
          'lat', 
          'lon']

cnums_lat_lon = {
    0: [12, 14],
    1: [13, 15]
}

table_list = []
for i in range(2):
    table = tables[i]
    # Select only nepal
    table = table[table[0] == 'Nepal Asia']
    # Select only neeeded columns
    table = table[[1, 2, 3] + cnums_lat_lon[i]]
    table.columns = cnames
    table = gpd.GeoDataFrame(
        table, 
        geometry=gpd.points_from_xy(
            table['lon'], table['lat']))
    table_list.append(table)
    
df_glofas = pd.concat(table_list, ignore_index=True)
```

### Scrape stations from government website

```python
if SCRAPE_GOV_STATIONS:
    # Get the stations
    df_gov = gpd.GeoDataFrame(columns=['id', 'name', 'lat', 'lon'])
    for nstation in range(5000):
        browser.get(f"{BASE_URL}/{nstation}?_k=uij9mp")
        name = browser.find_elements_by_class_name("stationTitle")
        if len(name) == 0:
            continue
        table = browser.find_elements_by_class_name("table")[0].text.split('\n')
        station_info = {
            'id': nstation,
            'name': name[0].text,
            'geometry': gpd.points_from_xy(table[2].split(' ')[-1],
                                           table[1].split(' ')[-1])
        }
        df_gov = df_gov.append(station_info, ignore_index=True)
    df_gov.to_file(GOV_STATION_FILENAME, driver="GPKG", index=False)
   
    # Get the basin - station relations
    browser.get(BASE_URL)
    e = browser.find_elements_by_class_name("FolderHierarchy")
    basin_station_list = e[0].text.split('\n')
    df_basin_station = pd.DataFrame({'name': basin_station_list})
    df_basin_station.to_csv(GOV_BASIN_FILENAME, index=False)
    
else:
    df_gov = gpd.read_file(GOV_STATION_FILENAME)
    basin_station_list = pd.read_csv(GOV_BASIN_FILENAME, header=None)[0].to_list()
```

```python
# basin_station_list contains the name of each basin followed by all the stations
# within it, these need to be disentangled

basin_list = [
    'Mahakali',
    'Karnali',
    'Babai',
    'West Rapti',
    'Narayani',
    'Bagmati',
    'Kamala',
    'Koshi', 
    'Kankai', 
    'Biring'
]

basin_station_dict = {}
active_basin = ''
for station in basin_station_list:
    if station in basin_list:
        basin_list.remove(station)
        active_basin = station
        basin_station_dict[active_basin] = []
    else:
        basin_station_dict[active_basin].append(station)
        

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        for item in d[key]:
            inverse[item] = key
    return inverse

station_basin_dict = invert_dict(basin_station_dict)

# Add the basins to the stations
# (there are still some stations with no associated basin)
df_gov['basin'] = df_gov['name'].map(station_basin_dict)

```

```python
# Want Koshi basin, south of Chatara
max_lat = df_gov[df_gov['name'] == "Chautara"]['geometry'][0].y


df_koshi = df_gov[df_gov['basin'] == 'Koshi']

df_kkoshi = df_koshi.cx[:, :max_lat]

df_karnali = df_gov[df_gov['basin'] == 'Karnali']
len(df_karnali)
```

71 stations in Koshi basin
57 south of Chatara
47 stations in Karnali


# Appendix


### Examine shapefile from gov website

```python
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
gpd.read_file(GOV_BASIN_SHAPEFILE, driver='kml')
```

```python

```
