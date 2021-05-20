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

GLOFAS_STATION_FILE = DATA_DIR / 'glb/glofas/station_list.pdf'
GLOFAS_STATION_OUTPUT = DATA_DIR / 'npl/glofas/npl_glofas_stations.gpkg'

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

## Read in Nepal GloFAS stations

```python
tables = tabula.read_pdf(GLOFAS_STATION_FILE, pages=[51,52], 
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
df_glofas.to_file(GLOFAS_STATION_OUTPUT, driver="GPKG", index=False)

```

## Scrape stations from government website

```python
SCRAPE_GOV_STATIONS = True
if SCRAPE_GOV_STATIONS:
    # Get the stations
    df_gov = gpd.GeoDataFrame(columns=['web_id', 'station_index', 'name'])
    for nstation in range(5000):
        browser.get(f"{BASE_URL}/{nstation}")
        name = browser.find_elements_by_class_name("stationTitle")
        if len(name) == 0:
            continue
        table = browser.find_elements_by_class_name("table")[0].text.split('\n')
        station_info = {
            'web_id': nstation,
            'station_index': table[0].split(' ')[-1],
            'name': name[0].text,
            'lat': table[1].split(' ')[-1],
            'lon': table[2].split(' ')[-1],
        }
        print(station_info)
        df_gov = df_gov.append(station_info, ignore_index=True)
    df_gov['geometry'] = gpd.points_from_xy(df_gov['lon'], df_gov['lat'])
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
gpd.points_from_xy(1, 2)
```

```python
table
```

### Connect government stations with basin

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

# Save the file again
df_gov.to_file(GOV_STATION_FILENAME, driver="GPKG", index=False)


```

### Count stations requested by partners

Ragindra suggested using all stations in the Koshi basin south of Chautara, and all from Karnali.

```python
# Get all Koshi stations
df_koshi = df_gov[df_gov['basin'] == 'Koshi']
n_koshi = len(df_koshi)
# Get all south of Chautara
max_lat = df_gov[df_gov['name'] == "Chautara"]['geometry'][0].y
df_koshi = df_koshi.cx[:, :max_lat]
n_koshi_chautara = len(df_koshi)

# Get all Karnali stations
df_karnali = df_gov[df_gov['basin'] == 'Karnali']
n_karnali = len(df_karnali)

print(f"Number of stations:\n{n_koshi} Koshi\n{n_koshi_chautara} Koshi south of Chautara\n{n_karnali} Karnali")
```

## Connect Gove with GloFAS stations

```python
from shapely.ops import nearest_points

#def near(point):
#    # find the nearest point and return the corresponding Place value
#    nearest = df_gov.geometry == nearest_points(
#        point, df_gov.geometry.unary_union)[1]
#    return df_gov[nearest].name
#    
#df_glofas['nearest'] = df_glofas.apply(lambda row: near(row.geometry), axis=1)

for i, row in df_glofas.iterrows():
    print('Glofas station:')
    print(row)
    idx = df_gov.geometry == nearest_points(row.geometry, df_gov.geometry.unary_union)[1]
    print('Nearest gov station:')
    print(df_gov[idx])
    print('\n')


```

# Appendix


### Examine shapefile from gov website

```python
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
gpd.read_file(GOV_BASIN_SHAPEFILE, driver='kml')
```

```python
df_glofas
```

```python
df_gov[df_gov['name'] == 'Chisapani(Karnali)']
```

```python

```
