```python
import os
from pathlib import Path
import requests
import yaml

import tabula
import fiona
import pandas as pd
import geopandas as gpd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup


DATA_DIR = Path(os.environ["AA_DATA_DIR"]) / 'public/exploration'
DATA_DIR_PRIVATE = Path(os.environ["AA_DATA_DIR"]) / 'private'


GLOFAS_STATION_FILE = DATA_DIR / 'glb/glofas/station_list.pdf'
GLOFAS_STATION_OUTPUT = DATA_DIR / 'npl/glofas/npl_glofas_stations.gpkg'

# Be careful if you want to run this, it takes awhile
# and you can easily reach max requests
SCRAPE_GOV_STATIONS = False
GOV_BASE_URL = f"http://www.hydrology.gov.np/#/basin"
GOV_DIR =  DATA_DIR / 'npl/dhm'
GOV_STATION_FILENAME = GOV_DIR / 'npl_dhm_hydrology_stations.gpkg'
GOV_BASIN_FILENAME = GOV_DIR / 'npl_dhm_hydrology_basins.csv'
GOV_BASIN_SHAPEFILE = GOV_DIR / 'Nepal_Basin_final.kml'

# Another station list    
DHM_BASE_URL = "http://www.dhm.gov.np/hydrological-station/"
DHM_STATION_FILENAME = GOV_DIR / 'npl_dhm_stations.gpkg'

# Station list to share
STATION_OUTPUT = GOV_DIR / 'station_list_glofas_final.xlsx'
STATION_OUTPUT_YAML = GOV_DIR / 'station_list_glofas_final.yml'

# From Ragindra
RCO_DIR = DATA_DIR_PRIVATE / 'exploration/npl/unrco'
BASINS_SHAPEFILE = RCO_DIR / 'shapefiles/Major_River_Basins.shp'

# Final stations file
STATIONS_FINAL = {
    'glofas': DATA_DIR / 'npl/glofas/npl_glofas_stations_final.gpkg',
    'dhm': GOV_DIR / 'npl_dhm_stations_final.gpkg'
}
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
          'glofas_id',
           'provider',
          'provider_id',
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
    table = table[[1, 2, 3, 4, 5, 7] + cnums_lat_lon[i]]
    table.columns = cnames
    table = gpd.GeoDataFrame(
        table, 
        geometry=gpd.points_from_xy(
            table['lon'], table['lat']))
    table_list.append(table)
    
df_glofas = pd.concat(table_list, ignore_index=True).set_crs("EPSG:4326")
# Fix a parsing error
df_glofas = df_glofas.replace({'DCrHoMss': 'DHM'})


df_glofas.to_file(GLOFAS_STATION_OUTPUT, driver="GPKG", index=False)

```

## Scrape stations from government website

```python
import time
if SCRAPE_GOV_STATIONS:

    df_gov = gpd.GeoDataFrame(columns=['web_id', 'station_index', 'name', 'lat', 'lon'])
    
    opts = Options()
    opts.set_headless()
    browser = webdriver.Firefox(options=opts)
       
    print(f'Navigating to {GOV_BASE_URL}...')
    browser.get(GOV_BASE_URL)
    print('...done')
    
    station_elements = []
    while len(station_elements) == 0:
        print('Stations list empty, retrying')
        station_elements = browser.find_elements_by_class_name('basin-station')
        time.sleep(5)
    print('looping through stations')
    for station in station_elements:
        # Navigate to link
        print(f'Going to station {station.text}')
        url = station.get_attribute("href")
        browser.get(url)    
        table = browser.find_elements_by_class_name("table")[0].text.split('\n')
        station_info = {
            'web_id': url.split("/")[-1],
            'station_index': table[0].split(' ')[-1],
            'name': station.text,
            'lat': table[1].split(' ')[-1],
            'lon': table[2].split(' ')[-1],
        }
        time.sleep(5)
        df_gov = df_gov.append(station_info, ignore_index=True)
    df_gov['geometry'] = gpd.points_from_xy(df_gov['lon'], df_gov['lat'])
    df_gov.to_file(GOV_STATION_FILENAME, driver="GPKG", index=False)
   
    # Get the basin - station relations
    browser.get(GOV_BASE_URL)
    e = browser.find_elements_by_class_name("FolderHierarchy")
    basin_station_list = e[0].text.split('\n')
    df_basin_station = pd.DataFrame({'name': basin_station_list})
    df_basin_station.to_csv(GOV_BASIN_FILENAME, index=False)
    
    browser.close()
    
else:
    df_gov = gpd.read_file(GOV_STATION_FILENAME, layer='npl_dhm_hydrology_stations')
    basin_station_list = pd.read_csv(GOV_BASIN_FILENAME, header=None)[0].to_list()
```

```python
# Add the basins to df_gov

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

df_gov['basin'] = df_gov['name'].map(station_basin_dict)

```

### Count stations requested by partners

Ragindra suggested using all stations in the Koshi basin south of Chatara, and all from Karnali. There is no station called Chatara but there is Chautara; however, I think it's wrong.

```python
# Get all Koshi stations
df_koshi = df_gov[df_gov['basin'] == 'Koshi']
n_koshi = len(df_koshi)
# Get all south of Chautara
max_lat = df_gov[df_gov['name'] == "Chautara"]['geometry'].iloc[0].y
df_koshi = df_koshi.cx[:, :max_lat]
n_koshi_chautara = len(df_koshi)

# Get all Karnali stations
df_karnali = df_gov[df_gov['basin'] == 'Karnali']
n_karnali = len(df_karnali)

print(f"Number of stations:\n{n_koshi} Koshi\n{n_koshi_chautara} Koshi south of Chautara\n{n_karnali} Karnali")
```

## Scrape DHM stations

```python
page = requests.get(DHM_BASE_URL)
soup = BeautifulSoup(page.content, 'html.parser')
```

```python
table = soup.find("table", { "class" : "list" })

cnames = ['station_id', 'river', 'name', 'lat', 'lon', 'elevation']
df_dhm = gpd.GeoDataFrame(columns=cnames)
rows = table.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    if len(cols) == 0: 
        continue
    cols = [ele.text.strip() for ele in cols]
    station_info = {cname: cols[i+1] for i, cname in enumerate(cnames)}
    df_dhm = df_dhm.append(station_info, ignore_index=True)

def dms2dd(dms, sep='_'):
    degrees, minutes, seconds = dms.split('_')
    return float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    
for q in ['lat', 'lon']:
    df_dhm[q] = df_dhm[q].apply(dms2dd)
df_dhm['geometry'] = gpd.points_from_xy(df_dhm['lon'], df_dhm['lat'])

df_dhm = df_dhm.set_crs("EPSG:4326")

df_dhm.to_file(DHM_STATION_FILENAME, driver="GPKG", index=False)

```

```python
# Try to join DHM and glofas

df_dhm = df_dhm.merge(df_glofas, 
                      how='outer', left_on='station_id', right_on='provider_id',
                      suffixes=['_dhm', '_glofas']
                     )
df_dhm = df_dhm.drop(columns=['river_basin'])

df_dhm = df_dhm.rename(columns={
    'station_id': 'id_dhm',
    'river': 'river_dhm', 
    'name': 'name_dhm', 
    'elevation': 'elevation_dhm', 
    'station_name': 'name_glofas',
    'river_name': 'river_glofas',
    'glofas_id': 'id_glofas',
    'provider': 'provider_glofas',
    'provider_id': 'provider_id_glofas'
    
})
df_dhm['geometry'] = df_dhm['geometry_glofas']
idx = df_dhm['geometry'] == None
df_dhm.loc[idx, 'geometry'] = df_dhm.loc[idx, 'geometry_dhm']
df_dhm = gpd.GeoDataFrame(df_dhm, geometry=df_dhm['geometry'])
```

## Use shapefile from Ragindra

```python
basin_list = [
    'koshi', 
    'karnali', 
    'west rapti', 
    'babai', 
    'bagmati'
]
df_basins = gpd.read_file(BASINS_SHAPEFILE)

basin_shape_dict = {
    basin : df_basins[df_basins['Major_Basi'] == f'{basin.title()} River Basin']['geometry'].iloc[0]
    for basin in basin_list
}
```

```python
# Get GloFAS and DHM stations for both basins
with pd.ExcelWriter(STATION_OUTPUT) as writer:
    for basin in basin_list:
        df = df_dhm[df_dhm.geometry.within(basin_shape_dict[basin])]
        df.to_excel(writer, sheet_name=basin, index=False)
```

## Get final list of stations to put into config file

```python
# Based on email exchange with Ragindra
# Using GloFAS ID since two stations in Koshi have exactly the same name
stations_final = {
    'koshi': [4475, 4619, 4425, 4403],
    'karnali': [4385, 915, 4469, 4393],
    'rapti': [4456],
    'bagmati': [4399],
    'babai': [4416]
}
stations_final_yaml = {}
for basin in stations_final.values():
    for glofas_id in basin:
        row = df_dhm.loc[df_dhm['id_glofas'] == f'G{str(glofas_id).zfill(4)}'].iloc[0]
        stations_final_yaml[row['name_glofas']] = {
            'lat': row['lat_glofas'],
            'lon': row['lon_glofas']
        }

        
with open(STATION_OUTPUT_YAML, 'w') as f:
    yaml.dump(stations_final_yaml, f)
```

### Create final points shapefile for making map visuals

```python
sheet_id = "1I3vszdCDDxFnlhieywY2C9MbU-0TmGVn"
url = "https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}"


  
for source in ['glofas', 'dhm']:
    df_final = gpd.GeoDataFrame()
    for sheet_name in ['Koshi', 'Karnali', 'West_Rapti', 'Bagmati', 'Babai']:
        df = pd.read_csv(url.format(sheet_id, sheet_name)).dropna(subset=['name_glofas'])
        df = gpd.GeoDataFrame(df[[f'name_{source}', f'lat_{source}', f'lon_{source}']])
        df_final = df_final.append(df, ignore_index=True)
    df_final['geometry'] = gpd.points_from_xy(df_final[f'lon_{source}'], df_final[f'lat_{source}'])
    df_final = df_final.set_crs("EPSG:4326")
    df_final.to_file(STATIONS_FINAL[source], driver="GPKG", index=False)

```

# Appendix


### Try to connect gov with dhm 

```python
# Match it to the gov hydrology stations
# It would be more proper to do a merge but it's hard with two possible columns

df_dhm = df_dhm.dropna(subset=['river_dhm'])
df_dhm['gov_name'] = ''
for i, row in df_dhm.iterrows():
    gov_name_1 = row['river_dhm'].split('River')[0].strip() + ' at ' + row['name_dhm']
    gov_name_2 = row['river_dhm'] + ' at ' + row['name_dhm']
    for gov_name in gov_name_1, gov_name_2:
        matches = df_gov['name'] == gov_name
        if sum(matches):
            df_dhm.at[i, 'gov_name'] = gov_name
```

```python
match_dict = {
    'Jamu': 'Jamu, Dailekh',
    'Kusum': 'Rapti River at Kusum (Velocity)',
    'Borlangpul': 'Aadhi Khola at Borlangpul',
    'Angsing': 'Jyagdi Khola at Dangsing',
    'Bimalnagar': 'Marsyangdi at Bimalnagar',
    'Betrawati': 'Trishuli at Betrawati',
    'Rajaiya': 'Rajaiya (rainfall station)',
    'Lothar': 'Lothar Khola at Lothar',
    'Jalbire': 'Balefi at Jalbire',
    'Rasnalu': 'KhimtiKhola at Rasnalu',
    'Chatara': 'Saptakoshi at Chatara (old)'
}
is_null = df_dhm['gov_name'] == ''
df_dhm['gov_name'][is_null] = df_dhm['name'][is_null].map(match_dict)
```

I give up, will find another way to get the basin
