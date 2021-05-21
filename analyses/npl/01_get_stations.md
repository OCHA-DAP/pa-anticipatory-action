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
from bs4 import BeautifulSoup


DATA_DIR = Path(os.environ["AA_DATA_DIR"]) / 'public/exploration'

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
if SCRAPE_GOV_STATIONS:

    df_gov = gpd.GeoDataFrame(columns=['web_id', 'station_index', 'name', 'lat', 'lon'])
    
    opts = Options()
    opts.set_headless()
    browser = webdriver.Firefox(options=opts)
       
    print(f'Navigating to {GOV_BASE_URL}...')
    browser.get(GOV_BASE_URL)
    print('...done')
    
    station_elements = []
    while len(stations) == 0:
        print('Stations list empty, retrying')
        station_elements = browser.find_elements_by_class_name('basin-station')
        time.sleep(5)
    print('looping through stations')
    for station in station_elements:
        # Navigate to link
        print(f'Going to station {station.text}')
        url = element.get_attribute("href")
        browser.get(url)    
        table = browser.find_elements_by_class_name("table")[0].text.split('\n')
        station_info = {
            'web_id': url.split("/")[-1],
            'station_index': table[0].split(' ')[-1],
            'name': station.text,
            'lat': table[1].split(' ')[-1],
            'lon': table[2].split(' ')[-1],
        }
        df_gov = df_gov.append(station_info, ignore_index=True)
    df_gov['geometry'] = gpd.points_from_xy(df_gov['lon'], df_gov['lat'])
    df_gov.to_file(GOV_STATION_FILENAME, driver="GPKG", index=False)
   
    # Get the basin - station relations
    browser.get(BASE_URL)
    e = browser.find_elements_by_class_name("FolderHierarchy")
    basin_station_list = e[0].text.split('\n')
    df_basin_station = pd.DataFrame({'name': basin_station_list})
    df_basin_station.to_csv(GOV_BASIN_FILENAME, index=False)
    
    browser.close()
    
else:
    df_gov = gpd.read_file(GOV_STATION_FILENAME)
    basin_station_list = pd.read_csv(GOV_BASIN_FILENAME, header=None)[0].to_list()
```

### Connect government stations with basin


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

## Get DHM stations

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

## Use shapefile from gov website

```python
basin_list = ["koshi", "karnali"]

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
df_basins = gpd.read_file(GOV_BASIN_SHAPEFILE, driver='kml')

basin_shape_dict = {
    basin : df_basins[df_basins['Name'] == f'{basin.capitalize()} Basin']['geometry'].iloc[0]
    for basin in basin_list
}

```

```python
# Get GloFAS and DHM stations for both basins
output_file = 'tmp.xlsx'
with pd.ExcelWriter(output_file) as writer:
    for basin in basin_list:
        df = df_dhm[df_dhm.geometry.within(basin_shape_dict[basin])]
        df.to_excel(writer, sheet_name=basin, index=False)
```

# Appendix


### Try to connect gov with dhm 

```python
df_dhm
```

```python
# Match it to the gov hydrology stations
# It would be more proper to do a merge but it's hard with two possible columns

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
