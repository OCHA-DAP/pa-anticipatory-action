```python
import os
from pathlib import Path
import requests

import tabula
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver



DATA_DIR = Path(os.environ["AA_DATA_DIR"])
STATION_FILE = DATA_DIR / 'public/exploration/npl/glofas/station_list.pdf'


```

```python
tables = tabula.read_pdf(STATION_FILE, pages=[51,52], 
                         pandas_options=dict(
                             header=None,
                         )
                        )

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
    table_list.append(table)
    
stations_glofas = pd.concat(table_list, ignore_index=True)
```

```python
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

```

```python
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
opts = Options()
opts.set_headless()
browser = webdriver.Firefox(options=opts)
```

```python
URL = "http://www.hydrology.gov.np/#/basin/144?_k=pywj7i"
browser.get(URL)
```

```python
e = browser.find_elements_by_class_name("stationTitle")
```

```python
e[0].text
```

```python
e = browser.find_elements_by_class_name("table")
```

```python
e[0].text.split('\n')
```

```python


URL = "http://www.hydrology.gov.np/#/basin/144"
browser.get(URL)
```

```python
e = browser.find_elements_by_class_name("stationTitle")
e[0].text
```

```python
import pandas as pd

df_np = pd.DataFrame(columns=['id', 'name', 'lat', 'lon'])

BASE_URL = f"http://www.hydrology.gov.np/#/basin"
for nstation in range(1000):
    browser.get(f"{BASE_URL}/{nstation}")
    name = browser.find_elements_by_class_name("stationTitle")
    if len(name) == 0:
        continue
    table = browser.find_elements_by_class_name("table")[0].text.split('\n')
    station_info = {
        'id': nstation,
        'name': name[0].text,
        'lat': table[1].split(' ')[-1],
        'lon': table[2].split(' ')[-1]
    }
    df_np = df_np.append(station_info, ignore_index=True)

```

```python
df_np
```

```python
url = "http://www.hydrology.gov.np/#/basin"
browser.get(URL)
e = browser.find_elements_by_class_name("FolderHierarchy")
station_list = e[0].text.split('\n')
```

```python
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
for station in station_list:
    if station in basin_list:
        active_basin = station
        basin_station_dict[active_basin] = []
    else:
        basin_station_dict[active_basin].append(station)
    
```

```python

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        for item in d[key]:
            inverse[item] = key
    return inverse

station_basin_dict = invert_dict(basin_station_dict)

```

```python
df_np['basin'] = df_np['name'].map(station_basin_dict)
```
