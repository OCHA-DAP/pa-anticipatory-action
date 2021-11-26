### Compare data from CDS and ECMWF directly (API) 
This notebook is for checking that the ECMWF data downloaded
from their API is equivalent and has been processed in a similar
manner to the data from CDS that we've been using so far.

The original data from the models is provided by ECMWF directly, which we retrieve from their API. This data has a 0.4 degree resolution. However, the big disadvantage is that it is not openly available. Till Nov 2021 we didn't have access to this data, so couldn't use it. 

Therefore we used the open source data provided by CDS. Which in principle is the same, but aggregated to a 1 degree resolution.  

In this notebook, we check if the values from both sources approximately agree and how many cells are include with the data from the different sources. 

```python
from pathlib import Path
import os
import sys
from datetime import datetime
from importlib import reload
import matplotlib
import geopandas as gpd
from rasterio.enums import Resampling
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.ecmwf_seasonal import processing
from src.indicators.drought.config import Config
reload(processing)
```

```python
ISO3 = 'mwi'
VAR = 'precip'
```

```python
#plot color
hdx_blue="#007ce0"
```

```python
config = Config()
PARAMETERS = config.parameters(ISO3)
COUNTRY_DATA_RAW_DIR = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / ISO3
ADM1_BOUND_PATH = os.path.join(
    COUNTRY_DATA_RAW_DIR, config.SHAPEFILE_DIR, PARAMETERS["path_admin1_shp"]
)
```

```python
#adm boundaries
gdf_adm=gpd.read_file(ADM1_BOUND_PATH)
gdf_south=gdf_adm[gdf_adm.ADM1_EN=="Southern"]
```

```python
# Read in the two dataframes: One from CDS and the other from ECMWF API
da_cds = processing.get_ecmwf_forecast(ISO3)[VAR]
da_ecmwf =  processing.get_ecmwf_forecast(ISO3, source_cds=False)[VAR]
```

```python
# Need to flatten the data arrays to compare parameters across x-y coordinates.
# Can play around with these to view different times, ensemble member numbers,
# and forecast steps. 
time = datetime(1999, 1, 1)
number = 3
step = 2

data_cds = da_cds.sel(time=time, number=number, step=step).rio.write_crs("EPSG:4326",inplace=True)
data_ecmwf = da_ecmwf.sel(time=time, number=number, step=step).rio.write_crs("EPSG:4326",inplace=True)
```

```python
# CDS data has 1 deg resolution
data_cds.plot()
```

```python
# API data has 0.4 deg resolution
data_ecmwf.plot()
```

## Comparing at same res (allowing ECMWF API to interpolate)
To understand if the two data sources approximately align, we downsample the resolution of the API data to the same resolution as the CDS data. 
The API call takes a resolution parameter so for testing we set this to 1 degree. Another option would be to coarsen the API data after downloading, but we assume that changing the res at the API call directly is more clean. 

```python
# Monkey-patch the filepath to access the 1 degree directory
# Not hard-coding it in for now because it probably won't be necessary for 
# further analysis
processing.ECMWF_API_FILEPATH = processing.ECMWF_API_FILEPATH.replace("ecmwf", "ecmwf_1deg")
```

```python
da_ecmwf_lowres =  processing.get_ecmwf_forecast(ISO3, source_cds=False)[VAR]
```

```python
data_ecmwf_lowres = da_ecmwf_lowres.sel(time=time, number=number, step=step)
```

```python
data_ecmwf_lowres.plot()
```

```python
# Compare difference
(data_cds - data_ecmwf_lowres).plot()
```

As we can see from the plot above there are some differences between the two data sources. However, in the given example these differ not more than 20 mm. Given the range of precipitation this seems reasonable. We assume the differences are caused by differences in interpolation methods between the two sources. Based on this simple analysis we assume the API data is correct and we continue using that. 


### Pixels included
For the trigger we only select the Southern region of Malawi. Due to the low resolution of CPS data this lead to only two cells being included. We plot how this changes with the higher resolution data

```python
#the where is a hack to plot it correctly, else cells shown as a line instead of square
data_cds_country=data_cds.rio.write_crs("EPSG:4326").rio.clip(gdf_adm["geometry"])
data_cds_south=data_cds.rio.write_crs("EPSG:4326").rio.clip(gdf_south["geometry"])
g=data_cds_country.where(data_cds_country.longitude>=35).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm.boundary.plot(ax=g.axes,color="grey")
g.axes.set_title(f"Cells with centre in area with ECMWF CDS data: {data_cds_south.count().values} cells included");
```

```python
data_ecmwf_south=data_ecmwf.rio.write_crs("EPSG:4326").rio.clip(gdf_south["geometry"])
g=data_ecmwf_south.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm.boundary.plot(ax=g.axes,color="grey")
g.axes.set_title(f"Cells with centre in area with ECMWF API data: {data_ecmwf_south.count().values} cells included");
```

Instead of only including the cells with the centre in the region, we can also choose to include all cells touching the region, or to do an approximate masking of the region. The results of this are shown below

```python
#plot all cells touching the region with API data
data_ecmwf_south_touch=data_ecmwf.rio.write_crs("EPSG:4326").rio.clip(gdf_south["geometry"],all_touched=True)
g=data_ecmwf_south_touch.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10),add_colorbar=False)
gdf_adm.boundary.plot(ax=g.axes,color="grey")
g.axes.set_title(f"Cells touching area with ECMWF API data: {data_ecmwf_south_touch.count().values} cells included");
```

```python
data_ecmwf_wavg = data_ecmwf.rio.reproject(
    data_ecmwf.rio.crs,
    #resolution it will be changed to
    resolution=0.05,
    #use nearest so cell values stay the same, only cut
    #into smaller pieces
    resampling=Resampling.nearest,
    nodata=np.nan,
)
```

```python
g=data_ecmwf_wavg.rio.clip(gdf_south["geometry"], all_touched=False).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(10,15),add_colorbar=False)
gdf_adm.boundary.plot(ax=g.axes,color="grey");
g.axes.set_title(f"Approximate weighted average with ECMWF API data");
```
