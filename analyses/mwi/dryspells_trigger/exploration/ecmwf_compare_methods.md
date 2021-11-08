# Compare ECMWF data when using rounded vs non-rounded coordinates
This notebook creates the figures for the analysis documented [here](https://docs.google.com/presentation/d/10IJDJhFPNoo8L4z959VAACarvQMJjUsP3OVBdST41B8/edit?usp=sharing).  
In summary, the ECMWF data used here is retrieved from the Copernicus Climate Data Store by using their API.     
When using this API, you can define an area, which for this analysis was an area surrounding Malawi.     
We originally used float coordinates to define the area boundaries. Later we realized that the original forecast is produced at integer coordinates.    
When downloading with the float coordinates, valid data is returned but it is interpolated. 

This notebook shows the differences on raster level between the two methods. All other notebooks that use this data, have the parameter `use_unrounded_area_coords` included. By setting this to `True` or `False`, the differences on outcomes of the analyses can be computed between the two download methods. 

```python
from pathlib import Path
import os
import sys
import geopandas as gpd
from importlib import reload
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
```

```python
path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.ecmwf_seasonal import processing
from src.indicators.drought.config import Config

reload(processing);
```

```python
COUNTRY_ISO3 = "mwi"
config = Config()
PARAMETERS = config.parameters(COUNTRY_ISO3)

COUNTRY_DATA_RAW_DIR = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / COUNTRY_ISO3

COUNTRY_DATA_PROCESSED_DIR = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / COUNTRY_ISO3

plots_dir=os.path.join(COUNTRY_DATA_PROCESSED_DIR,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")
```

```python
ADM1_BOUND_PATH = os.path.join(
    COUNTRY_DATA_RAW_DIR, config.SHAPEFILE_DIR, PARAMETERS["path_admin1_shp"]
)
```

```python
hdx_blue="#007ce0"
```

```python
gdf_adm=gpd.read_file(ADM1_BOUND_PATH)
gdf_south=gdf_adm[gdf_adm.ADM1_EN=="Southern"]
```

```python
#load the data with rounded (i.e. integer) and unrounded (i.e. float) area boundaries
da_for_unrounded=processing.get_ecmwf_forecast_by_leadtime("mwi",use_unrounded_area_coords=True)
da_for_unrounded = da_for_unrounded.rio.write_crs("EPSG:4326",inplace=True)
da_for_rounded=processing.get_ecmwf_forecast_by_leadtime("mwi",use_unrounded_area_coords=False)
da_for_rounded = da_for_rounded.rio.write_crs("EPSG:4326",inplace=True)
```

```python
# we can see that the lat and lon end with .126 and .67 respectively
da_for_unrounded
```

```python
# we can see that the lat and lon are at integer values
da_for_rounded
```

```python
#select a leadtime and ensemble number, such that we have 2D data for plotting
#in this notebook we are not interested in the exact values, but more on the 
#coordinates of the data
da_unrounded_sel=da_for_unrounded.sel(time="2020-01-01",leadtime=3,number=5)
da_rounded_sel=da_for_rounded.sel(time="2020-01-01",leadtime=3,number=5)
```

```python
#plot all data with mwi's boundaries
#here we can see that the centre of the cells is not at the
#integer value
g=da_unrounded_sel.plot()
gdf_adm.boundary.plot(ax=g.axes);
```

```python
#here we can see that the centre of the cells is 
#at the integer
g=da_rounded_sel.plot()
gdf_adm.boundary.plot(ax=g.axes);
```

```python
#nicer graphic, bounded to area around malawi
#also used a slide for a presentation
g=da_rounded_sel.sel(longitude=slice(33,36),latitude=slice(-9,-18)).plot(figsize=(6,10),levels=[0,50,100,150,200,250,300,350,400,450,500],cmap="Oranges",cbar_kwargs={'label': 'forecasted precipitation (mm)','pad':0.2})
g.axes.set_title("Example of ECMWF's forecasted monthly precipitation")
g.axes.set_xlabel("longitude")
g.axes.set_ylabel("latitude")
gdf_adm.boundary.plot(ax=g.axes,color="#888888",linewidth=2)
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
# plt.savefig(os.path.join(plots_seasonal_dir,f"mwi_example_raster_ecmwf.png"))
```

```python
g=da_unrounded_sel.sel(longitude=slice(32,36),latitude=slice(-9,-18)).plot(figsize=(6,10),levels=[0,50,100,150,200,250,300,350,400,450,500],cmap="Oranges",cbar_kwargs={'label': 'forecasted precipitation (mm)','pad':0.2})
g.axes.set_title("Unrounded coords: Example of ECMWF's forecasted monthly precipitation")
g.axes.set_xlabel("longitude")
g.axes.set_ylabel("latitude")
gdf_adm.boundary.plot(ax=g.axes,color="#888888",linewidth=2)
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
```

```python
# da_south_unrounded=da_for_unrounded.rio.clip(gdf_south["geometry"], all_touched=False)
```

To compute the trigger, we only include cells that have their centre within the Southern region.    
The plots below show which cells are included with the unrounded and rounded method.    
As you can see these differ and thus produce different results   

```python
da_sel_unrounded_south=da_unrounded_sel.rio.clip(gdf_south["geometry"], all_touched=False)
da_rounded_sel_country=da_rounded_sel.rio.clip(gdf_adm["geometry"], all_touched=False)
```

```python
g=da_sel_unrounded_south.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10))
gdf_adm.boundary.plot(ax=g.axes,color="grey");
```

```python
#the where is a hack to plot it correctly, else cells shown as a line instead of square
g=da_rounded_sel_country.squeeze().where(da_rounded_sel_country.longitude>=35).plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(6,10))
gdf_adm.boundary.plot(ax=g.axes,color="grey");
```

Compared to during the development, we are now more aware of the risk of only including two cells.   
For future improvement, we should better understand if we should include a wider area.   
Two options for future inclusion of cells are shown below

```python
#include cells around the region as well
g=da_rounded_sel.sel(longitude=slice(34,36),latitude=slice(-14,-17)).plot.imshow(figsize=(6,10),cmap=matplotlib.colors.ListedColormap([hdx_blue]))
gdf_adm.boundary.plot(ax=g.axes,color="grey");
```

Option 2: Cut cells into smaller pieces and include all smaller pieces within the southern region

```python
def interp_ds(ds):
    new_lon = np.arange(
        ds.longitude[0] - 0.125, ds.longitude[-1] + 0.25, 0.25
    )
    new_lat = np.arange(
        ds.latitude[0] + 0.125, ds.latitude[-1] - 0.25, -0.25
    )

    ds = ds.interp(latitude=new_lat, longitude=new_lon, method="nearest")
    return ds
```

```python
da_rounded_sel_inerp_south=interp_ds(da_rounded_sel).rio.clip(gdf_south["geometry"], all_touched=False)
```

```python
g=da_rounded_sel_inerp_south.plot.imshow(cmap=matplotlib.colors.ListedColormap([hdx_blue]),figsize=(10,15))
gdf_adm.boundary.plot(ax=g.axes,color="grey");
```
