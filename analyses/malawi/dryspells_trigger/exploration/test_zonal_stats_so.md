### Comparison methods for computing statistics within a polygon
Showing two methods on how statistics (in this case the mean) of all cells of a raster file within a polygon can be computed. 

One is rasterstats.zonal_stats which seems the status quo but is slow for large files
The other being to clip the raster to the polygon and then take the mean

This example includes all the data downloading


#### Set values and import libraries

```python
from rasterstats import zonal_stats
import rioxarray
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from pathlib import Path
```

```python
data_dir="data"
hdx_url_name="malawi-administrative-level-0-3-boundaries"
hdx_dataset_name='mwi_adm_nso_20181016_SHP.zip'
hdx_dataset_dir=Path(data_dir) / 'mwi_adm_nso_20181016_SHP'
hdx_adm1_filename='mwi_admbnda_adm1_nso_20181016.shp'
hdx_adm1_path=hdx_dataset_dir / hdx_adm1_filename
chirps_path=Path(data_dir)/"chirps"/"chirps_daily_2020_p25.nc"
```

```python
import shutil
import zipfile
from urllib.request import urlretrieve

```

```python
from hdx.hdx_configuration import Configuration 
from hdx.data.dataset import Dataset
Configuration.create(hdx_site='prod', user_agent='A_Quick_Example', hdx_read_only=True)
```

#### Download data

```python
def hdx_download_filename(hdx_address,filename,directory):
    dataset = Dataset.read_from_hdx(hdx_address)
    resources = dataset.get_resources()
    for resource in resources:
        if resource["name"]==filename:
            _, path = resource.download()
            shutil.move(path, Path(directory) / filename)
```

```python
hdx_download_filename(hdx_url_name,hdx_dataset_name,data_dir)
```

```python
with zipfile.ZipFile(Path(data_dir)/hdx_dataset_name,"r") as zip_ref:
    zip_ref.extractall(hdx_dataset_dir)
```

```python
#download chirps data
urlretrieve("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.2020.days_p25.nc", filename=chirps_path)
```

#### Example for one date 
Simplest example, placed on Stack Overflow

```python
ds=rioxarray.open_rasterio(chirps_path,masked=True)
ds=ds.rio.write_crs("EPSG:4326")
```

```python
ds_date=ds.sel(time="2020-01-01").squeeze()
```

##### Compute with raster stats

```python
start_time = time.time()
gdf = gpd.read_file(hdx_adm1_path)[["ADM1_EN", "geometry"]]
gdf["mean_adm"] = pd.DataFrame(
            zonal_stats(
                vectors=gdf,
                raster=ds_date.values,
                affine=ds_date.rio.transform(),
                nodata=np.nan,
                all_touched=False
            )
        )["mean"]
print("--- Raster stats ---")
display(gdf[["ADM1_EN","mean_adm"]])
print(f"--- Raster stats: {(time.time() - start_time):.2f} seconds ---")
```

##### Compute with clipping

```python
start_time = time.time()
gdf = gpd.read_file(hdx_adm1_path)[["ADM1_EN", "geometry"]]
df_adm=pd.DataFrame(index=gdf.ADM1_EN.unique())
for a in gdf.ADM1_EN.unique():
    gdf_adm=gdf[gdf.ADM1_EN==a]

    da_clip = ds_date.rio.set_spatial_dims(
        x_dim="x", y_dim="y"
    ).rio.clip(
        gdf_adm["geometry"], all_touched=False
    )

    grid_mean = da_clip.mean(dim=["x", "y"],skipna=True).rename("mean_adm")
    df_adm.loc[a,"mean_adm"]=grid_mean.values
print("--- Rio clip ---")
display(df_adm)
print(f"--- Rio clip: {(time.time() - start_time):.2f} seconds ---")
```

#### Example for computation across several dates

```python
#only select two dates for demonstration
ds_sel=ds.sel(time=slice("2020-01-01","2020-01-02"))
```

```python
ds_sel
```

```python
def compute_zonal_stats_rs(
    ds,
    adm_path,
    time_col="time",
    all_touched=False,
):
    """use zonal_stats to compute statistics on level in adm_path for all dates in ds"""
    df_list = []
    for t in ds[time_col].values:
        gdf = gpd.read_file(adm_path)[["ADM1_EN", "geometry"]]
        ds_date = ds.sel(time=t)

        gdf["mean_cell"] = pd.DataFrame(
            zonal_stats(
                vectors=gdf,
                raster=ds_date.values,
                affine=ds_date.rio.transform(),
                nodata=np.nan,
                all_touched=all_touched
            )
        )["mean"]

        gdf["time"] = pd.to_datetime(t.strftime('%Y-%m-%d'))

        df_list.append(gdf)
    df_zonal_stats = pd.concat(df_list)
    df_zonal_stats = df_zonal_stats.drop("geometry", axis=1)

    return df_zonal_stats
```

```python
start_time = time.time()
df_stats_rs=compute_zonal_stats_rs(ds_sel,hdx_adm1_path)
print(f"--- Raster stats: {(time.time() - start_time):.2f} seconds ---")
df_stats_rs
```

```python
def compute_zonal_stats_clip(
    da, 
    adm_path, 
    lon_coord="x",
    lat_coord="y",
    all_touched=False,
):
    """use rio.clip to compute statistics on level in adm_path for all dates in ds"""
    df_list=[]
    gdf = gpd.read_file(adm_path)[["ADM1_EN", "geometry"]]
    for a in gdf.ADM1_EN.unique():
        gdf_adm=gdf[gdf.ADM1_EN==a]
    
        da_clip = da.rio.set_spatial_dims(
            x_dim=lon_coord, y_dim=lat_coord
        ).rio.clip(
            gdf_adm["geometry"], all_touched=all_touched
        )

        grid_mean = da_clip.mean(dim=[lon_coord, lat_coord],skipna=True).rename("mean_adm")
        df_adm=grid_mean.to_dataframe().reset_index().drop("spatial_ref",axis=1)
        df_adm["ADM1_EN"]=a
        df_list.append(df_adm)

    df_zonal_stats = pd.concat(df_list)
    df_zonal_stats.time=pd.to_datetime(df_zonal_stats.time.apply(lambda x: x.strftime('%Y-%m-%d')))
    return df_zonal_stats
```

```python
start_time = time.time()
df_stats_clip=compute_zonal_stats_clip(ds_sel.rio.write_crs("EPSG:4326"),hdx_adm1_path)
print(f"--- Clip and mean: {(time.time() - start_time):.2f} seconds ---")
df_stats_clip
```
