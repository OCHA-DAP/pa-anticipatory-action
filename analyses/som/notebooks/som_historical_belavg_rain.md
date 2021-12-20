### Exploring historical below average precipitation seasons
This notebook explores the occurrences of historical below average precipitation. 

The main focus is on the OND and MAM seasons. 

This notebook was started to explore how of an extraordinary event it is that during 3+ consecutive OND/MAM seasons large parts of the country experience below average precipitation. 

Simeltaneously, it serves as a start for other explorations related to historical precipitation. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.chirps_rainfallobservations import compute_seasonal_tercile_raster, \
get_filepath_seasonal_lowertercile_raster
```

#### Set config values

```python
iso3="som"
config=Config()
parameters = config.parameters(iso3)

public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,iso3)
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

```python
## compute historical below average precipitation
## only needed if data needs to be updated
# get_chirps_data_monthly(config,iso3, use_cache=False)
# compute_seasonal_tercile_raster(config, iso3, use_cache=False)
```

```python
#open historical tercile data
ds = rioxarray.open_rasterio(get_filepath_seasonal_lowertercile_raster(iso3,config)).sortby("time")
```

```python
da=ds.precip
#units has a list of mm/month when not changing this
da.attrs["units"]="mm/month"
```

```python
#the month parameter indicates the last month of the season (=3month period)
#thus 12 and 5 indicate the OND and MAM season
da_ondmam=da.where(da.time.dt.month.isin([12,5]), drop=True)
```

```python
#plot seasons from 2000
da_ondmam.sel(time=slice('2000', '2021')).plot(    
    col="time",
    col_wrap=4,
    levels=[-666,0],
    colors=['#cccccc','#f2645a'],
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },)
```

```python
#compute the percentage of cells with below average precipitation
#all cells that are -666 didn't have below average precipitation
#so cells with >=0 did have bavg preciptiation
#there are also nan cells but these are automatically ignored by count, which is good
da_ondmam_perc_bavg=da_ondmam.where(da_ondmam>=0).count(dim=["x","y"])/da_ondmam.count(dim=["x","y"])*100
```

```python
#transform to dataframe
df_ondmam_perc_bavg=da_ondmam_perc_bavg.to_dataframe()
df_ondmam_perc_bavg.drop("spatial_ref",axis=1,inplace=True)
df_ondmam_perc_bavg.rename(columns={"precip":"perc_bavg"},inplace=True)
df_ondmam_perc_bavg.reset_index(inplace=True)
df_ondmam_perc_bavg.time=pd.to_datetime(df_ondmam_perc_bavg.time.apply(lambda x: x.strftime('%Y-%m-%d')))
```

```python
#plot years after 2010
df_ondmam_perc_bavg[df_ondmam_perc_bavg.time.dt.year>=2010]
```

#### How do we define a "below average season"? 
We wanted to quantify the country as having experienced a below average season. However, what does that mean? 

We decided to set a threshold on the minimum percentage of the country that had below average precipitation. 

We set this to 30% as we deemed that as a substantial part of the country. However, this is not backed up by evidence and very much open to discussion. 
For example with a 30% threshold the bad drought of 2016-2017 is not included, whereas with 28% it is.  
At the very least, a distinction should be made which parts of the areas are sensitive to bavg OND/MAM seasons. 

```python
perc_thresh=30
```

```python
df_ondmam_perc_bavg[f"ge_{perc_thresh}"]=np.where(df_ondmam_perc_bavg.perc_bavg>=perc_thresh,1,0)
```

```python
df_ondmam_perc_bavg['consecutive'] = df_ondmam_perc_bavg[f"ge_{perc_thresh}"].groupby( \
    (df_ondmam_perc_bavg[f"ge_{perc_thresh}"] != df_ondmam_perc_bavg[f"ge_{perc_thresh}"].shift()).cumsum()).transform('size') * \
    df_ondmam_perc_bavg[f"ge_{perc_thresh}"]

```

```python
#check occurrences of at last 3 bad consecutive seasons
df_ondmam_perc_bavg[df_ondmam_perc_bavg.consecutive>=3]
```

Many source state that the 2020 OND and 2021 MAM seasons were bad in Somalia. This is reflected by the percentage of the country having received below average rainfall.    

This is really an extreme situation for Somalia, and a third consecutive season with a high percentage of below average rain, would be an extraordinary event that can lead to a high impact. 

```python
df_ondmam_perc_bavg[df_ondmam_perc_bavg.time.dt.year.isin([2020,2021])]
```
