## Compute dry spells detected by ARC2
[ARC2](https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/.Africa/.DAILY/.ARC2/.daily/.est_prcp/) is a data set of daily observed precipitation data that is updated relatively quickly compared to other data sets. 

This notebook retrieves the data, computes dry spells, and merges it with observed dry spells in the CHIRPS dataset in order to compare the two. 

Note that this notebook is in a very experimental state. However, since after exploration it was decided to not use ARC2 as a data source as of now, we don't invest time in improving this. The reason of not using ARC2 was because it is observational data and a preference was given for forecast data. 

If later on it is decided to use ARC2, a part of this notebook should be converted into a .py script, and a faster method for computing the statistics should be sought after. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import rioxarray
from shapely.geometry import mapping
import geopandas as gpd
import xarray as xr
import requests
import numpy as np
from rasterstats import zonal_stats
from datetime import timedelta
```

#### Set config values

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
country="mwi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
dry_spells_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")

arc2_dir = os.path.join(country_data_exploration_dir,"arc2")
arc2_filepath = os.path.join(arc2_dir, "arc2_20002020_approxmwi.nc")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

### Download the data
Takes some time --> can better do this in a Python script

```python
#only include coordinates that approximately cover MWI --> else waaaay huger
#TODO: make url flexible to include up to current date
arc2_mwi_url="https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/.Africa/.DAILY/.ARC2/.daily/.est_prcp/T/%281%20Jan%202000%29%2830%20Mar%202021%29RANGEEDGES/X/%2832E%29%2836E%29RANGEEDGES/Y/%2820S%29%285S%29RANGEEDGES/data.nc"
```

```python
# strange things happen when just overwriting the file, so delete it first if it already exists
if os.path.exists(arc2_filepath):
    os.remove(arc2_filepath)

#have to authenticate by using a cookie
cookies = {
    '__dlauth_id': os.getenv("IRI_AUTH"),
}

# logger.info("Downloading arc2 NetCDF file. This might take some time")
response = requests.get(arc2_mwi_url, cookies=cookies, verify=False)

with open(arc2_filepath, "wb") as fd:
    for chunk in response.iter_content(chunk_size=128):
        fd.write(chunk)
```

```python
#open the data
ds=rioxarray.open_rasterio(arc2_filepath,masked=True).squeeze()
#convert to dataset instead of datarray --> makes selection of variables easier
ds=ds.to_dataset()
#fix units attribute
ds.attrs["units"]='mm/day'
```

```python
ds
```

```python
#clip to MWI
df_bound=gpd.read_file(adm1_bound_path)
ds_clip=ds.rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
```

```python
#simple plot
g=ds_clip.sel(T="2021-03-28").est_prcp.plot()
df_bound.boundary.plot(ax=g.axes)
```

```python
#compute the 14 day rolling sum per cell
#window is to the right, inclusive
ds_rolling=ds_clip.rolling(T=14,min_periods=14).sum().dropna(dim="T",how="all")
```

```python
#TODO: ARC2 has lower resolution than CHIRPS (0.1 vs 0.05)--> look into whether to use all cells touching instead of only center. Or possibly interpolating to higher resolution
def alldates_statistics(ds,raster_transform,adm_path,dim_col="est_prcp",ds_thresh_list=[2,4,8,10,15,20]):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    for date in ds.T.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(T=date)

        df["mean_cell"] = pd.DataFrame(
            zonal_stats(vectors=df, raster=ds_date[dim_col].values, affine=raster_transform, nodata=np.nan))["mean"]
        df["mean_cell_touched"] = pd.DataFrame(
            zonal_stats(vectors=df, raster=ds_date[dim_col].values, affine=raster_transform, nodata=np.nan,all_touched=True))["mean"]

        for thres in ds_thresh_list:
            # compute the percentage of the admin area that has cells below the threshold
            # set all values with below average rainfall to 1 and others to 0
            forecast_binary = np.where(ds_date[dim_col].values <= thres, 1, 0)
            # compute number of cells in admin region (sum) and number of cells in admin region with below average rainfall (count)
            bin_zonal = pd.DataFrame(
                zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],
                            nodata=np.nan))
            df[f'perc_se{thres}'] = bin_zonal['sum'] / bin_zonal['count'] * 100
        
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
  
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")
        
    return df_hist
```

```python
# #only needed if not computed yet
# #This takes several hours --> better do it in a .py script
# #compute statistics on adm2 level per date
# df=alldates_statistics(ds_rolling,ds_rolling.rio.transform(),adm2_bound_path)
# df.drop("geometry",axis=1).to_csv(os.path.join(country_data_exploration_dir,"arc2","mwi_arc2_precip_long.csv"))
```

```python
#approx total number of days expected
21*365*32
```

```python
#load the above 
#only show column names, since this is quick to load, and full dataframe isn't
pd.read_csv(os.path.join(arc2_dir,"mwi_arc2_precip_long.csv"),nrows=1)
```

```python
#load the data. Faster if only choosing a selection of columns
df=pd.read_csv(os.path.join(arc2_dir,"mwi_arc2_precip_long.csv"),usecols=["date","ADM2_PCODE","mean_cell","perc_se2"])# ,nrows=10000)
df.date=pd.to_datetime(df.date)
df.rename(columns={"ADM2_PCODE":"pcode"},inplace=True)
```

```python
df
```

```python
# df.to_csv(os.path.join(arc2_dir,"mwi_arc2_precip_long_sel.csv"))
```

```python
#define dry spells
df["dry_spell"]=np.where(df.mean_cell<=2,1,0)
```

```python
#select only dates with dry spell
df_ds=df[df["dry_spell"]==1]
df_ds=df_ds.sort_values(["pcode","date"]).reset_index(drop=True)
```

```python
#assign ID per consecutive sequence of dry spell days
df_ds["ID"]=df_ds.groupby("pcode").date.diff().dt.days.ne(1).cumsum()
```

```python
#group the data, such to have one entry per dry spell, where the start and end date are indicated. 
df_ds_grouped=pd.DataFrame({'dry_spell_first_date' : df_ds.groupby('ID').date.first()-timedelta(days=13), 
              'dry_spell_last_date' : df_ds.groupby('ID').date.last(),
              'dry_spell_duration' : df_ds.groupby('ID').size()+13, 
              'dry_spell_confirmation' : df_ds.groupby('ID').date.first(),
              'pcode' : df_ds.groupby('ID').pcode.first()}).reset_index(drop=True)


```

```python
df_ds_grouped
```

```python
# df_ds_grouped.to_csv(os.path.join(country_data_exploration_dir,"arc2","mwi_arc2_dry_spells.csv"))
```

```python
#add start year of the rainy season
df_ds_grouped["season_approx"]=np.where(df_ds_grouped.dry_spell_first_date.dt.month>=10,df_ds_grouped.dry_spell_first_date.dt.year,df_ds_grouped.dry_spell_first_date.dt.year-1)
```

```python
#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean_back.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])
```

```python
df_rain
```

```python
#add rainy season start and enddate to df
df_ds_grouped_raindates=df_ds_grouped.merge(df_rain[['pcode', 'season_approx', 'onset_date', 'cessation_date']], on = ['pcode', 'season_approx']) 
```

```python
#compute whether date on which dry spell was confirmed, was within the rainy season
df_ds_grouped_raindates["during_rainy_season"]=np.where((df_ds_grouped_raindates.dry_spell_confirmation>=df_ds_grouped_raindates.onset_date)&(df_ds_grouped_raindates.dry_spell_confirmation<=df_ds_grouped_raindates.cessation_date),1,0)
```

```python
#only select dry spells within rainy season
df_ds_rain=df_ds_grouped_raindates[df_ds_grouped_raindates.during_rainy_season==1]
```

```python
len(df_ds_rain)
```

### Heatmap
Structure data such that it can be used for the R code to create a heatmap

```python
#make list of all dates within a dry spell for arc2
#important to reset the index, since that is what is being joined on
df_ds_rain=df_ds_rain.reset_index(drop=True)
#create datetimeindex per row
a = [pd.date_range(*r, freq='D') for r in df_ds_rain[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_arc_daterange=df_ds_rain[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
df_arc_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_arc_daterange["dry_spell_arc"]=1
```

```python
#load chirps dry spells, since that is what we want to compare to
dry_spells_list_path=os.path.join(dry_spells_processed_dir,f"dry_spells_during_rainy_season_list_2000_2020_mean_back.csv")
df_ds_chirps=pd.read_csv(dry_spells_list_path)
df_ds_chirps["dry_spell_first_date"]=pd.to_datetime(df_ds_chirps["dry_spell_first_date"])
df_ds_chirps["dry_spell_last_date"]=pd.to_datetime(df_ds_chirps["dry_spell_last_date"])
df_ds_chirps["year"]=df_ds_chirps.dry_spell_first_date.dt.year

#create df with all dates that were part of a dry spell per adm2
#important to reset the index, since that is what is being joined on
df_ds_chirps_res=df_ds_chirps.reset_index(drop=True)
#create datetimeindex per row
a = [pd.date_range(*r, freq='D') for r in df_ds_chirps_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_chirps_daterange=df_ds_chirps_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
df_ds_chirps_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_chirps_daterange["dry_spell_chirps"]=1
```

```python
#merge arc2 and chirps data
df_ds_both=df_arc_daterange.merge(df_ds_chirps_daterange,how="outer",on=["date","pcode"])
df_ds_both.dry_spell_chirps=df_ds_both.dry_spell_chirps.replace(np.nan,0)
df_ds_both.dry_spell_arc=df_ds_both.dry_spell_arc.replace(np.nan,0)
df_ds_both["season_approx"]=np.where(df_ds_both.date.dt.month>=10,df_ds_both.date.dt.year,df_ds_both.date.dt.year-1)
```

```python
def label_ds(row):
    if row["dry_spell_chirps"]==1 and row["dry_spell_arc"]==1:
        return 3
    elif row["dry_spell_chirps"]==1:
        return 2
    elif row["dry_spell_arc"]==1:
        return 1
    else:
        return 0
```

```python
#encode dry spells and whether both sources observed a dry spell, or only one of the two
df_ds_both["dryspell_match"]=df_ds_both.apply(lambda row:label_ds(row),axis=1)
```

```python
#add dates that are not present in df_ds_both, i.e. outside rainy season
df_ds_both_filled=df_ds_both.sort_values('date').set_index(['date']).groupby('pcode').apply(lambda x: x.reindex(pd.date_range(pd.to_datetime('01-01-2000'), pd.to_datetime('31-12-2020'), name='date'),fill_value=0).drop('pcode',axis=1).reset_index()).reset_index().drop("level_1",axis=1)
```

```python
#cause for now we only wanna show till end of 2020 cause no processed chirps data after that
df_ds_both_filled=df_ds_both_filled[df_ds_both_filled.date.dt.year<=2020]
```

```python
df_ds_both_filled
```

```python
# df_ds_both_filled.to_csv(os.path.join(country_data_exploration_dir,"dryspells",f"dryspells_arc2_dates_viz_th2.csv"))
```

```python
df_ds_rain.groupby("pcode").dry_spell_duration.mean()
```

### Test with opendap
Which would be ideal, but failing due to not being able to get the packages right

this is a known issue due to NetCDF 1.5.5, see https://github.com/pydata/xarray/issues/4925. 
However, if I downgroad that in the requirements.txt to 1.5.1 the issue still persists
when using conda I do get the second example to work, but don't manage to correctly install geopandas, rioxarray and rasterio

```python
def fix_calendar(ds, timevar='F'):
    """
    Some datasets come with a wrong calendar attribute that isn't recognized by xarray
    So map this attribute that can be read by xarray
    Args:
        ds (xarray dataset): dataset of interest
        timevar (str): variable that contains the time in ds

    Returns:
        ds (xarray dataset): modified dataset
    """
    if "calendar" in ds[timevar].attrs.keys():
        if ds[timevar].attrs['calendar'] == '360':
            ds[timevar].attrs['calendar'] = '360_day'
    elif "units" in ds[timevar].attrs.keys():
        if "months since" in ds[timevar].attrs['units']:
            ds[timevar].attrs['calendar'] = '360_day'
    return ds

```

```python
arc2_url="https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/.Africa/.DAILY/.ARC2/.daily/dods"
```

```python
remote_data = xr.open_dataset(arc2_url,
                              decode_times=False)
```

```python
remote_data
```

```python
remote_data.T.attrs["units"]='days since 1960-01-01 00:00:00 UTC'
remote_data.T.attrs['calendar'] = 'julian'#'360_day' #"julian"#
```

```python
remote_data=remote_data.rename({"X":"lon","Y":"lat"})
remote_data=fix_calendar(remote_data,timevar="T")
remote_data = xr.decode_cf(remote_data)
```

```python
remote_data
```

```python
df_bound=gpd.read_file(adm1_bound_path)
```

```python
ds=remote_data.sel(T="2021-03-28").rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
```

```python
remote_data = xr.open_dataset("http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods",
                              decode_times=False)
tmax = remote_data["tmax"][:500, ::3, ::3]
tmax[0].plot()
```

```python

```
