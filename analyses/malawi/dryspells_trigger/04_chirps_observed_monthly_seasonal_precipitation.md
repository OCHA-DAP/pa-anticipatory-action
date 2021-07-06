<!-- #region -->
### Computation of observed below average precipitation
This notebook computes the historical **observed** precipitation for monthly and 3month periods. A 3month period is also refered to as a season. 

We compute the total precipitation during these periods. 
Moreover, we transform the observed precipitation to correspond with the format that precipitation **forecasts** are published in, i.e. terciles.   
We use this information for the analysis of dry spells, more specifically to see if, given perfect forecasting skill, there is information in the forecasted quantities for forecasting dryspells.   


We prefer to use as many readily available products as we can. However, we could only find historical monthly terciles (using [CAMS-OPI](https://iridl.ldeo.columbia.edu/maproom/Global/Precipitation/Percentiles.html?bbox=bb%3A-20%3A-40%3A55%3A40%3Abb&T=Dec%202020) data) and not historical seasonal terciles. We have therefore computed these seasonal terciles ourselves. As source we use CHIRPS data as this is also the source used to compute the dry spells, we have experience working with this source, it has a high resolution, and generally speaking shows good performance. 
<!-- #endregion -->

### set general variables and functions

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import xarray as xr
import cftime
import rioxarray
from shapely.geometry import mapping
import seaborn as sns
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp
from src.utils_general.plotting import plot_raster_boundaries_clip
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(public_data_dir,config.PROCESSED_DIR,country_iso3)

chirps_glb_dir=os.path.join(public_data_dir,config.RAW_DIR,config.GLOBAL_ISO3,"chirps")
chirps_mwi_dir=os.path.join(country_data_processed_dir,"chirps")
chirps_glb_monthly_path=os.path.join(chirps_glb_dir,"chirps_global_monthly.nc")
chirps_monthly_mwi_path=os.path.join(chirps_mwi_dir,"chirps_mwi_monthly.nc")
```

```python
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```python
#date to make plots for to test values. To be sure this is consistent across the different plots
test_date=cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0, 0)
test_date_dtime="2020-1-1"
```

### Load the data
The CHIRPS data is provided as monthly sum per 0.05x0.05 degree raster cell. This is global data, and we clip it to Malawi.  

```python
# # Only need to run if you want to update the data
# # The file is around 7GB so takes a while
# url_chirpsmonthly="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
# download_ftp(url_chirpsmonthly,chirps_glb_monthly_path)
# # clipping to MWI can take some time, so clip and save to a new file
# ds=xr.open_dataset(chirps_glb_monthly_path)
# df_bound = gpd.read_file(adm1_bound_path)
# ds_clip = ds.rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
# ds_clip.to_netcdf(chirps_monthly_mwi_path)
```

```python
#don't fully understand, but need masked=True to read nan values as nan correctly
ds_clip=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
```

```python
ds_clip=ds_clip.rename({"x":"lon","y":"lat"})
```

```python
#this loads the dataset completely
#default is to do so called "lazy loading", which causes aggregations on a larger dataset to be very slow. See http://xarray.pydata.org/en/stable/io.html#netcdf
#however, this .load() takes a bit of time, about half an hour
ds_clip.load()
```

```python
#show the data for each month of 2020, clipped to MWI
g=ds_clip.sel(time=ds_clip.time.dt.year.isin([2020])).precip.plot(
    col="time",
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    cmap="YlOrRd",
)

df_bound = gpd.read_file(adm2_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")
```

```python
ds_clip
```

### Compute total precipitation
Compute the total precipitation per month and season (=3month period). First the precipitation is computed at raster cell level, whereafter this is aggregated to admin level. Several methods of aggregation are implemented, such as the mean, the max, and taking different percentiles

```python
#compute the rolling sum over three month period. Rolling sum works backwards, i.e. value for month 3 is sum of month 1 till 3
seas_len=3
ds_season=ds_clip.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")
```

```python
ds_season
```

```python
def alldates_statistics_total(ds,raster_transform,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    for date in ds.time.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(time=date)
        
        # compute the percentage of the admin area that has below average precipitation
        #set all values with below average precipitation to 1 and others to 0
        forecast_binary = np.where(ds_date.precip.values<=50, 1, 0)
        #compute number of cells in admin region (sum) and number of cells in admin region with below average precipitation (count)
        bin_zonal = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
        
        df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan))["max"]
        df["mean_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan))["mean"]
        df["min_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan))["min"]
        percentile_list = [10,20,30,40,50,60,70,80]
        zonal_stats_percentile_dict=zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan, stats=" ".join([f"percentile_{str(p)}" for p in percentile_list]))[0]
        for p in percentile_list:
            df[[f"percentile_{str(p)}" for p in percentile_list]]=pd.DataFrame(zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan, stats=" ".join([f"percentile_{str(p)}" for p in percentile_list])))[[f"percentile_{str(p)}" for p in percentile_list]]
        
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
  
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")
    
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    df_hist['date_month']=df_hist.date.dt.to_period("M")
        
    return df_hist
```

```python
adm_levels=[2]
percentile_list = [10,20,30,40,50,60,70,80]
```

```python
#compute stats per admin
for a in adm_levels:
    if a==1:
        adm_bound_path=adm1_bound_path
    elif a==2:
        adm_bound_path=adm2_bound_path
    else: 
        print("not a valid admin level")
    #stats on monthly precipitation
    clip_transform=ds_clip.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform()
    #selecting years 2000-2020 since these are the years we have dry spell data. Thus thereby limiting computing time to relevant years
    df_month_total=alldates_statistics_total(ds_clip.sel(time=ds_clip.time.dt.year.isin(range(2000,2021))),clip_transform,adm_bound_path)
    # df_month_total.drop("geometry",axis=1).to_csv(os.path.join(chirps_mwi_dir,f"chirps_monthly_total_precipitation_admin{a}.csv"),index=False)
    #stats on seasonal precipitation
    df_season_total=alldates_statistics_total(ds_season.sel(time=ds_season.time.dt.year.isin(range(2000,2021))),clip_transform,adm_bound_path)
#     df_season_total.drop("geometry",axis=1).to_csv(os.path.join(chirps_mwi_dir,f"chirps_seasonal_total_precipitation_admin{a}.csv"),index=False)
```

### Below average tercile
    
Besides the total precipitaiton, we would like to know the occurrence of the lower tercile, since many forecasts are generated in this format.  
We do this at raster cell level, and thereafter aggregate to admin level. 

To compute the occurrence of below average per cell we have to perform several steps. First we compute the total precipitation per 3 month period. We then compute the climatological lower tercile where we use the period 1981-2010 to define the climatology. This period is chosen as it is also the period IRI uses for their seasonal forecasts. 

Thereafter, we use the climatology to get a yes/no if whether any cell within an admin2 region had below average precipitation. 

We use below-average and lower tercile interchangily in the comments but they refer to the same

Caveats: 
- One could argue to use a more recent climatological period, this is e.g. done by ECMWF
- No dry mask has been included in the computations. IRI does compute a dry mask for their forecasts, see more information [here]()

Future improvements:
- Change value used for indicating normal/above average cells (now set to -999 but tricky cause software can interpret that as "no data")
- Implement a faster method to compute regional statistics
- Discuss more the aggregation method from cell to admin. Now several stats are computed, but 50% of the area having below average was chosen as threshold to identify the whole adming as below average

```python
#plot the three month sum for NDJ 2019/2020
#dense spot seems to be a nature reserve that also looks more green on satellite imagery (Google Maps)
fig=plot_raster_boundaries_clip([ds_season.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))
```

```python
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
g=sns.displot(ds_season.sel(time=ds_season.time.dt.month==1).precip.values.flatten(),kde=True,aspect=2,color="#CCCCCC")
plt.axvline(2,color="#18998F",label="dry spell")
perc=np.percentile(ds_season.sel(time=ds_season.time.dt.month==1).precip.values.flatten()[~np.isnan(ds_season.sel(time=ds_season.time.dt.month==1).precip.values.flatten())], 33)
plt.axvline(perc,color="#C25048",label="below average")
plt.legend()
g.set(xlabel="Seasonal precipitation (mm)")
plt.title("Distribution of seasonal precipitation in DJF from 2000-2020")
```

```python
#define the years that are used to define the climatology. We use 1981-2010 since this is also the period used by IRI's seasonal forecasts
ds_season_climate=ds_season.sel(time=ds_season.time.dt.year.isin(range(1981,2011)))
```

```python
ds_season_climate
```

```python
#compute the thresholds for the lower tercile, i.e. below average, per season
#since we computed a rolling sum, each month represents a season
ds_season_climate_quantile=ds_season_climate.groupby(ds_season_climate.time.dt.month).quantile(0.33)
```

```python
ds_season_climate_quantile
```

```python
#plot the below average boundaries for the NDJ season
fig=plot_raster_boundaries_clip([ds_season_climate_quantile.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(month=1)],adm1_bound_path,colp_num=1,forec_val="precip",predef_bins=np.linspace(0,1000,10),cmap="YlOrRd")
```

```python
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
sns.distplot(ds_season_climate.sel(time=ds_season_climate.time.dt.month==1).precip.values.flatten())
```

```python
sns.distplot(ds_season_climate_quantile.sel(month=1).precip.values.flatten())
```

```python
#determine the raster cells that have below-average precipitation, other cells are set to -999
list_ds_seass=[]
for s in np.unique(ds_season.time.dt.month):
    ds_seas_sel=ds_season.sel(time=ds_season.time.dt.month==s)
    #keep original values of cells that are either nan or have below average precipitation, all others are set to -999
    ds_seas_below=ds_seas_sel.where((ds_seas_sel.precip.isnull())|(ds_seas_sel.precip<=ds_season_climate_quantile.sel(month=s).precip),-999)#,drop=True)
    list_ds_seass.append(ds_seas_below)
ds_season_below=xr.concat(list_ds_seass,dim="time")        
```

```python
ds_season_below
```

```python
# #can be used to inspect the values that are below average
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_season_below.sel(time=test_date)["precip"].values.flatten()[~np.isnan(ds_season_below.sel(time=test_date)["precip"].values.flatten())]))
```

```python
#fraction of values with below average value. Should be around 0.33
ds_season_below_notnan=ds_season_below.precip.values[~np.isnan(ds_season_below.precip.values)]
np.count_nonzero(ds_season_below_notnan!=-999)/np.count_nonzero(ds_season_below_notnan)
```

```python
#plot the cells that had below-average precipitation in NDJ of 2019/2020
#there is goegraphical autocorrelation, i.e. cells close to each other share similair values, this is as it is supposed to be
fig=plot_raster_boundaries_clip([ds_season_below.where(ds_season_below.precip!=-999).rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))
```

```python
def alldates_statistics(ds,raster_transform,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    for date in ds.time.values:
        df=gpd.read_file(adm_path)
        ds_date=ds.sel(time=date)
        
        # compute the percentage of the admin area that has below average precipitation
        #set all values with below average precipitation to 1 and others to 0
        forecast_binary = np.where(ds_date.precip.values!=-999, 1, 0)
        #compute number of cells in admin region (sum) and number of cells in admin region with below average precipitation (count)
        bin_zonal = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
        
        #same but then also including cells that only touch the admin region, i.e. don't have their cell within that region
        bin_zonal_touched = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, all_touched=True, stats=['count', 'sum'],nodata=np.nan))
        df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100
        
        #return the value of the cell with the highest value within the admin region. 
        #In our case if this isn't -999 it indicates that at least one cell has below average precipitation
        df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, all_touched=True,nodata=np.nan))["max"]
        df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds_date["precip"].values, affine=raster_transform, nodata=np.nan))["max"]
        
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
  
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")
    #all admins that are not -999, have at least one cell with below average precipitation
    df_hist[f"below_average_touched"]=np.where(df_hist["max_cell_touched"]!=-999,1,0)
    df_hist[f"below_average_max"]=np.where(df_hist["max_cell"]!=-999,1,0)
        
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    df_hist['date_month']=df_hist.date.dt.to_period("M")
        
    return df_hist
```

```python
#compute whether the maximum cell that touches an admin2 region has below average precipitation for each month since 2010
#have to write the crs for the transform to be correct!
df_belowavg_seas=alldates_statistics(ds_season_below.sel(time=ds_season_below.time.dt.year.isin(range(2000,2021))),ds_season_below.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform(),adm2_bound_path)
```

```python
# #quality check
# #check that all max touched values in df_belowavg_seas are also present in ds_season_below
# #this is to make sure all goes correctly with zonal_stats regarding crs's since it has happened that things go wrong there without insight why
# display(df_belowavg_seas[df_belowavg_seas.date==test_date_dtime])
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_season_below.sel(time=test_date)["precip"].values.flatten()[~np.isnan(ds_season_below.sel(time=test_date)["precip"].values.flatten())]))
```

```python
#percentage of area with below average rain
#TODO: decide which threshold to use
#could argue for 50%, i.e. majority
#could also argue for the threshold which results in around 1/3 of the occurences being below average, which is the definition of terciles. 
#in this case about the same but second method would change according to the data
#Don't know if large difference between adm regions
#Note: only including cells that have their centre within the region, do think this makes sense especially with the high resolution we got
for i in np.linspace(0,100,11):
    print(f"Fraction of adm2-date combinations where >={i}% of area received bel. avg. rain:","{:.2f}".format(len(df_belowavg_seas[df_belowavg_seas["perc_threshold"]>=i])/len(df_belowavg_seas)))
```

```python
#THIS is super interesting, that the area recieving below average rain clearly changed for 2010-2020 compared to 2000-2020
#Though would expect it to be the other way around, that more often areas recieve below average rain since 2010.. 
#percentage of area with below average rain
#TODO: decide which threshold to use
#could argue for 50%, i.e. majority
#could also argue for 30% as this results in around 1/3 of the occurences the area being below average, which is the definition of terciles. 
#Don't know if large difference between adm regions
#Note: only including cells that have their centre within the region, do think this makes sense especially with the high resolution we got
for i in np.linspace(0,100,11):
    print(f"Fraction of adm2-date combinations where >={i}% of area received bel. avg. rain 2010-2020:","{:.2f}".format(len(df_belowavg_seas[(df_belowavg_seas.date.dt.year>=2010)&(df_belowavg_seas["perc_threshold"]>=i)])/len(df_belowavg_seas[(df_belowavg_seas.date.dt.year>=2010)])))
```

```python
#This is when assigning the whole adm2 region as below average when any cell touching has below average precipitation 
#--> logical that it is a larger fraction than the climatological 0.33
print("fraction of adm2s with at least one cell with below average rain:",len(df_belowavg_seas[df_belowavg_seas["below_average_max"]==1])/len(df_belowavg_seas))
```

```python
#for now using 0.50 of the area as a threshold
df_belowavg_seas["below_average"]=np.where(df_belowavg_seas["perc_threshold"]>=50,1,0)
```

```python
#Hmm surprised that there is such a large difference across months.. Would want them to all be around 0.33?
#compute fraction of values having below average per month
df_bavg_month=df_belowavg_seas.groupby(df_belowavg_seas.date.dt.month).sum()
df_bavg_month['fract_below']=df_bavg_month["below_average"]/df_belowavg_seas.groupby(df_belowavg_seas.date.dt.month).count()["below_average"]
df_bavg_month[["fract_below"]]
```

```python
#compute fraction of values having below average per year
df_bavg_year=df_belowavg_seas.groupby(df_belowavg_seas.date.dt.year).sum()
df_bavg_year['fract_below']=df_bavg_year["below_average"]/df_belowavg_seas.groupby(df_belowavg_seas.date.dt.year).count()["below_average"]
df_bavg_year[["fract_below"]].head()
```

```python
#seems the deviation from 0.33 is not crazy large for any adm2 and not clear dependency on the size of the adm2
#compute fraction of values having below average per ADMIN2
df_bavg_adm2=df_belowavg_seas.groupby("ADM2_EN").sum()
df_bavg_adm2["fract_below"]=df_bavg_adm2["below_average"]/df_belowavg_seas.groupby("ADM2_EN").count()["below_average"]
df_bavg_adm2[["fract_below","Shape_Area"]]
```

```python
# #save to file
# #geometry takes long time to save, so remove that
# df_belowavg_seas.drop("geometry",axis=1).to_csv(os.path.join(chirps_mwi_dir,"chirps_seasonal_below_average_precipitation.csv"),index=False)
```

### CHIRPS monthly

```python
#ds_clip already contains the monthly data, so don't need further preprocessing for that
#define the years that are used to define the climatology. We use 1981-2010 since this is also the period used by IRI's seasonal forecasts
ds_month_climate=ds_clip.sel(time=ds_clip.time.dt.year.isin(range(1981,2011)))
```

```python
#compute the thresholds for the lower tercile, i.e. below average, per season
#since we computed a rolling sum, each month represents a season
ds_month_climate_quantile=ds_month_climate.groupby(ds_month_climate.time.dt.month).quantile(0.33,skipna=True)
```

```python
#plot all values and tercile boundaries to get a feeling for the general distribution and if the tercile boundaries make sense
sns.distplot(ds_month_climate.sel(time=ds_month_climate.time.dt.month==1).precip.values.flatten())
```

```python
sns.distplot(ds_month_climate_quantile.sel(month=1).precip.values.flatten())
```

```python
ds_month_climate_quantile
```

```python
#plot the below average boundaries for the NDJ season
fig=plot_raster_boundaries_clip([ds_month_climate_quantile.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(month=1)],adm1_bound_path,colp_num=1,forec_val="precip",predef_bins=np.linspace(0,1000,10),cmap="YlOrRd")
```

```python
#determine the raster cells that have below-average precipitation, other cells are set to -999
list_ds_months=[]
for s in np.unique(ds_clip.time.dt.month):
    ds_month_sel=ds_clip.sel(time=ds_clip.time.dt.month==s)
    #drop removes the dates with all nan values, i.e. no below average, but we dont want that
    ds_month_below=ds_month_sel.where((ds_month_sel.precip.isnull()) | (ds_month_sel.precip<=ds_month_climate_quantile.sel(month=s).precip) ,-999)
    list_ds_months.append(ds_month_below)
ds_month_below=xr.concat(list_ds_months,dim="time")        
```

```python
# #can be used to inspect the values that are below average
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_month_below.sel(time=test_month_date)["precip"].values.flatten()[~np.isnan(ds_month_below.sel(time=test_month_date)["precip"].values.flatten())]))
```

```python
ds_month_below_notnan=ds_month_below.precip.values[~np.isnan(ds_month_below.precip.values)]
np.count_nonzero(ds_month_below_notnan!=-999)/np.count_nonzero(ds_month_below_notnan)
```

```python
#not correct anymore cause using -999
#plot the cells that had below-average precipitation in DJF of 2019/2020
fig=plot_raster_boundaries_clip([ds_month_below.where(ds_month_below.precip!=-999).rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").sel(time=test_date)],adm1_bound_path,colp_num=1,forec_val="precip",cmap="YlOrRd",predef_bins=np.linspace(0,1000,10))
```

```python
#compute whether the maximum cell that touches an admin2 region has below average precipitation for each month since 2010
df_belowavg_month=alldates_statistics(ds_month_below.sel(time=ds_month_below.time.dt.year.isin(range(2000,2021))),ds_month_below.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.transform(),adm2_bound_path)
```

```python
# #quality check
# #check that all max touched values in df_belowavg_month are also present in ds_month_below
# #this is to make sure all goes correctly with zonal_stats regarding crs's since it has happened that things go wrong there without insight why
# display(df_belowavg_month[df_belowavg_month.date==test_date_dtime])
# with np.printoptions(threshold=np.inf):
#     print(np.unique(ds_month_below.sel(time=test_month_date)["precip"].values.flatten()[~np.isnan(ds_month_below.sel(time=test_month_date)["precip"].values.flatten())]))
```

```python
#percentage of area with below average rain
#TODO: decide which threshold to use
#could argue for 50%, i.e. majority
#could also argue for the threshold which results in around 1/3 of the occurences being below average, which is the definition of terciles. 
#in this case about the same but second method would change according to the data
#Don't know if large difference between adm regions
#Note: only including cells that have their centre within the region, do think this makes sense especially with the high resolution we got
for i in np.linspace(0,100,11):
    print(f"Fraction of adm2-date combinations where >={i}% of area received bel. avg. rain:","{:.2f}".format(len(df_belowavg_month[df_belowavg_month["perc_threshold"]>=i])/len(df_belowavg_month)))
```

```python
#This is when assigning the whole adm2 region as below average when any cell touching has below average precipitation 
#--> logical that it is a larger fraction than the climatological 0.33
print("fraction of adm2s with at least one cell with below average rain:",len(df_belowavg_month[df_belowavg_month["below_average_max"]==1])/len(df_belowavg_month))
```

```python
#for now using 0.50 as threshold
df_belowavg_month["below_average"]=np.where(df_belowavg_month["perc_threshold"]>=50,1,0)
```

```python
#Hmm strange that there is such a large difference across months.. Should in theory all be 0.33?
#compute fraction of values having below average per month
df_bavg_month=df_belowavg_month.groupby(df_belowavg_month.date.dt.month).sum()
df_bavg_month['fract_below']=df_bavg_month["below_average"]/df_belowavg_month.groupby(df_belowavg_month.date.dt.month).count()["below_average"]
df_bavg_month[["fract_below"]]
```

```python
#Hmm would have expected more deviation per year, e.g. I thought 2011 was very little precipitation year?
#compute fraction of values having below average per month
df_bavg_year=df_belowavg_month.groupby(df_belowavg_month.date.dt.year).sum()
df_bavg_year['fract_below']=df_bavg_year["below_average"]/df_belowavg_month.groupby(df_belowavg_month.date.dt.year).count()["below_average"]
df_bavg_year[["fract_below"]]
```

```python
#compute fraction of values having below average per ADMIN2
df_bavg_adm2=df_belowavg_month.groupby("ADM2_EN").sum()
df_bavg_adm2["fract_below"]=df_bavg_adm2["below_average"]/df_belowavg_month.groupby("ADM2_EN").count()["below_average"]
df_bavg_adm2[["fract_below"]]
```

```python
# #save to file
# #geometry takes long time to save, so remove that
# df_belowavg_month.drop("geometry",axis=1).to_csv(chirps_mwi_dir,"chirps_monthly_below_average_precipitation.csv"),index=False)
```
