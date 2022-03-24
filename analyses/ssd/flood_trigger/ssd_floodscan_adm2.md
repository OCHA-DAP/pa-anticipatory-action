### Floodscan

This notebook inspects the Floodscan data for South Sudan. 
A specific focus is on a subset of the counties in the states of Jonglei and Unity. 
As these experience flooding and might be the focus of the pilot

```python
import os
from pathlib import Path
import sys
from datetime import timedelta
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.testing import assert_frame_equal
import geopandas as gpd
import hvplot.xarray
import matplotlib as mpl
import numpy as np
from scipy import stats
from functools import reduce
import altair as alt
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.utils_general.statistics import get_return_periods_dataframe


# mpl.rcParams['figure.dpi'] = 300
```

```python
iso3="ssd"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
country_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
country_data_public_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
adm2_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin2_shp"]
```

```python
#admin2's of interest for now
#all located in Unity and Jonglei
adm2_list=['Panyijiar', 'Leer', 'Mayendit', 'Koch', 'Guit',
           'Fangak', 'Ayod', 'Duk', 'Twic East', 'Bor South',
          'Yirol East','Awerial'
          ]
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
gdf_adm2['include']=np.where(gdf_adm2["ADM2_EN"].isin(adm2_list),True,False)
adms=alt.Chart(gdf_adm2).mark_geoshape(stroke="black").encode(
    color=alt.Color("include",scale=alt.Scale(range=["grey","red"])),
    tooltip=["ADM2_EN"]
).properties(width=300,height=200,title="Admins of focus and SSD rivers")
gdf_rivers=gpd.read_file(country_data_public_exploration_dir/"rivers"/"ssd_main_rivers_fao_250k"/"ssd_main_rivers_fao_250k.shp")
rivers=alt.Chart(gdf_rivers).mark_geoshape(stroke="blue",filled=False).encode(tooltip=['CLASS'])
adms+rivers
```

```python
# #clip to country
# #this takes long, so only do when not saved the file yet
# #read floodscan data
# fs = floodscan.Floodscan()
# fs_raw = fs.read_raw_dataset()
# fs_clip = (fs_raw.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
#            .rio.write_crs("EPSG:4326")
#            .rio.clip(gdf_adm2["geometry"], all_touched=True))
```

```python
# #somehow cannot save the file with these attributes
# fs_clip.SFED_AREA.attrs.pop('grid_mapping')
# fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
# fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
# fs_clip.to_netcdf(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
```

```python
fs_clip=xr.load_dataset(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
#I dont fully understand why, these grid mappings re-occur and what they mean
#but if having them, later on getting crs problems when computing stats
fs_clip.SFED_AREA.attrs.pop('grid_mapping')
fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
fs_clip=fs_clip.rio.write_crs("EPSG:4326",inplace=True)
```

We plot the data. We can see that the resolution is high (300 arcseconds = 0.083 degrees). We therefore decide that when analyzing the data at the admin2 level, it is fine to only look at the cells with their centre within the admin2. 

```python
g=fs_clip.sel(time='2020-08-03T00:00:00.000000000').SFED_AREA.plot()
gdf_adm2.boundary.plot(ax=g.axes);
```

there are three different variables in the data: SFED_AREA, NDT_SFED_AREA, and LWMASK_AREA. 
SFED_AREA is the data we use. NDT_SFED_AREA is the same data but with no threshold, i.e. more senstitive. LWMASK_AREA is a mask with static water bodies. This is already not included in SFED_AREA. See for more information on the data types `floodscan.py` or the [user guide](https://www.aer.com/contentassets/22663ebfdb7c467599f363466b32770f/floodscan_data_users_guide_v05r01_r01.pdf)

```python
da_clip=fs_clip.SFED_AREA
```

We plot the values across the country. We focus on the rainy season, which is approximately from July till October with peak flooding months in August and September. 

From the plot below we can see that the fraction of the area that is flooded (SFED_AREA) is generally very low. 
Note that from the first graph you cannot see the higher values well. The second graphs only shows values larger than 0.01. From this we can see that a large number of values is <0.01 and higher values occur though significantly less often.

```python
fs_clip_rainy=fs_clip.sel(time=fs_clip.time.dt.month.isin(range(7,11)))
```

```python tags=[]
fs_clip_rainy.hvplot.hist('SFED_AREA',alpha=0.5)
```

```python
fs_clip_rainy.where(fs_clip_rainy>0.01).hvplot.hist('SFED_AREA',alpha=0.5)
```

When only focusing on the states of Jonglei and Unity, we can see that on average they show slightly higher values, though the majority of the values is still below 0.2

```python
(fs_clip_rainy.where(fs_clip_rainy>0.01).rio.write_crs("EPSG:4326").rio.clip(
    gdf_adm2.loc[gdf_adm2.ADM1_EN.isin(['Unity','Jonglei']),'geometry'])
.hvplot.hist('SFED_AREA',alpha=0.5))
```

### Compute stats per county
Now that we have analyzed the raster data, we can aggregate this to get the statistics per county. 

```python tags=[]
# #takes a bit of time so only compute when not saved yet
# df_floodscan=compute_raster_statistics(
#         gdf=gdf_adm2,
#         bound_col='ADM2_PCODE',
#         raster_array=da_clip,
#         lon_coord="lon",
#         lat_coord="lat",
#         stats_list=["median","mean","max"],
#         #computes value where 20% of the area is above that value
#         percentile_list=[80],
#         #Decided to only use centres, but can change that
#         all_touched=False,
#     )
# df_floodscan=df_floodscan.merge(gdf_adm2[["ADM2_EN","ADM2_PCODE"]],on="ADM2_PCODE",how="left")
# # df_floodscan.to_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm2_stats.csv',index=False)
```

```python
df_floodscan=pd.read_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm2_stats.csv',parse_dates=['time'])
```

With these statistics we can plot the timeseries per admin2. 

We plot these for the mean as well as for the median. We can see very different patterns, due to many cells having values around zero. I am not sure which method of aggregation makes most sense here.. I would incline towards the median though you then see that this drops very quickly again which might "downgrade" the presence of a flood.

For both variables however, we can see that the patterns per county differ heavily. Both in absolute as in relative numbers. We do see clear peaks during some years, which is a good indication that this data and the aggregation methodology might at least be able to pick up fluctuations in the normal situation. 

```python tags=[]
def plot_floodscan_timeseries(df,var_col,adm_list,adm_col="ADM2_EN"):
    for adm in adm_list:
        fig, ax = plt.subplots()
        df_floodscan_sel = df[df[adm_col]==adm].copy()
        df_floodscan_sel.loc[:,'mean_cell_rolling'] = (df_floodscan_sel
                                                       .loc[:,var_col]
                                                       .transform(lambda x: x.rolling(5, 1).mean())
                                                      )
        sns.lineplot(data=df_floodscan_sel, x="time", y=var_col, lw=0.25, label='Original')
        sns.lineplot(data=df_floodscan_sel, x="time", 
                     y="mean_cell_rolling", lw=0.25, label='5-day moving\navg')   
        ax.set_ylabel('Mean flooded fraction')
        ax.set_xlabel('Date')
        ax.set_title(f'Flooding in {adm}, 1998-2020, aggregation {var_col}')
        ax.legend()
```

```python
plot_floodscan_timeseries(df_floodscan,"mean_ADM2_PCODE",adm2_list)
```

```python
# plot_floodscan_timeseries(df_floodscan,"median_ADM2_PCODE",adm2_list)
```

Next we can identify consecutive dates of significantly above average (>3 standard deviations) surface water coverage. We'll consider these to be flood events. This threshold is set with the intent to capture events that are significant outliers, but could be refined/validated with future work.

We only consider the rainy season (Jul-Oct) here

```python
df_floodscan['month'] = pd.DatetimeIndex(df_floodscan['time']).month
df_floodscan_rainy = df_floodscan.loc[(df_floodscan['month'] >= 7) | (df_floodscan['month'] <= 10)]
```

```python
#this is copied from amazing Hannah's mwi flood work
# Assign an eventID to each flood 
# ie. consecutive dates in a dataframe filtered to keep only outliers in flood fraction
def get_groups_consec_dates(df):
    dt = df['time']
    day = pd.Timedelta('1d')
    breaks = dt.diff() != day
    groups = breaks.cumsum()
    groups = groups.reset_index()
    groups.columns = ['index', 'eventID']
    df_out = df.merge(groups, left_index=True, right_on='index')
    return df_out

# Get basic summary statistics for each flood event
def get_flood_summary(df):
    s1 = df.groupby('eventID')['time'].min().reset_index().rename(columns={'time': 'start_date'})
    s2 = df.groupby('eventID')['time'].max().reset_index().rename(columns={'time': 'end_date'})
    s3 = df.groupby('eventID')['time'].count().reset_index().rename(columns={'time': 'num_days'})
    s4 = df.groupby('eventID')['mean_cell_rolling'].max().reset_index().rename(columns={'mean_cell_rolling': 'max_flood_frac'})
    dfs = [s1, s2, s3, s4]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['eventID'],
                                            how='outer'), dfs)
    return df_merged

# Merge overlapping flood events
# Each row in the input df should be an event
# With start and end date columns: ['start_date'] and ['end_date']
def merge_events(df):
    df['flood_id'] = 0
    f_id = 1
    
    # Loop through all of the events and tag the ones that are part of an overlap
    for i in range(1, len(df.index)):        
        start = df['start_date'].iloc[i,]
        end = df['end_date'].iloc[i-1,]
        if start < end:
            df.loc[i, 'flood_id'] = f_id
            df.loc[i-1, 'flood_id'] = f_id
        else:           
            df.loc[i-1, 'flood_id'] = f_id
            f_id += 1
    
    # Now for each event, extract the min start data and max end date
    df_start = df.groupby('flood_id')['start_date'].min().to_frame().reset_index()
    df_end = df.groupby('flood_id')['end_date'].max().to_frame().reset_index()
    
    df_events = df_start.merge(df_end, on='flood_id').sort_values(by='start_date')
    return df_events
```

```python
def compute_flood_events(df_rainy,var_col,adm_list,outlier_thresh=3,adm_col='ADM2_EN',save_data=False):
    list_floods_all=[]
    for adm in adm_list:
        df_floodscan_sel = df_rainy[df_rainy[adm_col]==adm].copy()
        df_floodscan_sel['mean_cell_rolling'] = df_floodscan_sel[var_col].transform(lambda x: x.rolling(5, 1).mean())
        df_floods_summary = df_floodscan_sel[(np.abs(stats.zscore(df_floodscan_sel['mean_cell_rolling'])) >= outlier_thresh)]
        df_floods_summary = get_groups_consec_dates(df_floods_summary)
        df_floods_summary = get_flood_summary(df_floods_summary)

        df_summary_clean = merge_events(df_floods_summary)

        if save_data: 
            df_summary_clean.to_csv(country_data_exploration_dir / f'{adm}_floodscan_event_summary.csv', index=False)   
        df_summary_clean[adm_col]=adm
        list_floods_all.append(df_summary_clean)
        df_floods_all=pd.concat(list_floods_all)
        df_floods_all['year']=df_floods_all.start_date.dt.year
    return df_floods_all
```

```python
# df_floods_all_median=compute_flood_events(df_floodscan_rainy,"median_ADM2_PCODE",adm2_list)
```

```python
df_floods_all_mean=compute_flood_events(df_floodscan_rainy,"mean_ADM2_PCODE",adm2_list)
```

From the table and plot below we can see very different numbers of "flood events" per county. Since we define a flood event based on standard deviation, this means that in some counties there is a higher fluctuation than in others. I am not 100% sure whether this is a fair method to compare the different counties. 

We also see that the different counties see "flood events" during different years. Where 2020, 2014, and 1998 saw the most counties with a flood event

```python
df_year=df_floods_all_mean[['ADM2_EN','year']].drop_duplicates()
```

```python
df_year.groupby("ADM2_EN").count()
```

```python
df_year.groupby("year").count().sort_values('ADM2_EN',ascending=False)
```

```python
#add missing years
df_year['flood_event']=True
df_year.set_index(['ADM2_EN', 'year'], inplace=True)
index = pd.MultiIndex.from_product(df_year.index.levels)
df_year=df_year.reindex(index).reset_index()
#plot years with flood per county
alt.Chart(df_year).mark_rect().encode(
    x="year:N",
    y="ADM2_EN:N",
     color=alt.Color('flood_event:N',scale=alt.Scale(range=["#D3D3D3",'red']))
).properties(
    title="Flood detected per county"
)
```

### Return period
Instead of looking at the standard deviation, we can also look at the return period. We do this by looking at the peak value for each year and county. We compute the return period separately per county. 

```python
df_floodscan_rolling=df_floodscan.sort_values('time').set_index('time').groupby('ADM2_PCODE',as_index=False).mean_ADM2_PCODE.rolling(10,min_periods=10).mean().reset_index().rename(columns={'mean_ADM2_PCODE':'mean_rolling'})
```

```python
df_floodscan=df_floodscan.merge(df_floodscan_rolling,on=['time','ADM2_PCODE'])
```

```python
df_floodscan['year']=df_floodscan.time.dt.year
```

```python
#get one row per adm2-year combination that saw the highest mean value
df_floodscan_peak=df_floodscan.sort_values('mean_rolling', ascending=False).drop_duplicates(['year','ADM2_PCODE'])
```

```python
years = np.arange(1.5, 10, 0.5)
```

```python
df_floodscan_rp=df_floodscan_peak.copy()
```

For now using the empirical return period, cause then it has the same number of identified years per county. But should possibly switch to analytical

```python
for adm in adm2_list:
    df_adm=df_floodscan_peak[df_floodscan_peak.ADM2_EN==adm].copy()
    df_rps_ana=get_return_periods_dataframe(df_adm, rp_var="mean_rolling",years=years, method="analytical",round_rp=False)
    df_rps_emp=get_return_periods_dataframe(df_adm, rp_var="mean_rolling",years=years, method="empirical",round_rp=False)
    df_floodscan_rp.loc[df_floodscan_rp.ADM2_EN==adm,'rp5']=np.where((df_adm.mean_rolling>=df_rps_emp.loc[5,'rp']),1,0)
```

```python
#plot years with flood per county
alt.Chart(df_floodscan_rp[df_floodscan_rp.ADM2_EN.isin(adm2_list)]).mark_rect().encode(
    x="year:N",
    y="ADM2_EN:N",
     color=alt.Color('rp5:N',scale=alt.Scale(range=["#D3D3D3",'red']))
).properties(
    title="1 in 5 year return period peak"
)
```

```python
df_floodscan_peak[df_floodscan_peak.mean_ADM0_PCODE>=df_rps_ana.loc[5,'rp']].sort_values('year')
```

### Peaks per year
We are interested to know whether flooding in certain counties always occurs earlier than in other counties. This could be interesting information to use for the trigger. A hypthosis is that counties South see flooding earlier since the Nile is flowing from South to North. 

We look at the rank of peaking of flooding in each year for each admin, and only during the years that we identified as the flooding being exceptional

```python
#get one row per adm2-year combination that saw the highest mean value
df_floodscan['year']=df_floodscan.time.dt.year
df_floodscan_peak=df_floodscan.sort_values('mean_ADM2_PCODE', ascending=False).drop_duplicates(['year','ADM2_EN'])
```

```python
df_floodscan_peak_sel=df_floodscan_peak[df_floodscan_peak.ADM2_EN.isin(adm2_list)]
```

It is a little hard to visualize but from the plots below we can see there is not a crystal clear pattern in which admins reach their peak first/last. 
However, if you want to you can see a bit of pattern. E.g. Ayod is relatively often late while Bor South is relatively often early. 

I am doubting if peak flooding is the best measure though.. 
As we can see from the table below, they can heavily differ in time, i.e. not being part of the same possible flood. 

```python
alt.Chart(df_floodscan_peak_sel).mark_line(point=True).encode(
    x="year:O", y="rank:O", color=alt.Color("ADM2_EN:N")
).transform_window(
    rank="rank()",
    sort=[
        alt.SortField("time", order="ascending"),
    ],
    groupby=["year"],
).properties(title="Ranking of occurrence of peak flooding per county and year",
            width=600,height=300)
```

```python
alt.Chart(df_floodscan_peak_sel).mark_rect().encode(
    x="year:O", y="ADM2_EN", color=alt.Color("rank:O",scale=alt.Scale(scheme="goldred"))
).transform_window(
    rank="rank()",
    sort=[
        alt.SortField("time", order="ascending"),
    ],
    groupby=["year"],
).properties(title="Ranking of occurrence of peak flooding per county and year",
            width=600)
```

```python
#peak floodings differ by 5 months --> possibly not part of the same flood? 
df_floodscan_peak_sel[df_floodscan_peak_sel.year==1998].sort_values('time')
```

We repeat the exercise but only including the years that we identified as exceptional flood levels, to filter out the noise. 
From here we can still not see a super clear pattern. However, for example Panyijar reaches often relatively late, while Bor South is relatively early. But for example in Koch, there is no consistent pattern. 

```python
df_floods_all_mean_first=df_floods_all_mean.sort_values('start_date', ascending=False).drop_duplicates(['year','ADM2_EN'])
alt.Chart(df_floods_all_mean_first).mark_rect().encode(
    x="year:O", y="ADM2_EN", color=alt.Color("rank:O",scale=alt.Scale(scheme="goldred"))
).transform_window(
    rank="rank()",
    sort=[
        alt.SortField("start_date", order="ascending"),
    ],
    groupby=["year"],
).properties(title="Ranking of occurrence of peak flooding per county and year",
            width=600)
```

### Bonus: Compare computation methods
We computed the statistics in this notebook with our newer `compute_raster_statistics` function. In the `floodscan.py` script we use an older method. We compare the results of the two methods and see they are equal. We thus prefer to use the newer method

```python
df_floodscan_old=pd.read_csv("/Volumes/GoogleDrive/Shared drives/Predictive Analytics/CERF Anticipatory Action/General - All AA projects/Data/private/processed/SSD/floodscan/ssd_floodscan_stats_adm2.csv",
                             index_col=None,parse_dates=['date'])
df_floodscan_old=df_floodscan_old.merge(gdf_adm2[['ADM2_EN',"ADM2_PCODE"]],on="ADM2_EN",how="left")
df_floodscan_old=(df_floodscan_old[['date','ADM2_PCODE','mean_cell','max_cell']]
 .rename(columns={'mean_cell':'mean_ADM2_PCODE',
                  'max_cell':'max_ADM2_PCODE','date':'time'})
                  .sort_values(['time','ADM2_PCODE'])
                  .reset_index(drop=True)
)
```

```python
df_floodscan=pd.read_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm2_stats.nc',parse_dates=['time'])
df_floodscan_sel=(df_floodscan[['time','ADM2_PCODE','mean_ADM2_PCODE','max_ADM2_PCODE']]
                         .sort_values(['time','ADM2_PCODE'])
                        .reset_index(drop=True))
```

```python
#use assert_frame_equal cause with df_floodscan_sel.equals(df_floodscan_old)
#it is wrongly giving False
assert_frame_equal(df_floodscan_sel,df_floodscan_old)
```

```python

```