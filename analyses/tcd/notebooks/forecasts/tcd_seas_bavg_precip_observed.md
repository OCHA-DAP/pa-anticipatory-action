<!-- #region -->
# Observed lower tercile precipitation
This notebook explores the occurrence of historical below average precipitation in Chad. The dataset used is CHIRPS. 
The area of interest for the pilot are the following admin1 areas: Barh el Gazel, Batha, Kanem, Lac (une partie), Ouaddaï (une partie), Sila (une partie), Wadi Fira. It still has to be discussed which parts of the `une partie` regions will be included.


Therefore this analysis is mainly focussed on those areas. 

The main seasons that we are interested in for the trigger ar June-July-August (JJA) and July-August-September (JAS). This is the main focus of this analysis

If you are not working with CHD's Google Drive for the data access or want to update the data, run `src/tcd/get_chirps_data.py` before first time running the notebook

Resources
- [CHC's Early Warning Explorer](https://chc-ewx2.chc.ucsb.edu) is a nice resource to scroll through historically observed CHIRPS data
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import math
from dateutil.relativedelta import relativedelta

import altair as alt
#to plot maps with altair
import gpdvega

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data
from src.utils_general.statistics import get_return_periods_dataframe
from src.utils_general.raster_manipulation import compute_raster_statistics
```

```python
iso3="tcd"
```

```python
config=Config()
parameters = config.parameters(iso3)
data_raw_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR)
data_processed_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR)
country_data_raw_dir = os.path.join(data_raw_dir,iso3)
chirps_country_processed_dir = os.path.join(data_processed_dir,iso3,"chirps")

chirps_country_processed_path = os.path.join(chirps_country_processed_dir,"monthly",f"{iso3}_chirps_monthly.nc")
chirps_seasonal_lower_tercile_processed_path = os.path.join(chirps_country_processed_dir,"seasonal",f"{iso3}_chirps_seasonal_lowertercile.nc")
cerf_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,config.GLOBAL_ISO3,"cerf")
cerf_path=os.path.join(cerf_dir,'CERF Allocations.csv')

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

```python
month_season_mapping={1:"NDJ",2:"DJF",3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND"}
```

```python
hdx_red="#F2645A"
hdx_blue="#66B0EC"
hdx_green="#1EBFB3"
grey_med="#CCCCCC"
```

### Define parameters

```python
adm_sel=['Barh-El-Gazel','Batha','Kanem','Lac','Ouaddaï','Sila','Wadi Fira']
#the end months of 3 months periods that we are interested in
#i.e. end_month=8 represents the JJA season
end_months_sel=[8,9]
```

### Included areas

```python
gdf_adm=gpd.read_file(adm1_bound_path)
adm_col="admin1Name"
gdf_reg=gdf_adm[gdf_adm[adm_col].isin(adm_sel)]
```

```python
gdf_adm['include']=np.where(gdf_adm[adm_col].isin(adm_sel),True,False)
alt.Chart(gdf_adm).mark_geoshape(stroke="black").encode(
    color=alt.Color("include",scale=alt.Scale(range=["grey","red"])),
    tooltip=[adm_col]
).properties(width=600,title="Included admins")
```

## Analyzing observed precipitation patterns
We first have a look at the observational data per month to better understand the precipitation patterns in Chad.

Below the total precipitation during each month of 2020 is plotted. We can clearly see that most rainfall is received between June and September.    
We can also see that the Southern parts of the country receives more rainfall than the northern part

```python
#show distribution of rainfall across months for 2020 to understand rainy season patterns
ds_country=xr.load_dataset(chirps_country_processed_path)
da_country=ds_country.precip
da_country=da_country.rio.set_crs("EPSG:4326",inplace=True)
#show the data for each month of 2020, clipped to MWI
g=da_country.sel(time=da_country.time.dt.year.isin([2020])).plot(
    col="time",
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    cmap='Blues',
)

for ax in g.axes.flat:
    gdf_adm.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
```

IRI's seasonal forecast is produced as a probability of the rainfall being in the lower tercile, also referred to as being below average. We therefore compute the occurrence of observed below average rainfall.  

Below the areas with below average rainfall for the JJA and JAS season between 2004 and 2011 are shown. The date indicates the end of the rainy season, i.e. 2011-08 equals the JJA season in 2020.

We can see that there is geogspatial correlation between pixels receiving below average rainfall, which we would expect. 
However, this correlation occurs in smaller regions than in our full region of interest. For example in 2007 we can see that only the South-Eastern part of the region of interest received below average rainfall. 
The largest part of the country saw below average rainfall in 2004.

```python
ds_season_below=rioxarray.open_rasterio(chirps_seasonal_lower_tercile_processed_path)
da_season_below=ds_season_below.precip.sortby("time")
```

```python
da_season_below_rainy=da_season_below.sel(time=da_season_below.time.dt.month.isin(end_months_sel))
```

```python
da_plot=da_season_below_rainy.sel(time=da_season_below_rainy.time.dt.year.isin(range(2004,2012))).sortby("time")
da_plot_titles=[f"{month_season_mapping[int(m.dt.month.values)]} {int(m.dt.year.values)}" for m in da_plot.time]
g=da_plot.plot(
    col="time",
    col_wrap=2,
    levels=[-666,0],
    add_colorbar=False,
    cmap="YlOrRd",
)

for ax, title in zip(g.axes.flat, da_plot_titles):
    gdf_adm.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
    ax.set_title(title)
g.fig.suptitle("Pixels with below average rainfall",size=24)
g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
```

The plots below show the distribution of seasonal rainfall across the region. The red line indicates the tercile value averaged across all raster cells. This means it is slightly different for each raster cell, but it helps to get a general feeling

```python
seas_len=3
da_season=da_country.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")
```

```python
da_season_reg=da_season.rio.set_crs("EPSG:4326").rio.clip(gdf_reg["geometry"])
```

```python
colp_num=2
num_plots=len(end_months_sel)
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,6))
for i, m in enumerate(end_months_sel):
    ax = fig.add_subplot(rows,colp_num,i+1)
    da_season_reg_sel=da_season_reg.sel(time=da_season_reg.time.dt.month==m)
    g=sns.histplot(da_season_reg_sel.values.flatten(),color="#CCCCCC",ax=ax)
    perc=np.percentile(da_season_reg_sel.values.flatten()[~np.isnan(da_season_reg_sel.values.flatten())], 33)
    plt.axvline(perc,color="#C25048",label="lower tercile cap")
    ax.legend()
    ax.set(xlabel="Seasonal precipitation (mm)")
    ax.set_title(f"Distribution of seasonal precipitation in {month_season_mapping[m]} from {da_season_reg_sel.time.dt.year.values.min()}-{da_season_reg_sel.time.dt.year.values.max()}")
    ax.set_xlim(0,np.nanmax(da_season_reg.values))
```

We also show the distribution of rain across the whole country. We can see that this is significantly lower than in our region of interest

```python
colp_num=2
num_plots=len(end_months_sel)
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,6))
for i, m in enumerate(end_months_sel):
    ax = fig.add_subplot(rows,colp_num,i+1)
    da_season_sel=da_season.sel(time=da_season.time.dt.month==m)
    g=sns.histplot(da_season_sel.values.flatten(),color="#CCCCCC",ax=ax)
    perc=np.percentile(da_season_sel.values.flatten()[~np.isnan(da_season_sel.values.flatten())], 33)
    plt.axvline(perc,color="#C25048",label="lower tercile cap")
    ax.legend()
    ax.set(xlabel="Seasonal precipitation (mm)")
    ax.set_title(f"Distribution of seasonal precipitation in {month_season_mapping[m]} from 2000-2020")
    ax.set_xlim(0,np.nanmax(da_season.values))
```

Now that we have analyzed the data on pixel level, we aggregate to the area of interest. This area is the total area of the admins of interest.
We compute the percentage of the area having experienced below average rainfall for each season.

```python
gdf_adm1=gpd.read_file(adm1_bound_path)
#select the adms of interest
gdf_reg=gdf_adm1[gdf_adm1[adm_col].isin(adm_sel)]
```

```python
#since we are only selecting below avg values, only the count stat makes sense
#e.g. the mean doesn't reflect the actual situation, for that da_country would need to be used
gdf_reg_dissolved=gdf_reg.dissolve(by="admin0Name")
gdf_reg_dissolved=gdf_reg_dissolved[["admin0Pcod","geometry"]]
da_season_below_clip=da_season_below.rio.clip(gdf_reg["geometry"],all_touched=False)
da_season_below_bavg=da_season_below_clip.where(da_season_below >=0)

df_stats_reg=compute_raster_statistics(
        gdf=gdf_reg_dissolved,
        bound_col="admin0Pcod",
        raster_array=da_season_below_bavg,
        lon_coord="x",
        lat_coord="y",
        stats_list=["count"],
        percentile_list=[90],
    )
df_stats_reg_all=compute_raster_statistics(gdf=gdf_reg_dissolved,bound_col="admin0Pcod",raster_array=da_season_below,lon_coord="x",lat_coord="y",stats_list=["count"])

df_stats_reg["perc_bavg"] = df_stats_reg[f"count_admin0Pcod"]/df_stats_reg_all[f"count_admin0Pcod"]*100
df_stats_reg.time=pd.to_datetime(df_stats_reg.time.apply(lambda x: x.strftime("%Y-%m-%d")))
df_stats_reg["end_time"]=pd.to_datetime(df_stats_reg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg["end_month"]=df_stats_reg.end_time.dt.to_period("M")
df_stats_reg["start_time"]=df_stats_reg.end_time.apply(lambda x: x+relativedelta(months=-2))
df_stats_reg["start_month"]=df_stats_reg.start_time.dt.to_period("M")
df_stats_reg["season"]=df_stats_reg.end_month.apply(lambda x:month_season_mapping[x.month])
df_stats_reg["seas_year"]=df_stats_reg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
df_stats_reg["rainy_seas"]=np.where(df_stats_reg.start_month.dt.month.isin(end_months_sel),1,0)
df_stats_reg=df_stats_reg.sort_values("start_month")
df_stats_reg["rainy_seas_str"]=df_stats_reg["rainy_seas"].replace({0:"outside rainy season",1:"rainy season"})
df_stats_reg["year"]=df_stats_reg.end_month.dt.year
```

```python
df_stats_reg_rainy=df_stats_reg[df_stats_reg.end_month.dt.month.isin(end_months_sel)]
```

We can see that most occurences saw none of a small part of the region experience below average rainfall. This is logical, indicating geographical correlation.

```python
histo = alt.Chart().mark_bar(color="#CCCCCC").encode(
    x=alt.X("perc_bavg",bin=alt.Bin(step=10),title="Percentage of area with below average precipitation"),
    y="count()",
).properties(width=500)

histo.facet(column=alt.Column("season:N",sort=df_stats_reg_rainy.season.unique(),title="Season"),
                                       data=df_stats_reg_rainy[["season","perc_bavg"]],title=f"Distribution of percentage of area with below average precipitation from {df_stats_reg_rainy.end_month.dt.year.min()}-{df_stats_reg_rainy.end_month.dt.year.max()}"
           )
```

## Compute the return period
We compute the 1 in x years return period thereshold to better understand the historical trend and which years had extreme below average rainfall


Questions:
- I now mix the JJA and JAS season. Do you think that is sensible?

```python
df_stats_reg_rainy=df_stats_reg[df_stats_reg.end_month.dt.month.isin(end_months_sel)]
```

```python
df_stats_reg_rainy.sort_values("perc_bavg",ascending=False)
```

To compute the return period we are using the empirical method and not the analytic. This because the analytical method uses a GEV (Generalized Extreme Value) distribution to fit the curve. Due to the cap of the distribution being 100, the integral of the probability distribution becomes larger than 1 (i.e. assigning probabilities to larger values than 100). Moreover, the GEV method is expecting a more spread distribution to fit well. 

The empirical fitting is looking good though, so it is fine to stick to that

```python
years = np.arange(1.5, 20.5, 0.5)
df_rps_empirical_rainy = get_return_periods_dataframe(df_stats_reg_rainy, rp_var="perc_bavg",years=years, method="empirical")
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_empirical_rainy.index, df_rps_empirical_rainy["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('% below average')
```

```python
#round to multiples of 5
df_rps_empirical_rainy["rp_round"]=5*np.round(df_rps_empirical_rainy["rp"] / 5)
```

```python
df_rps_empirical_rainy.loc[[3,5]]
```

```python
#perc of occcurrences reaching 3rp since 2017
len(df_stats_reg_rainy[(df_stats_reg_rainy.start_month.dt.year>=2017)&(df_stats_reg_rainy.perc_bavg>=df_rps_empirical_rainy.loc[3,"rp_round"])])
```

As we can see since 2017 the 30% threshold of below average rainfall for a 1 in 3 year return period wasn't met. From this we can conclude that the last 4 years were less extreme than on average. This is important to take into account in the trigger design, as it leads to be wanting to trigger less often during these 4 years than on average.


### Plot percentage of below average rainfall


To get a better historical overview, we plot the percentage of below average rainfall per 3month period. The periods indicated as rainy season are JJA and JAS. 

From here we can see that 1982,1983,1984,1987,1989, 1990,1993,1996, and 2004 saw a percentage of the area receiving below average rain that reaches above the 1 in 5 year return period.    
1986, 2000, 2007, 2008, 2011, 2013,2015 saw 1 in 3 year return period percentages.

```python
df_stats_reg["season"]=df_stats_reg.end_month.apply(lambda x:month_season_mapping[x.month])
df_stats_reg["seas_year"]=df_stats_reg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
df_stats_reg["rainy_seas"]=np.where(df_stats_reg.end_month.dt.month.isin(end_months_sel),1,0)
df_stats_reg=df_stats_reg.sort_values("start_month")
df_stats_reg["rainy_seas_str"]=df_stats_reg["rainy_seas"].replace({0:"outside rainy season",1:"rainy season"})
df_stats_reg["year"]=df_stats_reg.end_month.dt.year
```

```python
g = sns.catplot(data=df_stats_reg, x="season",y="perc_bavg",col="year", hue="rainy_seas_str", col_wrap=4, kind="bar",
                  palette={"outside rainy season":grey_med,"rainy season":hdx_blue}, height=4, aspect=2)
g.map(plt.axhline, y=df_rps_empirical_rainy.loc[5,"rp_round"], linestyle='dashed', color=hdx_red, zorder=1,label="5 year return period")
g.map(plt.axhline,y=df_rps_empirical_rainy.loc[3,"rp_round"], linestyle='dashed', color=hdx_green, zorder=1,label="3 year return period")
```

### Correlation with other sources of historical drought


We have now identified the years that had large areas of below average precipitation during JJA and JAS in our region of interest. 

To understand if this correlates with drought that results in humanitarian needs, we compare this with a few data sources. 

It was shared by us [by email](https://docs.google.com/document/d/1c_L9Z_Y_P0vMmIiPaaeEQe8Vf6bzZHFUcsawis3zfCQ/edit) [we really need the original documents here or figure out who reported these years] that the worst drought years observed were 1993,1997,2001,2004,2009,2011,and 2017.

We assume years before 1993 were not reported in these sources, which leaves us with 29 years from 1993 to 2021. In that case we see a partial overlap with the reported years and observed below average precipitation. The dataframe shows the rank of the drought years. 

From the dataframe we can see that 1993 and 2004 were specifically bad years in observed precipitation. 2011 and 2009 also had some deficit, being the 8th and 9th worst year observed out of the 29. 

1996 was also a perticularly bad year in terms of precipitation, which might be the drought referred to by the year 1997 as impacts are sometimes felt later. 

With that same reasoning the 2001 drought might be caused by a deficit of rainfall in 2000 and 2009 drought might be worsened by deficits in 2007 and 2008.

2017 barely saw any below average rainfall in our region of interest. It could be that the drought is caused by the deficit of rain between ASO and DJF

Summarizing, we see some relation between the reported drought years and below average precipitation, thought this relation is not crystal clear. 

```python
df_rank=df_stats_reg_rainy[df_stats_reg_rainy.year>=1993].sort_values("perc_bavg",ascending=False).drop_duplicates("year").reset_index()
```

```python
drought_years=[1993,1997,2001,2004,2009,2011,2017]
```

```python
df_rank[df_rank.year.isin(drought_years)]
```

```python
df_rank.head()
```

#### CERF allocations


Another source of historical drought that is especially relevant to our case is the CERF allocations. We inspect the CERF allocations for drought since 2006, to see if these correlate with high levels of observed below average precipitation


As can be seen below, drought-related CERF funding since 2006 was released in 2010,2011,2012, and 2018. 

Sometimes CERF allocations are delayed compared to the actual event. 
We would therefore expect that for the years CERF funding was released, the rainy season of that year or the year before to be dryer. 
From the descriptions it seems likely all 2010-2012 findung could be caused by the 2011 deficits, and maybe the 2007/2008 rain deficits. 
It is unclear what caused the 2018 CERF drought-related funding.

```python
df=pd.read_csv(cerf_path,parse_dates=["dateUSGSignature"])
df["date"]=df.dateUSGSignature.dt.to_period('M')
df_country=df[df.countryCode==iso3.upper()]
df_countryd=df_country[df_country.emergencyTypeName=="Drought"]
#group year-month combinations together
df_countrydgy=df_countryd[["date","totalAmountApproved"]].groupby("date").sum()
```

```python
pd.set_option('display.max_colwidth', None)
df_countryd[["date","projectTitle"]]
```

```python
ax = df_countrydgy.plot(figsize=(16, 8), color='#86bf91',legend=False,kind="bar")

vals = ax.get_yticks()
for tick in vals[1:]:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

ax.set_xlabel("Month", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Total amount of funds released", labelpad=20, weight='bold', size=12)

ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Funds allocated by CERF for drought in {iso3} from 2006 till 2019");
```

### Conclusions
While we cannot relate all reported drought years with below average precipitation, we generally see quite a good correspondence.   
From this analysis it also seems that a 1 in 3 year return period is not too sensitive. Simeltaneously, we can conclude that the years for which we have IRI data (since 2017), there was not much observed below average precipitation

```python

```
