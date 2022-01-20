# Observed lower tercile precipitation
This notebook explores the occurrence of historical below average precipitation in Chad. The dataset used is CHIRPS. 
The area of interest for the pilot are the following admin1 areas: Lac, Kanem, Barh-El-Gazel, Batha, and Wadi Fira. 

Therefore this analysis is mainly focussed on those areas. 

The main seasons that we are interested in for the trigger ar June-July-August (JJA) and July-August-September (JAS). This is the main focus of this analysis

If you are not working with CHD's Google Drive for the data access or want to update the data, run `src/tcd/get_chirps_data.py` before first time running the notebook

Resources
- [CHC's Early Warning Explorer](https://chc-ewx2.chc.ucsb.edu) is a nice resource to scroll through historically observed CHIRPS data

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
import calendar
import holoviews as hv
import hvplot.xarray
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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
data_raw_dir=Path(config.DATA_DIR)/config.PUBLIC_DIR/config.RAW_DIR
data_processed_dir=Path(config.DATA_DIR)/config.PUBLIC_DIR/config.PROCESSED_DIR
country_data_raw_dir = data_raw_dir/iso3
chirps_country_processed_dir = data_processed_dir/iso3/"chirps"

chirps_country_processed_path = chirps_country_processed_dir/"monthly"/f"{iso3}_chirps_monthly.nc"
chirps_seasonal_lower_tercile_processed_path = chirps_country_processed_dir/"seasonal"/f"{iso3}_chirps_seasonal_lowertercile.nc"
chirps_seasonal_tercile_bounds_processed_path = chirps_country_processed_dir/"seasonal"/f"{iso3}_chirps_seasonal_tercile_bounds.nc"
cerf_dir=data_raw_dir/config.GLOBAL_ISO3/"cerf"
cerf_path=cerf_dir/'CERF Allocations.csv'
plot_dir = data_processed_dir/iso3/"plots"

adm1_bound_path=country_data_raw_dir/config.SHAPEFILE_DIR/parameters["path_admin1_shp"]
adm2_path=data_processed_dir/iso3/config.SHAPEFILE_DIR / "tcd_adm2_area_of_interest.gpkg"
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
gdf_adm1=gpd.read_file(adm1_bound_path)
gdf_adm2=gpd.read_file(adm2_path)
incl_adm_col="area_of_interest"
gdf_aoi = gdf_adm2[gdf_adm2[incl_adm_col] == True]
#the end months of 3 months periods that we are interested in
#i.e. end_month=8 represents the JJA season
end_months_sel=[8,9]
months_sel=[6,7,8,9]
```

### Included areas

```python
alt.Chart(gdf_adm2).mark_geoshape(stroke="black").encode(
    color=alt.Color(incl_adm_col,scale=alt.Scale(range=["grey","red"])),
    tooltip=["admin2Name"]
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
    levels=20,
    cmap='Blues',
)

for ax in g.axes.flat:
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
```

Next we plot the median over all years of the monthly rainfall. We can see a similair pattern as in 2020

```python
#compute median rainfall per month, grouped by the years
da_country_month_med=da_country.groupby(
        da_country.time.dt.month
    ).median()

da_country_month_med=da_country.groupby(
        da_country.time.dt.month
    ).median()
g=da_country_month_med.plot(
    col="month",
    col_wrap=6,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Monthly precipitation (mm)"
    },
    levels=20,
    cmap='Blues',
)

for ax in g.axes.flat:
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    #dissolve to one polygon so we only plot the outer boundaries
    gdf_aoi.dissolve("admin0Pcod").boundary.plot(linewidth=0.5,ax=ax,color="red")
    ax.axis("off")
    ax.set_title(calendar.month_name[int(ax.get_title().split(" ")[-1])])
g.fig.suptitle(f"Median per month between {da_country.time.dt.year.values.min()} and {da_country.time.dt.year.values.max()}",size=16)
g.fig.subplots_adjust(top=0.9,bottom=0.4,hspace=0.3)
# plt.savefig(plot_dir/"tcd_map_med_monthly_precip.png")
```

To understand the differences between years, we aggregate the raster data to our area of interest by taking the mean. We then plot the observed mean precipitation per year and month. 

```python
da_country_aoi=da_country.rio.set_crs("EPSG:4326").rio.clip(gdf_aoi["geometry"])
da_country_aoi_mean=da_country_aoi.mean(dim=("latitude","longitude"))
da_country_aoi_mean_sel=da_country_aoi_mean.sel(time=da_country_aoi_mean.time.dt.month.isin(months_sel))
```

```python
df_country_aoi_mean_sel=da_country_aoi_mean_sel.to_dataframe().reset_index()
df_country_aoi_mean_sel["year"]=df_country_aoi_mean_sel.time.dt.year
df_country_aoi_mean_sel["month"]=df_country_aoi_mean_sel.time.dt.month
```

```python
df_country_aoi_mean_sel[["month","precip"]].groupby("month").mean()
```

```python
plot=alt.Chart().mark_bar(color=hdx_blue,opacity=0.7).encode(
    x=alt.X('year:N',title="Year"),
    y=alt.Y('precip', title = "mean precipitation"),
).properties(width=350,height=250)
(plot).facet(columns=2,
             facet="month:N",
             data=df_country_aoi_mean_sel, 
title=["Mean precipitation in the AOI per month and year"]).resolve_axis(
    x='independent',
)
```

From the graph above we can see that July and August clearly recieve the most rain. Moreover, we can see that the pattern can strongly differ per month. For example in 2000 the rain was very poor in August but normal in the other months. In 2004 it wasn't great during all months. It is unclear which month has the most impact and whether we should look at them in relative or absolute terms as the quantities differ so much per month. 

However, for now we stick to the notion of below average 3-monthly (=season) rainfall which we will explore next. 


### Observed seasonal rainfall


The plots below show the distribution of seasonal rainfall across the region for the JJA and JAS season. The values are computed per raster cell. 
We can see that the rainfall ranges from 0 to 700 mm with about one third of the observations being below 150mm. The distribution for JAS is slightly wider than for JJA meaning that the values are on average a bit higher.
<!-- The red line indicates the tercile value averaged across all raster cells. This means it is slightly different for each raster cell, but it helps to get a general feeling -->

```python
seas_len=3
da_season=da_country.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")
```

```python
da_season_reg=da_season.rio.set_crs("EPSG:4326").rio.clip(gdf_aoi["geometry"])
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
    ax.set(xlabel="Seasonal precipitation (mm)")
    ax.set_title(f"Distribution of seasonal precipitation in {month_season_mapping[m]} from {da_season_reg_sel.time.dt.year.values.min()}-{da_season_reg_sel.time.dt.year.values.max()}")
    ax.set_xlim(0,np.nanmax(da_season_reg.values))
```

We also show the distribution of rain across the whole country. We can see that this is significantly higher than in our region of interest

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
#     perc=np.percentile(da_season_sel.values.flatten()[~np.isnan(da_season_sel.values.flatten())], 33)
#     plt.axvline(perc,color="#C25048",label="lower tercile cap")
    ax.legend()
    ax.set(xlabel="Seasonal precipitation (mm)")
    ax.set_title(f"Distribution of seasonal precipitation in {month_season_mapping[m]} from 2000-2020")
    ax.set_xlim(0,np.nanmax(da_season.values))
```

### Observed below average tercile


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
da_season_below_selm=da_season_below.sel(time=da_season_below.time.dt.month.isin(end_months_sel))
```

```python
da_plot=da_season_below_selm.sel(time=da_season_below_selm.time.dt.year.isin(range(2004,2012))).sortby("time")
da_plot_titles=[f"{month_season_mapping[int(m.dt.month.values)]} {int(m.dt.year.values)}" for m in da_plot.time]
g=da_plot.plot(
    col="time",
    col_wrap=2,
    levels=[-666,0],
    add_colorbar=False,
    cmap="YlOrRd",
)

for ax, title in zip(g.axes.flat, da_plot_titles):
    gdf_adm1.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
    ax.set_title(title)
g.fig.suptitle("Pixels with below average rainfall",size=24)
g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
```

Above we showed which pixels were in the lower tercile for different years. Next we inspect the lower tercile bounds. When we are working with terciles, the bounds are computed per raster cell. A tercile is thus a relative definition instead of an absolute. When the range of precipitation seen in a raster call is very narrow, the difference between the tercile bounds will also be very now. This might lead to the forecast being too sensitive for small variations. 

Therefore we check if the lower tercile bounds look reasonable. 

```python
ds_season_bounds=rioxarray.open_rasterio(chirps_seasonal_tercile_bounds_processed_path)
da_season_bounds_reg=ds_season_bounds.rio.write_crs("EPSG:4326").rio.clip(gdf_aoi["geometry"]).precip
```

```python
g=da_season_bounds_reg.where(ds_season_bounds.month.isin(end_months_sel),drop=True).plot(col="month")
for m,ax in zip(end_months_sel,g.axes.flatten()):
    ax.set_title(f"lower tercile boundary for {month_season_mapping[m]}");
```

```python
plots_list = []
for m in end_months_sel:
    this_plot =  da_season_bounds_reg.sel(month=m).hvplot.kde().opts(ylabel="JAS precipitation",
title=f"Lower tercile boundary per raster cell for {month_season_mapping[m]}")
    plots_list.append(this_plot)
hv.Layout(plots_list).cols(1)
```

From the plots above we can see that the tercile boundaries differ quite widely across the region. Especially in the north-western part of our region of interest the boundaries are quite low. It might therefore be advised to exclude this region. However, I don't know if that is a reasonable thing to do and should maybe be discussed with meteorologists. 


### Percentage of area with below average rainfall


Now that we have analyzed the data on pixel level, we aggregate to the area of interest. This area is the total area of the admins of interest.
We compute the percentage of the area having experienced below average rainfall for each season.

```python
#since we are only selecting below avg values, only the count stat makes sense
#e.g. the mean doesn't reflect the actual situation, for that da_country would need to be used
gdf_aoi_dissolved=gdf_aoi.dissolve(by="admin0Name")
gdf_aoi_dissolved=gdf_aoi_dissolved[["admin0Pcod","geometry"]]
da_season_below_clip=da_season_below.rio.clip(gdf_aoi["geometry"],all_touched=False)
da_season_below_bavg=da_season_below_clip.where(da_season_below >=0)

df_stats_reg=compute_raster_statistics(
        gdf=gdf_aoi_dissolved,
        bound_col="admin0Pcod",
        raster_array=da_season_below_bavg,
        lon_coord="x",
        lat_coord="y",
        stats_list=["count"],
        percentile_list=[90],
    )
df_stats_reg_all=compute_raster_statistics(gdf=gdf_aoi_dissolved,bound_col="admin0Pcod",raster_array=da_season_below,lon_coord="x",lat_coord="y",stats_list=["count"])

df_stats_reg["perc_bavg"] = df_stats_reg[f"count_admin0Pcod"]/df_stats_reg_all[f"count_admin0Pcod"]*100
df_stats_reg.time=pd.to_datetime(df_stats_reg.time.apply(lambda x: x.strftime("%Y-%m-%d")))
df_stats_reg["end_time"]=pd.to_datetime(df_stats_reg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg["end_month"]=df_stats_reg.end_time.dt.to_period("M")
df_stats_reg["start_time"]=df_stats_reg.end_time.apply(lambda x: x+relativedelta(months=-2))
df_stats_reg["start_month"]=df_stats_reg.start_time.dt.to_period("M")
df_stats_reg["season"]=df_stats_reg.end_month.apply(lambda x:month_season_mapping[x.month])
df_stats_reg["seas_year"]=df_stats_reg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
df_stats_reg["seas_trig"]=np.where(df_stats_reg.end_month.dt.month.isin(end_months_sel),True,False)
df_stats_reg=df_stats_reg.sort_values("start_month")
df_stats_reg["seas_trig_str"]=df_stats_reg["seas_trig"].replace({True:"season included in trigger",False:"season not included in trigger"})
df_stats_reg["year"]=df_stats_reg.end_month.dt.year
```

```python
df_stats_reg_selm=df_stats_reg[df_stats_reg.end_month.dt.month.isin(end_months_sel)]
```

We can see that most occurences saw none of a small part of the region experience below average rainfall. This is logical, indicating geographical correlation.

```python
histo = alt.Chart().mark_bar(color="#CCCCCC").encode(
    x=alt.X("perc_bavg",bin=alt.Bin(step=10),title="Percentage of area with below average precipitation"),
    y="count()",
).properties(width=350)

histo.facet(column=alt.Column("season:N",sort=df_stats_reg_selm.season.unique(),title="Season"),
                                       data=df_stats_reg_selm[["season","perc_bavg"]],title=f"Distribution of percentage of area with below average precipitation from {df_stats_reg_selm.end_month.dt.year.min()}-{df_stats_reg_selm.end_month.dt.year.max()}"
           )
```

## Compute the return period
We compute the 1 in x years return period thereshold to better understand the historical trend and which years had extreme below average rainfall


Questions:
- I now mix the JJA and JAS season. Do you think that is sensible?

```python
df_stats_reg_selm=df_stats_reg[df_stats_reg.end_month.dt.month.isin(end_months_sel)]
```

```python
df_stats_reg_selm.sort_values("perc_bavg",ascending=False)
```

To compute the return period we are using the empirical method and not the analytic. This because the analytical method uses a GEV (Generalized Extreme Value) distribution to fit the curve. Due to the cap of the distribution being 100, the integral of the probability distribution becomes larger than 1 (i.e. assigning probabilities to larger values than 100). Moreover, the GEV method is expecting a more spread distribution to fit well. 

The empirical fitting is looking good though, so it is fine to stick to that

```python
years = np.arange(1.5, 20.5, 0.5)
df_rps_empirical_selm = get_return_periods_dataframe(df_stats_reg_selm, rp_var="perc_bavg",years=years, method="empirical")
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_empirical_selm.index, df_rps_empirical_selm["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('% below average')
```

```python
#round to multiples of 5
df_rps_empirical_selm["rp_round"]=5*np.round(df_rps_empirical_selm["rp"] / 5)
```

```python
df_rps_empirical_selm.loc[[3,5]]
```

```python
#perc of occcurrences reaching 3rp since 2017
len(df_stats_reg_selm[(df_stats_reg_selm.start_month.dt.year>=2017)&(df_stats_reg_selm.perc_bavg>=df_rps_empirical_selm.loc[3,"rp_round"])])
```

As we can see since 2017 the 30% threshold of below average rainfall for a 1 in 3 year return period wasn't met. From this we can conclude that the last 4 years were less extreme than on average. This is important to take into account in the trigger design, as it leads to be wanting to trigger less often during these 4 years than on average.


### Plot percentage of below average rainfall


To get a better historical overview, we plot the percentage of below average rainfall per 3month period. The periods indicated as rainy season are JJA and JAS. 

From here we can see that 1982,1983,1984,1987,1989, 1990,1993,1996, and 2004 saw a percentage of the area receiving below average rain that reaches above the 1 in 5 year return period.    
1986, 2000, 2007, 2008, 2011, 2013,2015 saw 1 in 3 year return period percentages.

```python
g = sns.catplot(data=df_stats_reg, x="season",y="perc_bavg",col="year", hue="seas_trig_str", col_wrap=3, kind="bar",
                  palette={"season not included in trigger":grey_med,"season included in trigger":hdx_blue}, height=4, aspect=2,legend=False,order=list(month_season_mapping.values()))
g.map(plt.axhline, y=df_rps_empirical_selm.loc[5,"rp_round"], linestyle='dashed', color=hdx_red, zorder=1,label="5 year return period")
#add_legend also adds the rp lines to the legend
g.map(plt.axhline,y=df_rps_empirical_selm.loc[3,"rp_round"], linestyle='dashed', color=hdx_green, zorder=1,label="3 year return period").add_legend()
g.set_ylabels("% of area observed below average precip")
g.set_xlabels("3-month period");
```

```python
#the years with highest percentage of bavg since 2000
#computing this since other data sources we use only start from 2000
df_stats_reg_selm[df_stats_reg_selm.year>=2000].sort_values("perc_bavg",ascending=False).head(10)
```

compute the rp per season as these can differ per season, depending on the geospatial correlation
However, I am surprised by the relatively large difference, especially when lookin at the 5 year return period..

```python
df_rps_all_list=[]
for seas in df_stats_reg_selm.season.unique():
    df_rps=get_return_periods_dataframe(df_stats_reg_selm[df_stats_reg_selm.season==seas], rp_var="perc_bavg",method="empirical")
    df_rps["season"]=seas
    df_rps_all_list.append(df_rps)
df_rps_seas=pd.concat(df_rps_all_list).rename_axis("rp_year").reset_index()
```

```python
df_rps_seas
```

```python
for ry in [3,5]:
    df_stats_reg_selm=df_stats_reg_selm.merge(df_rps_seas.loc[df_rps_seas.rp_year==ry,["rp","season"]].rename(columns={"rp":f"rp_{ry}"}),on="season")
```

We can also plot only the seasons of interest, which removes the clutter from the above bar plot.  
We can see that large areas of below average rainfall were more common in the 80s and 90s. In the last 25 years the 5 year return period was only reached once. 

```python
#TODO: the plot should have labels for the return periods lines but cannot get it to work..
plot=alt.Chart().mark_bar(color=hdx_blue,opacity=0.7).encode(
    x=alt.X('year:N',title="Year"),
    y=alt.Y('perc_bavg', title = "% of area with bavg precip"),
).properties(width=350,height=200)
rp3_line = alt.Chart().mark_rule(color=hdx_green,strokeDash=[12,6]).encode(
    y="rp_3:Q",)
rp5_line = alt.Chart().mark_rule(color=hdx_red,strokeDash=[12,6]).encode(
    y="rp_5:Q")
(plot+rp3_line+rp5_line).facet(column=alt.Column("season:N",sort=df_stats_reg_selm.season.unique(),title="season"),data=df_stats_reg_selm[["year","perc_bavg","season","rp_3","rp_5"]], 
title=["Percentage of the area with bavg obs precip by year and season","the green and red line are the 3 and 5 year return period"])
```

### Correlation with other sources of historical drought

<!-- #region -->
We have now identified the years that had large areas of below average precipitation during JJA and JAS in our region of interest. 

To understand if this correlates with drought that results in humanitarian needs, we compare this with a list of historical drought years that was compiled from several data sources. 


From this literature research, the most substantial drought years were 1968-1973, 1983, 1984, 1993, 1997, 2001, 2004, 2009, 2011, and 2017.
Note however that this list couldn't be fully validated, so not too much value should be attached to it. 

We see a partial overlap with the reported years and observed below average precipitation. The dataframe shows the rank of the drought years. 

There might also be delay between the precipitation and the drought years. 

e.g. 1996 was also a perticularly bad year in terms of precipitation, which might be the drought referred to by the year 1997 as impacts are sometimes felt later. 

With that same reasoning the 2001 drought might be caused by a deficit of rainfall in 2000 and 2009 drought might be worsened by deficits in 2007 and 2008.

2017 barely saw any below average rainfall in our region of interest. It could be that the drought is caused by the deficit of rain between ASO and DJF
<!-- #endregion -->

```python
df_rank=df_stats_reg_selm.sort_values("perc_bavg",ascending=False).drop_duplicates("year").reset_index()
```

```python
df_rank.head()
```

```python
#remains questionable what are the actual drought years 
#but this is a list compiled from several sources
drought_years=[1968,1969,1970,1971,1972,1973, 1983, 1984, 1993, 1997, 2001, 2004, 2009, 2011, 2017]
```

### Relation of below average precipitation and drought


By looking at the confusion matrices we can see that the drought years don't correspond very well with observed below average precipitation. There might be a number of reasons:  
1) It seems that precipitation patterns have changed. In the 80s and 90s it was much more common that large areas experience below average rainfall
2) Below average rainfall over a 3 month period might not capture all types of droughts. For example a late onset or long dry spells could also have a significant socio-economic impact. 
3) socio-economic drought might also be (partially) caused by other factors than precipitation, such as temperature

```python
def compute_confusionmatrix_column(df,subplot_col,target_col,predict_col, ylabel,xlabel,colp_num=3,title=None,adjust_top=None):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df[subplot_col].unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(15,8))
    for i, m in enumerate(df.sort_values(by=subplot_col)[subplot_col].unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df[subplot_col]==m,target_col]
        y_predicted = df.loc[df[subplot_col]==m,predict_col]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{subplot_col}={m}")
    if title is not None:
        fig.suptitle(title)
    if adjust_top is not None:
        plt.subplots_adjust(top=adjust_top)
    return fig
```

```python
df_stats_drought=df_stats_reg_selm.copy()
```

```python
#drought years indicated by drought_years
df_stats_drought["drought"]=np.where(df_stats_drought.year.isin(drought_years),True,False)
df_stats_drought["rp3"]=np.where(df_stats_drought.perc_bavg>=df_rps_empirical_selm.loc[3,"rp_round"],True,False)
df_stats_drought["rp5"]=np.where(df_stats_drought.perc_bavg>=df_rps_empirical_selm.loc[5,"rp_round"],True,False)
df_stats_drought["bavg_20"]=np.where(df_stats_drought.perc_bavg>=20,True,False)
```

```python
fig=compute_confusionmatrix_column(df_stats_drought,"season","drought","bavg_20","drought year in framework",
                               f">=20% of area below average",
                              title=f"Correspondence of >=20% area observed below average precipitation and reported drought years",
                                  adjust_top=1.2)
```

```python
rp_cm=[3,5]
for rp in rp_cm:
    fig=compute_confusionmatrix_column(df_stats_drought,"season","drought",f"rp{rp}","drought year in framework",
                                   f"percentage bavg above {rp} year return period (>={int(df_rps_empirical_selm.loc[rp,'rp_round'])}%)",
                                  title=f"Correspondence of 1 in {rp} year observed below average precipitation and reported drought years",
                                      adjust_top=1.2)
```

```python
def compute_performance_metrics(df,subgroup_col,target_col,predict_col,num_dec=2):
    df_perf=pd.DataFrame(index=df[subgroup_col].unique(),columns=["valid_act_rate","fa_rate","det_rate","miss_rate","acc"],dtype=float)
    for i, m in enumerate(df.sort_values(by=subgroup_col)[subgroup_col].unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df[subgroup_col]==m,target_col]
        y_predicted = df.loc[df[subgroup_col]==m,predict_col]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        valid_act_rate=tp/(tp+fp)
        fa_rate=1-valid_act_rate
        det_rate=tp/(tp+fn)
        miss_rate=1-det_rate
        acc=(tp+tn) / (tp+tn+fp+fn)
        df_perf.loc[m,["valid_act_rate","fa_rate","det_rate","miss_rate","acc"]]=[valid_act_rate,fa_rate,det_rate,miss_rate,acc]
    #multiply by 100 to get percentage
    df_perf=df_perf*100
    return df_perf.round(decimals=num_dec)
```

```python
compute_performance_metrics(df_stats_drought,"season","drought","rp5")
```

```python
#group the seasons by taking the max value
#max value of [True,False] is True
df_stats_drought_yearly=df_stats_drought.groupby("year").max()
```

```python
rp=3
fig=compute_confusionmatrix_column(df_stats_drought_yearly,"admin0Pcod","drought",f"rp{rp}","drought year in framework",
                                   f"percentage bavg above {rp} year return period (>={int(df_rps_empirical_selm.loc[rp,'rp_round'])}%)",
                                  title=f"Correspondence of 1 in {rp} year observed below average precipitation and reported drought years")
```

```python
plot=alt.Chart().mark_bar(color=hdx_blue,opacity=0.7).encode(
    x=alt.X('year:N',title="Year"),
    y=alt.Y('perc_bavg', title = "% of area with bavg precip"),
).properties(width=600,height=400)
(plot).facet(column=alt.Column("season:N",sort=df_stats_drought.season.unique(),title="season"),data=df_stats_drought[["year","perc_bavg","drought","season"]], 
title=["Percentage of the area with bavg obs precip by year and season"])
# plt_facet.save(str(plot_dir / "tcd_obs_bavg_perc_jja_jas.png"))#,format="png")
```

```python
plot=alt.Chart().mark_bar(color=hdx_blue,opacity=0.7).encode(
    x=alt.X('year:N',title="Year"),
    y=alt.Y('perc_bavg', title = "% of AOI with below avg precip"),
    color=alt.Color('drought:N',scale=alt.Scale(range=[grey_med,hdx_red])),
).properties(width=600,height=400)
(plot).facet(column=alt.Column("season:N",sort=df_stats_drought.season.unique(),title="season"),data=df_stats_drought[["year","perc_bavg","drought","season"]], 
title=["Percentage of the AOI with below average observed precipitation by year and season"])
# plt_facet.save(str(plot_dir/"tcd_obs_bavg_perc_drought_jja_jas.png"))
```

### Conclusions
We see that the correspondence between historical below average precipitation and drought years is not great.  
From this analysis it also seems that a 1 in 5 year return period suits best with the historical droughts. 

We can also conclude that the years for which we have IRI data (since 2017), there was not much observed below average precipitation


#### Anenx: CERF allocations


Below the code to analyze the years CERF released funding


As can be seen below, drought-related CERF funding since 2006 was released in 2010,2011,2012, and 2018. 

Sometimes CERF allocations are delayed compared to the actual event. 
We would therefore expect that for the years CERF funding was released, the rainy season of that year or the year before to be dryer. 
From the dates and descriptions it seems likely that the 2010 funding relates to the 2009 drought mentioned in the framework, the 2011 and 2012 funding to the 2011 drought, and the 2018 funding to the 2017 drought. In that sense the CERF allocations correspond with the years mentioned by the drought framework. 

As mentioned above the 2009 and 2011 years were not the worst years in terms of rainfall deficits, and in 2017 no rainfall deficits were seen. One reasoning might be that deficits of previous years caused a cumulative effect, though these deficits weren't particularly massive either. 

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
