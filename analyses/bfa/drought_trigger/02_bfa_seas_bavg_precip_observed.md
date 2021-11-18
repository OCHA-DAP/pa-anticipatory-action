---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: antact
  language: python
  name: antact
---

# Observed lower tercile precipitation
This notebook explores the occurrence of below average precipitation in Burkina Faso. The dataset used is CHIRPS. 
The area of interest for the pilot are 4 admin1 areas: Boucle de Mounhoun, Centre Nord, Sahel, and Nord. Therefore this analysis is mainly focussed on those areas. 

The trigger is mainly focussing on the JJA and ASO seasons, but this analysis focusses on all seasons. 

Resources
- [CHC's Early Warning Explorer](https://chc-ewx2.chc.ucsb.edu) is a nice resource to scroll through historically observed CHIRPS data

```{code-cell} ipython3
:tags: [remove_cell]

%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
:tags: [remove_cell]

import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from rasterstats import zonal_stats
from IPython.display import Markdown as md
from myst_nb import glue
from dateutil.relativedelta import relativedelta
import math
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import re

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

from src.indicators.drought.iri_rainfallforecast import get_iri_data
from src.utils_general.statistics import get_return_period_function_analytical, get_return_period_function_empirical
```

```{code-cell} ipython3
:tags: [remove_cell]

country="bfa"
country_iso3="bfa"
adm_sel=["Boucle du Mouhoun","Nord","Centre-Nord","Sahel"]
adm_sel_str=re.sub(r"[ -]", "", "".join(adm_sel)).lower()
config=Config()
parameters = config.parameters(country)
data_raw_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR)
data_processed_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR)
country_data_raw_dir = os.path.join(data_raw_dir,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country)
glb_data_raw_dir = os.path.join(data_raw_dir,"glb")
chirps_monthly_raw_dir = os.path.join(glb_data_raw_dir,"chirps","monthly")
chirps_country_processed_dir = os.path.join(data_processed_dir,country,"chirps")

chirps_monthly_path=os.path.join(chirps_monthly_raw_dir,"chirps_glb_monthly.nc")
chirps_country_processed_path = os.path.join(chirps_country_processed_dir,"monthly",f"{country}_chirps_monthly.nc")
chirps_seasonal_lower_tercile_processed_path = os.path.join(chirps_country_processed_dir,"seasonal",f"{country}_chirps_seasonal_lowertercile.nc")
chirps_exploration_dir = os.path.join(country_data_exploration_dir,"chirps")
stats_reg_observed_path=os.path.join(chirps_exploration_dir,f"{country}_chirps_seasonal_bavg_stats_{''.join(adm_sel_str)}.csv")
cerf_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,config.GLOBAL_ISO3,"cerf")
cerf_path=os.path.join(cerf_dir,'CERF Allocations.csv')

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```{code-cell} ipython3
:tags: [remove_cell]

month_season_mapping={1:"NDJ",2:"DJF",3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND"}
```

```{code-cell} ipython3
hdx_red="#F2645A"
hdx_blue="#66B0EC"
hdx_green="#1EBFB3"
grey_med="#CCCCCC"
```

## Analyzing observed precipitation patterns
Before we compare the forecasts and observations, we first have a look at the observational data to better understand the precipitation patterns in Burkina Faso. 

Below the total precipitation during each month of 2020 is plotted. We can clearly see that most rainfall is received between June and September.

```{code-cell} ipython3
:tags: [hide_input]

#show distribution of rainfall across months for 2020 to understand rainy season patterns
ds_country=xr.open_dataset(chirps_country_processed_path)
#show the data for each month of 2020, clipped to MWI
g=ds_country.sel(time=ds_country.time.dt.year.isin([2020])).precip.plot(
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

df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
```

IRI's seasonal forecast is produced as a probability of the rainfall being in the lower tercile, also referred to as being below average. We therefore compute the occurrence of observed below average rainfall.  

Below the areas with below average rainfall for each season ending in 2020 are shown. The date indicates the end of the rainy season, i.e. 2020-06 equals the April-May-June season.

```{code-cell} ipython3
:tags: [remove_cell]

#TODO: change structure ds such that time is divided into a year and a season
ds_season_below=rioxarray.open_rasterio(chirps_seasonal_lower_tercile_processed_path)
```

```{code-cell} ipython3
:tags: [remove_cell]

ds_season_below
```

```{code-cell} ipython3
:tags: [hide_input]

#show the data for each month of 2020, clipped to MWI
#TODO change subplot titles
g=ds_season_below.sel(time=ds_season_below.time.dt.year.isin([2020])).precip.plot(
    col="time",
    col_wrap=6,
    levels=[-666,0],
    add_colorbar=False,
    cmap="YlOrRd",
)

df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="grey")
    ax.axis("off")
g.fig.suptitle("Pixels with below average rainfall",size=24)
g.fig.tight_layout()
```

The plots below show the distribution of seasonal rainfall across the whole country (so not only the region of interest). The red line indicates the tercile value averaged across all raster cells. This means it is slightly different for each raster cell, but it helps to get a general feeling

```{code-cell} ipython3
:tags: [hide_input]

seas_len=3
ds_season=ds_country.rolling(time=seas_len,min_periods=seas_len).sum().dropna(dim="time",how="all")

# season_end_months=[6,10]
season_end_months=[8]
colp_num=2
num_plots=len(season_end_months)
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,6))
for i, m in enumerate(season_end_months):
    ax = fig.add_subplot(rows,colp_num,i+1)
    ds_season_sel=ds_season.sel(time=ds_season.time.dt.month==m)
    g=sns.histplot(ds_season_sel.precip.values.flatten(),color="#CCCCCC",ax=ax)
    perc=np.percentile(ds_season_sel.precip.values.flatten()[~np.isnan(ds_season_sel.precip.values.flatten())], 33)
    plt.axvline(perc,color="#C25048",label="lower tercile cap")
    ax.legend()
    ax.set(xlabel="Seasonal precipitation (mm)")
    ax.set_title(f"Distribution of seasonal precipitation in {month_season_mapping[m]} from 2000-2020")
    ax.set_xlim(0,np.nanmax(ds_season.precip.values))
```

Now that we have analyzed the data on pixel level, we aggregate to the area of interest. This area is the total area of the 4 admin1s: Boucle de Mounhoun, Centre Nord, Sahel, and Nord   
We compute the percentage of the area having experienced below average rainfall for each season.   
We can see that about 1/3 of the time none of the area experiences below average rainfall.   
Logically, it is less likely that larger areas have below average rainfall. However, this diminishment is relatively small, indicating geographical correlation.

```{code-cell} ipython3
:tags: [remove_cell]

def compute_zonal_stats_xarray(raster,shapefile,lon_coord="lon",lat_coord="lat",var_name="prob"):
    raster_clip=raster.rio.set_spatial_dims(x_dim=lon_coord,y_dim=lat_coord).rio.clip(shapefile.geometry.apply(mapping),raster.rio.crs,all_touched=False)
    raster_clip_bavg=raster_clip.where(raster_clip[var_name] >=0)
    grid_mean = raster_clip_bavg.mean(dim=[lon_coord,lat_coord]).rename({var_name: "mean_cell"})
    grid_min = raster_clip_bavg.min(dim=[lon_coord,lat_coord]).rename({var_name: "min_cell"})
    grid_max = raster_clip_bavg.max(dim=[lon_coord,lat_coord]).rename({var_name: "max_cell"})
    grid_std = raster_clip_bavg.std(dim=[lon_coord,lat_coord]).rename({var_name: "std_cell"})
    grid_quant90 = raster_clip_bavg.quantile(0.9,dim=[lon_coord,lat_coord]).rename({var_name: "10quant_cell"})
    grid_percbavg = raster_clip_bavg.count(dim=[lon_coord,lat_coord])/raster_clip.count(dim=[lon_coord,lat_coord])*100
    grid_percbavg=grid_percbavg.rename({var_name: "bavg_cell"})
    zonal_stats_xr = xr.merge([grid_mean, grid_min, grid_max, grid_std,grid_quant90,grid_percbavg]).drop("spatial_ref")
    zonal_stats_df=zonal_stats_xr.to_dataframe()
    zonal_stats_df=zonal_stats_df.reset_index()
    zonal_stats_df=zonal_stats_df.drop("quantile",axis=1)
    return zonal_stats_df
```

```{code-cell} ipython3
:tags: [remove_cell]

gdf_adm1=gpd.read_file(adm1_bound_path)
#select the adms of interest
gdf_reg=gdf_adm1[gdf_adm1.ADM1_FR.isin(adm_sel)]
```

```{code-cell} ipython3
:tags: [remove_cell]

#compute stats
df_stats_reg=compute_zonal_stats_xarray(ds_season_below,gdf_reg,lon_coord="x",lat_coord="y",var_name="precip")
#TODO: check if there are cases where nan shouldn't be filled with 0. But at least in cases where no below avg was observed, this is nan otherwise
df_stats_reg=df_stats_reg.fillna(0)
df_stats_reg["end_time"]=pd.to_datetime(df_stats_reg["time"].apply(lambda x: x.strftime('%Y-%m-%d')))
df_stats_reg["end_month"]=df_stats_reg.end_time.dt.to_period("M")
df_stats_reg["start_time"]=df_stats_reg.end_time.apply(lambda x: x+relativedelta(months=-2))
df_stats_reg["start_month"]=df_stats_reg.start_time.dt.to_period("M")
```

```{code-cell} ipython3
# df_stats_reg.to_csv(stats_reg_observed_path,index=False)
```

```{code-cell} ipython3
:tags: [remove_cell]

perc_obs_bavg_1in3 = int(np.percentile(df_stats_reg.bavg_cell, 66)) #round cause later on used as threshold
glue("perc_obs_bavg_1in3", perc_obs_bavg_1in3)
```

If we would use a threshold on the percentage of the area that experiences below average rainfall such that it is a 1 in 3 occurence event, this threshold would be {glue:text}`perc_obs_bavg_1in3`%

```{code-cell} ipython3
:tags: [hide_input]

#plot distribution of area with below avg rainfall
g=sns.histplot(df_stats_reg.bavg_cell,kde=True,color="#CCCCCC")
sns.despine()
g.set(xlabel="Percentage of area with below average precipitation")
plt.title("Distribution of percentage of area with below average precipitation from 1982-2021");
```

```{code-cell} ipython3
:tags: [remove_cell]

threshold_for_prob=40
glue("threshold_for_prob", threshold_for_prob)
```

## Compute the return period
Better understand the historical trend and which years had extreme below average rainfall

```{code-cell} ipython3
def get_return_periods(
    df: pd.DataFrame,
    rp_var: str,
    years: list = None,
    method: str = "analytical",
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    :param df: Dataframe with data to compute rp on
    :param rp_var: column name to compute return period on
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :param show_plots: If method is analytical, can show the histogram and GEV distribution overlaid
    :return: Dataframe with return period years as index and stations as columns
    """
    if years is None:
        years = [1.5, 2, 3, 5]#, 10]#, 20]
    df_rps = pd.DataFrame(columns=["rp"],index=years)
    if method == "analytical":
        f_rp = get_return_period_function_analytical(
            df_rp=df, rp_var=rp_var, show_plots=show_plots
        )
    elif method == "empirical":
        f_rp = get_return_period_function_empirical(
            df_rp=df, rp_var=rp_var,
        )
    else:
        logger.error(f"{method} is not a valid keyword for method")
        return None
    df_rps["rp"] = np.round(f_rp(years))
    return df_rps
```

Not using the analytical method since, this uses a GEV (Generalized Extreme Value) distribution to fit the curve. Due to the cap of the distribution being 100, the integral of the probability distribution becomes larger than 1 (i.e. assigning probabilities to larger values than 100). Moreover, the GEV method is expecting a more spread distribution to fit well. 

The empirical fitting is looking good though, so we are using this.

```{code-cell} ipython3
years = np.arange(1.5, 20.5, 0.5)
df_rps_empirical = get_return_periods(df_stats_reg, rp_var="bavg_cell",years=years, method="empirical")
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(df_rps_empirical.index, df_rps_empirical["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('% below average')
```

```{code-cell} ipython3
#round to multiples of 5
df_rps_empirical["rp_round"]=5*np.round(df_rps_empirical["rp"] / 5)
```

```{code-cell} ipython3
df_rps_empirical.loc[[3,5]]
```

```{code-cell} ipython3
#average perc of occurrences reaching 5 rp
len(df_stats_reg[df_stats_reg.bavg_cell>=df_rps_empirical.loc[5,"rp_round"]])/len(df_stats_reg)
```

```{code-cell} ipython3
#perc of occcurrences reaching 5rp since 2017
len(df_stats_reg[(df_stats_reg.start_month.dt.year>=2017)&(df_stats_reg.bavg_cell>=df_rps_empirical.loc[5,"rp_round"])])/len(df_stats_reg[(df_stats_reg.start_month.dt.year>=2017)])
```

```{code-cell} ipython3
#perc of occcurrences reaching 3rp since 2017
len(df_stats_reg[(df_stats_reg.start_month.dt.year>=2017)&(df_stats_reg.bavg_cell>=df_rps_empirical.loc[3,"rp_round"])])/len(df_stats_reg[(df_stats_reg.start_month.dt.year>=2017)])
```

As we can see the occurrences of 1 in 3 and 1 in 5 year events since 2017 is a lot lower than would be expected on average. From this we can conclude that the last 4 years were less extreme than on average. This is important to take into account in the trigger design, as it leads to be wanting to trigger less often during these 4 years than on average.

+++

Below we are distinguishing between rainy and not rainy season, to have more clearity in the plot. The seasons chosen here as rainy season are an approximation and not officially approved. 
They are based on [FewsNet's calendar](https://fews.net/west-africa/burkina-faso) which indicates the rainy season to be from mid May till mid October. We therefore set the seasons starting between May and September to be within the rainy season.

```{code-cell} ipython3
df_stats_reg["season"]=df_stats_reg.end_month.apply(lambda x:month_season_mapping[x.month])
df_stats_reg["seas_year"]=df_stats_reg.apply(lambda x: f"{x.season} {x.end_month.year}",axis=1)
df_stats_reg["rainy_seas"]=np.where((df_stats_reg.start_month.dt.month>=5)&(df_stats_reg.start_month.dt.month<=9),1,0)
df_stats_reg=df_stats_reg.sort_values("start_month")
df_stats_reg["rainy_seas_str"]=df_stats_reg["rainy_seas"].replace({0:"outside rainy season",1:"rainy season"})
```

```{code-cell} ipython3
:tags: [hide_input]

#determine historically below average rainy seasons, observed data
fig, ax = plt.subplots(figsize=(20,8))

stats_byear=df_stats_reg[df_stats_reg.start_month.dt.year>=2017]

# plt.bar(stats_byear["seas_year"],stats_byear["bavg_cell"])
sns.barplot(x='seas_year', y='bavg_cell', data=stats_byear, hue="rainy_seas_str",ax=ax,palette={"outside rainy season":grey_med,"rainy season":hdx_blue})
sns.despine(fig)
x_dates = stats_byear.seas_year.unique()
# x_dates=stats_byear.end_month.dt.year
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right');
ax.set_ylabel("Percentage of area",size=16)
ax.set_xlabel("Season",size=16)
ax.set_ylim(0,100)
ax.set_title("Percentage of area with observed below average precipitation",size=20);
ax.axhline(y=df_rps_empirical.loc[5,"rp_round"], linestyle='dashed', color=hdx_red, zorder=1,label="5 year return period")
ax.axhline(y=df_rps_empirical.loc[3,"rp_round"], linestyle='dashed', color=hdx_green, zorder=1,label="3 year return period")
plt.legend()
```

```{code-cell} ipython3
df_stats_reg["year"]=df_stats_reg.end_month.dt.year
g = sns.catplot(data=df_stats_reg[df_stats_reg.year>=2000], x="season",y="bavg_cell",col="year", hue="rainy_seas_str", col_wrap=4, kind="bar",
                  palette={"outside rainy season":grey_med,"rainy season":hdx_blue}, height=4, aspect=2)
g.map(plt.axhline, y=df_rps_empirical.loc[5,"rp_round"], linestyle='dashed', color=hdx_red, zorder=1,label="5 year return period")
g.map(plt.axhline,y=df_rps_empirical.loc[3,"rp_round"], linestyle='dashed', color=hdx_green, zorder=1,label="3 year return period")
```

### CERF allocations

+++

We inspect the CERF allocations for drought since 2006, to see if these correlate with high levels of observed below average precipitation

+++

As can be seen below, most drought-related CERF funding since 2006 was released in 2008, 2012, 2014, and 2018. 
This means that since 2006, about once every 3 years substantial funding related to drought was released. This might be an indication that we would want the trigger to be met on average once every 3 years, though this might be a too simplistic view. 

Sometimes CERF allocations are delayed compared to the actual event. 
We would therefore expect that for the years CERF funding was released, the rainy season of that year or the year before to be dryer. 

From the precipitation graphs above, we can see that 2008, 2011, and 2017 were relatively worse seasons. Though definitely not as bad as was seen between 2000 and 2004. The reason for the funding in 2014 cannot be clearly seen in the precipitation patterns.

```{code-cell} ipython3
df=pd.read_csv(cerf_path,parse_dates=["dateUSGSignature"])
df["date"]=df.dateUSGSignature.dt.to_period('M')
df_country=df[df.countryCode==parameters["iso3_code"].upper()]
df_countryd=df_country[df_country.emergencyTypeName=="Drought"]
#group year-month combinations together
df_countrydgy=df_countryd[["date","totalAmountApproved"]].groupby("date").sum()
```

```{code-cell} ipython3
pd.set_option('display.max_colwidth', None)
```

```{code-cell} ipython3
df_countryd.tail(n=1).projectTitle
```

```{code-cell} ipython3
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

plt.title(f"Funds allocated by CERF for drought in {country.upper()} from 2006 till 2019");
```

```{code-cell} ipython3

```
