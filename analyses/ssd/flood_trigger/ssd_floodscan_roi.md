### Flooded fraction in the region of interest
As we can see in the notebook `ssd_floodscan_adm2`, there is a lot of fluctuation between the counties.
At the same time, the country level, as analyzed in `ssd_floodscan_country` might be too large. 
We therefore also examine a subset of counties, referred to as the region of interest. 

This notebook looks at the flooded fraction over time, and sees if the flooded fraction earlier in the season gives a signal for more extensive flooding later on. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import sys
from datetime import timedelta
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.util.testing import assert_frame_equal
import geopandas as gpd
import hvplot.xarray
import matplotlib as mpl
import numpy as np
from scipy import stats
from functools import reduce
import altair as alt
import panel.widgets as pnw
import calendar
import rioxarray

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.utils_general.statistics import get_return_periods_dataframe


# mpl.rcParams['figure.dpi'] = 300
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
```

#### define functions

```R
plotFloodedFraction <- function (df,y_col,facet_col,title){
df %>%
ggplot(
aes_string(
x = "time",
y = y_col
)
) +
stat_smooth(
geom = "area",
span = 1/4,
fill = "#ef6666"
) +
scale_x_date(
date_breaks = "3 months",
date_labels = "%b"
) +
facet_wrap(
as.formula(paste("~", facet_col)),
scales="free_x",
ncol=5
) +
ylab("Flooded fraction")+
xlab("Month")+
labs(title = title)+
theme_minimal()
}
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
gdf_adm2=gpd.read_file(adm2_bound_path)
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
fs_clip=xr.load_dataset(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
#I dont fully understand why, these grid mappings re-occur and what they mean
#but if having them, later on getting crs problems when computing stats
fs_clip.SFED_AREA.attrs.pop('grid_mapping')
fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
fs_clip=fs_clip.rio.write_crs("EPSG:4326",inplace=True)
```

```python
da_clip=fs_clip.SFED_AREA
```

```python
gdf_reg=gdf_adm2[gdf_adm2.ADM2_EN.isin(adm2_list)]
adm0_col="ADM0_EN"
pcode0_col="ADM0_PCODE"
```

```python
da_reg=da_clip.rio.clip(gdf_reg.geometry)
```

```python
#plot raster data with time slider
#can select subset of data to make it easier navigatible
#with slider
(da_reg
#  .sel(time=da_clip.time.dt.year==2014)
 .interactive.sel(time=pnw.DiscreteSlider).plot(
vmin=0,vmax=1,cmap="GnBu"))

```

```python
gdf_reg_dissolved=gdf_reg.dissolve(by=adm0_col)
gdf_reg_dissolved=gdf_reg_dissolved[[pcode0_col,"geometry"]]

df_floodscan_reg=compute_raster_statistics(
        gdf=gdf_reg_dissolved,
        bound_col=pcode0_col,
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=False,
    )
df_floodscan_reg['year']=df_floodscan_reg.time.dt.year
```

We can plot the data over all years. 
We see a yearly pattern where some years the peak is higher than others (though a max of 1.75% of the country is flooded). 

We see that some peaks have very high outliers, while others are wider. Which to classify as a flood, I am unsure about. With the method of std, we are now looking at the high outliers. 

```python
#should document but for now removing 2022 as it is not a full year
#but does have very high values till feb, so computations get a bit skewed with that
df_floodscan_reg=df_floodscan_reg[df_floodscan_reg.year<=2021]
```

```python
df_floodscan_reg['mean_rolling']=df_floodscan_reg.sort_values('time').mean_ADM0_PCODE.rolling(10,min_periods=10).mean()
```

```python
df_floodscan_reg['month'] = pd.DatetimeIndex(df_floodscan_reg['time']).month
df_floodscan_reg_rainy = df_floodscan_reg.loc[(df_floodscan_reg['month'] >= 7) & (df_floodscan_reg['month'] <= 10)]
```

```python
fig, ax = plt.subplots(figsize=(20,6))
sns.lineplot(data=df_floodscan_reg, x="time", y="mean_ADM0_PCODE", lw=0.25, label='Original')
sns.lineplot(data=df_floodscan_reg, x="time", 
             y="mean_rolling", lw=0.25, label='10-day moving\navg')   
ax.set_ylabel('Flooded fraction')
ax.set_xlabel('Date')
ax.set_title(f'Flooding in SSD, 1998-2020')
ax.legend()
```

Next we compute the return period and check which years had a peak above the return period. 
It is discussable whether only looking at the peak is the best method.. 

```python
#get one row per adm2-year combination that saw the highest mean value
df_floodscan_peak=df_floodscan_reg.sort_values('mean_rolling', ascending=False).drop_duplicates(['year'])
```

```python
years = np.arange(1.5, 20.5, 0.5)
```

```python
df_rps_ana=get_return_periods_dataframe(df_floodscan_peak, rp_var="mean_rolling",years=years, method="analytical",round_rp=False)
df_rps_emp=get_return_periods_dataframe(df_floodscan_peak, rp_var="mean_rolling",years=years, method="empirical",round_rp=False)
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_ana.index, df_rps_ana["rp"], label='analytical')
ax.plot(df_rps_emp.index, df_rps_emp["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('Fraction flooded');
```

We now use the empirical method, but could also use the analytical method. For the return periods of our interest, this will return less years. 

```python
df_floodscan_peak['rp3']=np.where(df_floodscan_peak.mean_rolling>=df_rps_emp.loc[3,'rp'],True,False)
df_floodscan_peak['rp5']=np.where(df_floodscan_peak.mean_rolling>=df_rps_emp.loc[5,'rp'],True,False)
```

```python
df_floodscan_peak[df_floodscan_peak.rp3].sort_values('mean_rolling')
```

```python
df_floodscan_peak[df_floodscan_peak.rp5].sort_values('mean_rolling')
```

```python
timest_rp5=list(df_floodscan_peak[df_floodscan_peak.rp5].sort_values('year').time)
da_clip_peak=da_clip.sel(time=da_clip.time.isin(timest_rp5))
```

```python
g=da_clip_peak.plot(col='time',
#                     cmap="GnBu"
                   )
for ax in g.axes.flat:
    ax.axis("off")
g.fig.suptitle(f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1);
```

```python
g=da_clip_peak.rio.clip(gdf_reg.geometry).plot(col='time')
for ax in g.axes.flat:
    ax.axis("off")
g.fig.suptitle(f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1);
```

Next we plot the smoothed data per year (with ggplot cause it is awesome). 

We can see that: 

- 2014, 2016, 2017, 2020, and 2021 indeed had clearly the highest peak. 
- This is the line with the signal on country level
- We also see some differences, e.g. the 2019 peak is lower. 

```R magic_args="-i df_floodscan_reg -w 40 -h 20 --units cm"
df_plot <- df_floodscan_reg %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM0_PCODE = mean_ADM0_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM0_PCODE','year',"Flooded fraction of ROI")
```

```python
#get one row per adm2-year-month combination that saw the highest mean value
df_floodscan_peak_month=df_floodscan_reg.sort_values('mean_rolling', 
                                                     ascending=False).drop_duplicates(['year','month'])
```

#TODO: check if assignment of month is correct..


I am sorry, from here the code gets really ugly

```python
# #method to compute if value of current month is above return period threshold
# #but I think the next method is better
# df_floodscan_rp_month=df_floodscan_peak_month.copy().dropna()
# for m in df_floodscan_reg.month.unique():
#     df_month=df_floodscan_rp_month[df_floodscan_rp_month.month==m]#.dropna()
#     df_rps_ana=get_return_periods_dataframe(df_month, rp_var="mean_rolling",years=years, method="analytical",round_rp=False)
#     df_rps_emp=get_return_periods_dataframe(df_month, rp_var="mean_rolling",years=years, method="empirical",round_rp=False)
#     df_floodscan_rp_month.loc[df_floodscan_rp_month.month==m,'rp5']=np.where((df_month.mean_rolling>=df_rps_emp.loc[5,'rp']),True,False)
#     df_floodscan_rp_month.loc[df_floodscan_rp_month.month==m,'rp3']=np.where((df_month.mean_rolling>=df_rps_emp.loc[3,'rp']),True,False)
```

```python
# df_floodscan_rp_month=df_floodscan_rp_month.merge(df_floodscan_peak[["year","rp3","rp5"]],on='year',suffixes=("","_year"))
# df_floodscan_rp_month.rp3=df_floodscan_rp_month.rp3.astype(bool)
```

```python
list_df_all=[]
for m in df_floodscan_reg.month.unique():
    df_month=df_floodscan_reg[df_floodscan_reg.month<=m].dropna()
    df_month_peak=df_month.sort_values('mean_rolling', ascending=False).drop_duplicates(['year'])
    df_rps_ana=get_return_periods_dataframe(df_month_peak, rp_var="mean_rolling",years=years, method="analytical",round_rp=False)
    df_rps_emp=get_return_periods_dataframe(df_month_peak, rp_var="mean_rolling",years=years, method="empirical",round_rp=False)
    df_month_peak['rp5']=np.where(df_month_peak.mean_rolling>=df_rps_emp.loc[5,'rp'],True,False)
    df_month_peak['rp3']=np.where(df_month_peak.mean_rolling>=df_rps_emp.loc[3,'rp'],True,False)
    df_month_peak['month']=m
    list_df_all.append(df_month_peak[['month','year','rp3','rp5']])
df_floodscan_rp_month_peak=pd.concat(list_df_all)
```

```python
df_floodscan_rp_month=df_floodscan_rp_month_peak.merge(df_floodscan_peak[["year","rp3","rp5"]],on='year',suffixes=("","_year"))
df_floodscan_rp_month.rp3=df_floodscan_rp_month.rp3.astype(bool)
```

```python
col_year="rp3_year"
col_month="rp3"
```

```python
def calc_precision(TP, FP):
    return TP / (TP + FP)

def calc_recall(TP, FN):
    return TP / (TP + FN)
```

```python
df_floodscan_rp_month["TP"]=np.where((df_floodscan_rp_month[col_year])&(df_floodscan_rp_month[col_month]),1,0)
df_floodscan_rp_month["FP"]=np.where((~df_floodscan_rp_month[col_year])&(df_floodscan_rp_month[col_month]),1,0)
df_floodscan_rp_month["TN"]=np.where((~df_floodscan_rp_month[col_year])&(~df_floodscan_rp_month[col_month]),1,0)
df_floodscan_rp_month["FN"]=np.where((df_floodscan_rp_month[col_year])&(~df_floodscan_rp_month[col_month]),1,0)
```

```python
df_metr=pd.DataFrame(columns=['month','recall','precision'])
for m in df_floodscan_rp_month.month.unique():
    df_month=df_floodscan_rp_month[df_floodscan_rp_month.month==m]
    df_sum = df_month.sum()
    precision = calc_precision(df_sum.TP, df_sum.FP)
    recall = calc_recall(df_sum.TP, df_sum.FN)
    df_metr.loc[m]=[int(m),precision,recall]
df_metr.month=df_metr.month.astype(int)
df_metr=df_metr.sort_values("month")
df_metr['month_abbr']=df_metr.month.apply(lambda x: calendar.month_abbr[x])
```

TODO: I am pretty sure smth is icky here, I will look into it!

```python
alt.Chart(df_metr).transform_fold(
    ['recall','precision'],
    as_=['metric', 'value']
).mark_line().encode(
    x=alt.X('month_abbr:N',sort=['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],axis=alt.Axis(grid=True),title="month"),
    y='value:Q',
    color='metric:N'
).properties(width=500,height=300,title="Performance peak flood till month against peak flood till end of year, 1 in 3 year return period")

```
