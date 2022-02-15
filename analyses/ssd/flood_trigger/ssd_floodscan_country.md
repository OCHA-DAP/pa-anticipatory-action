### Flooded fraction in the country
As we can see in the notebook `ssd_floodscan_adm2`, there is a lot of fluctuation between the counties. Since the division of counties is pretty artificial, we can also look at the total flooded fraction in the country. With this we can get a general idea of the floods across the country, after which we can zoom in on the specific counties. 

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
plotFloodedFraction <- function (df,y_col,facet_col){
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
ncol=4
) +
ylab("Flooded fraction")+
xlab("Month")+
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
cerf_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,config.GLOBAL_ISO3,"cerf")
cerf_path=os.path.join(cerf_dir,'CERF Allocations.csv')
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
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
#plot raster data with time slider
#can select subset of data to make it easier navigatible
#with slider
(da_clip
#  .sel(time=da_clip.time.dt.year==2014)
 .interactive.sel(time=pnw.DiscreteSlider).plot(
vmin=0,vmax=1))

```

```python
#gif of the timeseries
#the first loop it is whacky but after that it is beautiful
time = pnw.Player(name='time', start=0, end=122, 
                  step=7,
                  loop_policy='loop')

#select a year else it takes ages
da_clip.sel(time=(da_clip.time.dt.year==2021)&(da_clip.time.dt.month.isin([7,8,9,10]))).interactive(loc='bottom').isel(
    time=time).plot(
#     cmap="GnBu",
    vmin=0,vmax=1)
```

```python
df_floodscan_country=compute_raster_statistics(
        gdf=gdf_adm2,
        bound_col='ADM0_PCODE',
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","mean","max","count","sum"], #std, count
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        #Decided to only use centres, but can change that
        all_touched=False,
    )
df_floodscan_country['year']=df_floodscan_country.time.dt.year
```

```python
# df_floodscan_country.to_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm0_stats.csv')
```

We can plot the data over all years. 
We see a yearly pattern where some years the peak is higher than others (though a max of 1.75% of the country is flooded). 

We see that some peaks have very high outliers, while others are wider. Which to classify as a flood, I am unsure about. With the method of std, we are now looking at the high outliers. 

```python
df_floodscan_country['mean_rolling']=df_floodscan_country.sort_values('time').mean_ADM0_PCODE.rolling(10,min_periods=10).mean()
```

```python
df_floodscan_country['month'] = pd.DatetimeIndex(df_floodscan_country['time']).month
df_floodscan_country_rainy = df_floodscan_country.loc[(df_floodscan_country['month'] >= 7) | (df_floodscan_country['month'] <= 10)]
```

```python
fig, ax = plt.subplots(figsize=(20,6))
sns.lineplot(data=df_floodscan_country, x="time", y="mean_ADM0_PCODE", lw=0.25, label='Original')
sns.lineplot(data=df_floodscan_country, x="time", 
             y="mean_rolling", lw=0.25, label='10-day moving\navg')   
ax.set_ylabel('Flooded fraction')
ax.set_xlabel('Date')
ax.set_title(f'Flooding in SSD, 1998-2021')
ax.legend()
```

Next we compute the return period and check which years had a peak above the return period. 
It is discussable whether only looking at the peak is the best method.. 

```python
#get one row per adm2-year combination that saw the highest mean value
df_floodscan_peak=df_floodscan_country_rainy.sort_values('mean_rolling', ascending=False).drop_duplicates(['year'])
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

```python
df_floodscan_peak[df_floodscan_peak.mean_rolling>=df_rps_ana.loc[3,'rp']].sort_values('year')
```

```python
df_floodscan_peak[df_floodscan_peak.mean_rolling>=df_rps_ana.loc[5,'rp']].sort_values('year')
```

Next we plot the smoothed data per year (with ggplot cause it is awesome). 

We can see that: 

- 2014, 2017, and 2020 indeed had clearly the highest peak. 
- The peak is generally between Sep and Dec, which is quite late in the rainy season. 
- We can see several smaller peaks, such as in 2019. How to interpret these, I don't know

```python
df_floodscan_country_2021=df_floodscan_country[df_floodscan_country.year<=2021]
```

```R magic_args="-i df_floodscan_country_2021 -w 30 -h 20 --units cm"
df_plot <- df_floodscan_country_2021 %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM0_PCODE = mean_ADM0_PCODE*100)
plotFloodedFraction(df_plot,'mean_ADM0_PCODE','year')
```

### CERF allocations
To compare if the years of 2014, 2017, and 2020 correspond with actual flooding, one source we can look at are the CERF allocations

```python
df_cerf=pd.read_csv(cerf_path,parse_dates=["dateUSGSignature"])
df_cerf["date"]=df_cerf.dateUSGSignature.dt.to_period('M')
df_country=df_cerf[df_cerf.countryCode==iso3.upper()]
df_countryd=df_country[df_country.emergencyTypeName=="Flood"]
#group year-month combinations together
df_countrydgy=df_countryd[["date","totalAmountApproved"]].groupby("date").sum()
```

We can see that since 2006, allocations have occured in 2019, 2020, and 2021. 

during 2020 and 2021 we did see a large peak in the floodscan data. 

2019 saw above average flood levels, but not as extreme as during 2014 and 2017 when there were no allocations. 


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

plt.title(f"Funds allocated by CERF for floods in {iso3} from 2006 till 2021");
```

```python
#to get titles of the specific projects
# pd.set_option('display.max_colwidth', None)
# df_countryd[["date","projectTitle"]]
```

Possibly 2019 saw more local patterns. Looking at the [2019 flood allocation](https://cerf.un.org/sites/default/files/resources/19-RR-SSD-39576_South%20Sudan_CERF_Report.pdf), floods are explicity reported in the counties of Akobo, Mankien-Mayom, Rumbek North. We can thus evaluate the floodscan data for those counties

```python
df_floodscan_adm2=pd.read_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm2_stats.nc',parse_dates=['time'])
df_floodscan_adm2['year']=df_floodscan_adm2.time.dt.year
```

```python
#sel data
#need to remove some cols, else loading into R takes very long
df_floodscan_adm2_sel=df_floodscan_adm2.loc[(df_floodscan_adm2.ADM2_EN.isin(['Akobo','Mayom','Rumbek North'])),['time','mean_ADM2_PCODE','year','ADM2_EN']]
```

From the plot above we can see that Mayom saw quite a significant percentage of flooding. Akobo to some extent as well. 
However, it is important to look at the relativeness of these fraction compared to other years

```python
gdf_adm2['cerf_2019']=np.where(gdf_adm2["ADM2_EN"].isin(['Akobo','Mayom','Rumbek North']),True,False)
alt.Chart(gdf_adm2).mark_geoshape(stroke="black").encode(
    color=alt.Color("cerf_2019",scale=alt.Scale(range=["grey","red"])),
    tooltip=["ADM2_EN"]
).properties(width=400,height=300,title="Counties specfically mentioned by CERF 2019 allocation")
```

```R magic_args="-i df_floodscan_adm2_sel -w 30 -h 20 --units cm"
df_plot_adm2 <- df_floodscan_adm2_sel %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100) %>%
filter(year==2019)

plotFloodedFraction(df_plot_adm2,'mean_ADM2_PCODE','ADM2_EN')
```

From the plots below we can see that for Akobo and Mayom the flooding was definitely above average. Though we saw more substantial flooding in 2010 and 2014, when there was no allocation. For Rumbek North the flooding doesn't seem very exceptional though possibly slighlty more than normally. 

What to do with these results, I don't know hahah. I don't know whether to trust FloodScan or CERF more, but at least they don't have perfect correspondence. 

Also interesting to see that the counties mentioned in the report, are different than those we had in mind. Note that many of the CERF funding went to IDPs

```R magic_args="-i df_floodscan_adm2_sel -w 30 -h 20 --units cm"
df_plot_adm2 <- df_floodscan_adm2_sel %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100) %>%
filter(ADM2_EN=="Akobo")
plotFloodedFraction(df_plot_adm2,'mean_ADM2_PCODE','year')
```

```R magic_args="-i df_floodscan_adm2_sel -w 30 -h 20 --units cm"
df_plot_adm2 <- df_floodscan_adm2_sel %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100) %>%
filter(ADM2_EN=="Mayom")
plotFloodedFraction(df_plot_adm2,'mean_ADM2_PCODE','year')
```

```R magic_args="-i df_floodscan_adm2_sel -w 30 -h 20 --units cm"
df_plot_adm2 <- df_floodscan_adm2_sel %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM2_PCODE = mean_ADM2_PCODE*100) %>%
filter(ADM2_EN=="Rumbek North")
plotFloodedFraction(df_plot_adm2,'mean_ADM2_PCODE','year')
```
