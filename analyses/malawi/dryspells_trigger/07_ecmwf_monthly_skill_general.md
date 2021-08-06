### Evaluating the forecast skill of ECMWF seasonal forecast in Malawi
This notebook is to compare the forecast skill of ECMWF's seasonal forecast for various lead times. We use the monthly total precipitation that is forecasted by this forecast. As ground truth, we use CHIRPS observations. 

We first assess the skill at cell level. To assess this skill across ensemble members, we compute the Continuous Ranked Probability Score (CRPS). We investigate the CRPS for different sets of data, e.g. for months with low rainfall. 

Thereafter we look at the bias of the median of all ensemble members. 

`mwi_ecmwf_monthly_skill_southern.md` assesses the skill at the admin1 level for the Southern region in Malawi.     
`mwi_ecmwf_monthly_skill_dryspells.md` assesses the skill of the forecast for dry spells specifically. 

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import plotly.express as px 
import seaborn as sns
import calendar
import math
import geopandas as gpd
from matplotlib.lines import Line2D

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'size'   : 16}

mpl.rc('font', **font)
```

```python
#set plot colors
hdx_blue='#66B0EC'
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(country_data_exploration_dir,'chirps')
cod_ab_dir=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3,"cod_ab")
adm1_shp_path=os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3,"cod_ab",parameters["path_admin1_shp"])
```

### Read in forecast and observational data

```python
gdf_adm1=gpd.read_file(adm1_shp_path)
```

```python
da_for=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_for_clip=da_for.rio.set_spatial_dims(x_dim="longitude",y_dim="latitude").rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"], all_touched=True)
```

```python
#get the global dataset, such that we can clip it with a buffer
da_obs_glb=xr.open_dataset(config.CHIRPS_MONTHLY_RAW_PATH)
```

```python
#clip global data to bounding box around mwi
bb=gdf_adm1.total_bounds
buf=1
da_obs_mwi=da_obs_glb.rio.write_crs("EPSG:4326").rio.clip_box(bb[0]-buf,bb[1]-buf,bb[2]+buf,bb[3]+buf).precip
```

```python
g=da_obs_mwi.sel(time="2020-01-01").plot()
gdf_adm1.boundary.plot(ax=g.axes)
```

```python
#downsample observational data to be the same resolution as the forecast
#note: interpolation to downsample is probably not the best method
#optimally we would like to take the mean across the cells to the new coordinates
#but we couldn't find a way how to (xarray's coarsen only takes a frequency of coordinates)
da_obs_interp=da_obs_mwi.interp(latitude=da_for["latitude"],longitude=da_for["longitude"],method="linear")
da_obs_clip=da_obs_interp.rio.clip(gdf_adm1["geometry"],all_touched=True)
```

```python
g=da_obs_clip.sel(time="2020-01-01").plot(cmap="YlOrRd")
gdf_adm1.boundary.plot(ax=g.axes,color="grey")
```

```python
da_for_clip=da_for.rio.write_crs("EPSG:4326").rio.clip(gdf_adm1["geometry"],all_touched=True)
```

```python
g=da_for_clip.sel(time="2020-01-01",leadtime=2,number=10).plot()
gdf_adm1.boundary.plot(ax=g.axes);
```

Small check to see if the data looks as expected.

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-01-01'

rf_list_slice = da_for_clip.sel(time=start,latitude=da_for_clip.latitude.values[3],longitude=da_for_clip.longitude.values[2])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

```python
#only include times that are present in both datasets (assuming no missing data)
da_for_clip=da_for_clip.sel(time=slice(da_obs_clip.time.min(), da_obs_clip.time.max()))
da_obs_clip=da_obs_clip.sel(time=slice(da_for_clip.time.min(), da_for_clip.time.max()))
```

```python
#takes a minute to compute, due to taking the median
df_obs=da_obs_clip.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forec_all=da_for_clip.to_dataframe(name="precip").reset_index()
df_forobs_all=df_forec_all.merge(df_obs,how="left",on=["time","latitude","longitude"],suffixes=("_for","_obs"))
df_forobs_all["diff_forobs"]=df_forobs_all["precip_for"]-df_forobs_all["precip_obs"]
df_forobs_all["month"]=df_forobs_all.time.dt.month
```

```python
#take median of ensemble members
df_forobs_med=df_forobs_all.groupby(["time","leadtime","latitude","longitude"],as_index=False).median()
df_forobs_med.drop(["number","spatial_ref"],inplace=True,axis=1)
```

```python
fig, axes = plt.subplots(1,3,figsize=(20,6),sharey=True)
sns.boxplot(data=df_forobs_all,x="month",y="precip_obs",ax=axes[0],color=hdx_blue)
sns.boxplot(data=df_forobs_med,x="month",y="precip_for",ax=axes[1],color=hdx_blue)
sns.boxplot(data=df_forobs_all,x="month",y="precip_for",ax=axes[2],color=hdx_blue)

axes[0].set_title("Observed")
axes[1].set_title("Forecasted - median")
axes[2].set_title("Forecasted - all ensemble members")

for ax in axes:
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_xlabel("Month number")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelleft=True)
fig.suptitle("Ranges of precipitation for different months");
```

```python
leadtimes=df_forobs_all.month.unique()
num_plots = len(leadtimes)
colp_num=4
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,12))
for i, l in enumerate(leadtimes):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.boxplot(data=df_forobs_all[df_forobs_all.month==l],x="leadtime",y="precip_for", ax=ax, color=hdx_blue)
    ax.set_title(f"Month number = {l}")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlabel("Leadtime")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
fig.suptitle("Foreceasted ranges of precipitation for each month across leadtimes");
fig.tight_layout()
```

From the below plot we can see that the range of forecasted values differs per year. Many of the observed values are not close to the medians, and sometimes even outside the whole range of the ensemble members. 

```python
#plot the range of forecasted values and the observed values
lt_sel=4
df_forec_lt = df_forec_all[df_forec_all['leadtime'] == lt_sel]
df_forec_lt = df_forec_lt[df_forec_lt.time.dt.year>1993].groupby(['number', 'time']).mean().reset_index()
df_obs = df_obs[df_obs.time.dt.year>1993].groupby(['time']).mean().reset_index()
# #uncomment if want to use one cell instead of mean across country
# lat_min, lat_max = -13, -12
# lon_min, lon_max = 33, 34
# df_forec_lt = df_forec_lt[(df_forec_lt['latitude'] < lat_max) & (df_forec_lt['latitude'] > lat_min) &
#                     (df_forec_lt['longitude']>lon_min) & (df_forec_lt['longitude'] < lon_max)]
# df_obs = df_obs[(df_obs['latitude'] < lat_max) & (df_obs['latitude'] > lat_min) &
#                     (df_obs['longitude']>lon_min) & (df_obs['longitude'] < lon_max)]
df_forec_lt['year'] = (df_forec_lt.time.dt.year).astype(str)
df_obs['year'] = (df_obs.time.dt.year).astype(str)
sel_m=[1,2]
num_plots = len(sel_m)
colp_num=2
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,8))
    
for i,m in enumerate(sel_m):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_obs_m=df_obs[df_obs.time.dt.month==m]
    df_forec_lt_m=df_forec_lt[df_forec_lt.time.dt.month==m]
    ax.plot(df_obs_m.year, df_obs_m.precip, 'ro',label="observed")
    g=sns.boxplot(data=df_forec_lt_m,x="year",y="precip",ax=ax,color=hdx_blue, whis=[0,100],labels=["forecasted range"])
    handles, labels = g.get_legend_handles_labels()
    ax.set_title(calendar.month_name[m])

    ax.set_ylabel("Monthly precipitation")
    ax.set_xlabel("Year")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelleft=True)
    ax.tick_params(axis='x', labelrotation=80)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=hdx_blue, lw=4))
    labels.append("Forecasted ensemble range")
fig.suptitle(f"Ranges of precipitation for leadtime {lt_sel}");
fig.legend(handles,labels,bbox_to_anchor=(1.1, 0.9));
```

From the plot below we can see that the range of ensemble members differs per leadtime when inspecting specific dates. Most of the time it is consistent across leadtimes whether the forecasts are under or overpredicting the observation

```python
df_forec_mean = df_forec_all[df_forec_all.time.dt.year>1993].groupby(['number', 'time','leadtime']).mean().reset_index()
df_obs_mean = df_obs[df_obs.time.dt.year>1993].groupby(['time']).mean().reset_index()
df_forec_mean['year'] = (df_forec_mean.time.dt.year).astype(str)
df_obs_mean['year'] = (df_obs_mean.time.dt.year).astype(str)

num_plots = len(range(2005,2011))
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,8))
    
for i,m in enumerate(range(2005,2011)):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_obs_m=df_obs_mean[(df_obs_mean.time.dt.month==1)&(df_obs_mean.year==str(m))]
    df_forec_m=df_forec_mean[(df_forec_mean.time.dt.month==1)&(df_forec_mean.year==str(m))]
    sns.boxplot(data=df_forec_m,x="leadtime",y="precip",ax=ax,color=hdx_blue, whis=[0,100],labels=["forecasted range"])
    plt.axhline(y=df_obs_m.precip.values, color='r', linestyle='-',label="observed")
    ax.set_title(m)

    ax.set_ylabel("Monthly precipitation")
    ax.set_xlabel("Leadtime")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelleft=True)
    ax.tick_params(axis='x', labelrotation=80)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=hdx_blue, lw=4))
    labels.append("Forecasted ensemble range")
fig.suptitle(f"Ranges of precipitation for January in different years across leadtimes");
fig.tight_layout()
fig.legend(handles,labels,bbox_to_anchor=(1.2, 0.9));

```

#### Compute the Continuous Ranked Probability Score (CRPS)

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. More information on the CRPS can for example be found [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjJoPGs44DyAhUlJMUKHadUDogQFjAAegQIBxAD&url=https%3A%2F%2Fconfluence.ecmwf.int%2Fdownload%2Fattachments%2F50042306%2Fhandout_v2.pdf%3Fversion%3D1%26modificationDate%3D1441745383382%26api%3Dv2&usg=AOvVaw0oiem39qkz_Yv_3LvbBnld)

```python
thresh_list=[210,180,170]
sel_m=[[11,12,1,2,3,4],[1,2]]
norm_meth=[None,"mean"]
n_str_dict={None:"","mean":"n"}
leadtimes = da_for_clip.leadtime.values
df_crps = pd.DataFrame(index=leadtimes)
for n in norm_meth:
    n_str=n_str_dict[n]
    df_crps_all=processing.get_crps_ecmwf(da_obs_clip,
                                          da_for_clip,
                                          normalization=n
                                         ).rename(columns={"crps":f"{n_str}crps"})
    df_crps=pd.concat([df_crps,df_crps_all],axis=1)
    for thresh in thresh_list:
        df_crps_th=processing.get_crps_ecmwf(da_obs_clip,
                                             da_for_clip,
                                             normalization=n,
                                             thresh=thresh
                                            ).rename(columns={"crps":f"{n_str}crps_{thresh}"})
        df_crps=pd.concat([df_crps,df_crps_th],axis=1)
    for m in sel_m:
        month_str="".join([calendar.month_abbr[i].lower() for i in m])
        da_obs_clip_m = da_obs_clip.where(da_obs_clip.time.dt.month.isin(m), drop=True)
        da_for_clip_m = da_for_clip.where(da_for_clip.time.dt.month.isin(m), drop=True)
        df_crps_m=processing.get_crps_ecmwf(da_obs_clip_m,
                                            da_for_clip_m,
                                            normalization = n,
                                           ).rename(columns={"crps":f"{n_str}crps_{month_str}"})
        df_crps=pd.concat([df_crps,df_crps_m],axis=1)
        for thresh in thresh_list:
            df_crps_m=processing.get_crps_ecmwf(da_obs_clip_m,
                                                da_for_clip_m,
                                                thresh=thresh,
                                                normalization = n,
                                               ).rename(columns={"crps":f"{n_str}crps_{month_str}_{thresh}"})
            df_crps=pd.concat([df_crps,df_crps_m],axis=1)
```

From the plots below we can see that
- The CRPS is pretty high, resulting in a 30% error across all values
- The CRPS is relatively steady across leadtimes. We can see a dip at 1 month leadtime, and a smaller dip at 4 months leadtime
- The CRPS differs across months and thresholds
- For the months around the rainy season (Nov-Apr), we can see that the CRPS is relatively high, but the normalized CRPS is lower. This is largely due to the fact that the precipitation during the rainy season is higher. 
    - While lower during Nov-Apr, and specifically JanFeb, the error is still 25% which is quite large. 
- When selecting on the precipitation threshold, we can see that the higher the threshold the lower the normalized CRPS. 
    - This is likely because there are many very low values in the data, which causes the normalized CRPS to get large quickly even though the absolute difference might be small

```python
fig, axes = plt.subplots(1,2,figsize=(20,8))
months_col = ["crps","crps_novdecjanfebmarapr","crps_janfeb"]
for c in months_col:
    if "_" in c:
        label= f"{c.split('_')[-1]}"
    else:
        label="all"
    axes[0].plot(df_crps.index, df_crps[c], label=label)
    axes[1].plot(df_crps.index, df_crps[f"n{c}"]*100, label=label)
axes[0].set_title("CRPS")
axes[0].set_ylabel("CRPS [mm]")
axes[1].set_title("Normalized CRPS")
axes[1].set_ylabel("Normalized CRPS [% error]")

for ax in axes:
    ax.set_xlabel("Leadtime [months]")
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

fig.suptitle("CRPS for different months")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

```python
fig, axes = plt.subplots(1,2,figsize=(20,8))
thresh_col = ["crps","crps_210","crps_180","crps_170"]
for c in thresh_col:
    if "_" in c:
        label= f"<={c.split('_')[-1]}"
    else:
        label="all"
    axes[0].plot(df_crps.index, df_crps[c], label=label)
    axes[1].plot(df_crps.index, df_crps[f"n{c}"]*100, label=label)
axes[0].set_title("CRPS")
axes[0].set_ylabel("CRPS [mm]")
axes[1].set_title("Normalized CRPS")
axes[1].set_ylabel("Normalized CRPS [% error]")

for ax in axes:
    ax.set_xlabel("Leadtime [months]")
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.suptitle("CRPS for different thresholds")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

For the use of the forecast to predict dry spells, we are especially interested in the months of January and February, which have relatively low values. We therefore inspect the normalized crps for values which are not greater than 210 and 170 mm for these two months. It is important to note that the statistical significance becomes smaller here since we have less data points that meet these criteria. 

Nevertheleess, we can see that the results are not very promising. When only looking at January and February, the normalized CRPS becomes higher the lower we set the threshold. While the NCRPS for Jan and Feb was relatively lower compared to all months if we didn't set this threshold, it is significantly larger if we set the threshold. 

Moreover, the NCRPS for <=210 and <=170 mm during Jan and Feb is larger than across all months, indicating extra difficulty of prediction during those months

```python
fig, axes = plt.subplots(1,2,figsize=(20,8))
# axes[1].plot(df_crps["ncrps"]*100,label="all")

axes[1].plot(df_crps["ncrps_janfeb_210"]*100,label="janfeb and <=210 mm")
axes[1].plot(df_crps["ncrps_janfeb_170"]*100,label="janfeb and <=170 mm")
axes[1].plot(df_crps["ncrps_janfeb"]*100,label="janfeb")

axes[1].set_title("Normalized CRPS")
axes[1].set_ylabel("Normalized CRPS [% error]")

# axes[0].plot(df_crps["crps"],label="all")

axes[0].plot(df_crps["crps_janfeb_210"],label="janfeb and <=210 mm")
axes[0].plot(df_crps["crps_janfeb_170"],label="janfeb and <=170 mm")
axes[0].plot(df_crps["crps_janfeb"],label="janfeb")

axes[0].set_title("CRPS")
axes[0].set_ylabel("CRPS [mm]")

for ax in axes:
    ax.set_xlabel("Leadtime [months]")
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

fig.suptitle("CRPS for JanFeb with different thresholds")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

### Compute the bias
While the CRPS gives a good indication of the skill across ensemble members, we want to understand better which direction the error has and whether it differs across ranges of precipitation.

To do so we aggregate the ensemble members to one number, for which we chose to use the median. 

We firstly look at the distribution of observed and forecasted values. Thereafter we compute the bias.

Often the bias is computed using the MPE. However, for very small values the MPE is not suitable as it disproportionally explodes. This is also the case for our data. We therefore instead solely focus on the difference between forecasted and observed values, instead of looking at the percentual difference. 


Below we plot the range of observed and forecasted values per month, but across all leadtimes. We can see that they both show the same pattern across the months. However, the observed data shows a broader range of values, and especially has more outliers. The medians are relatively in the same range for observed and forecasted values, though when zooming in you can see that they also differ. 


Next we look at the bias. We firstly plot the observed vs forecasted-observed values across all leadtimes, dates, and cells. 
From this we can see that

- Most months with very low precipitation were correctly classified. 
- Months with less than 300mm have the tendency to be overpredicted, i.e. we see a positive bias
- Months with more than 300mm have the tendency to be underpredicted, i.e. we see a negative bias

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs_med,y="diff_forobs",x="precip_obs", kind="hex",height=10,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_med.precip_obs.max()+20,10)
group = df_forobs_med.groupby(pd.cut(df_forobs_med.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted - Observed monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.fig.suptitle("Bias plot of observed vs forecasted values")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
```

```python
#plot the observed vs forecast
g=sns.jointplot(data=df_forobs_med,y="precip_for",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_med.precip_obs.max()+20,10)
group = df_forobs_med.groupby(pd.cut(df_forobs_med.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.precip_for.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.fig.suptitle("Plot of observed vs forecasted values")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
```

Since our main months of interest are January and February, we zoom in on these months. 
From here we can see that the bias is a lot higher for these months compared to all months. Again for values up to 300 the forecast has a tendency to overpredict and for higher values to underpredict. 

It is hard to say whether this increased bias is due to the period or the range of precipitation, while these are heavily intertwined. 

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
df_forobs_perc_selm=df_forobs_perc[(df_forobs_perc.time.dt.month.isin([1,2]))&(df_forobs_perc.leadtime==2)]
g=sns.jointplot(data=df_forobs_perc_selm,y="diff_forobs",x="precip_obs", kind="hex",height=10)
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_perc_selm.precip_obs.max()+20,20)
group = df_forobs_perc_selm.groupby(pd.cut(df_forobs_perc_selm.precip_obs, bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group.diff_forobs.median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted - Observed monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.fig.suptitle("Bias plot of observed vs forecasted values for January and February")
g.fig.subplots_adjust(top=0.95)
```

Experiment with using a differen probability/percentile instead of 50% (=median)

```python
#take median of ensemble members
df_forobs_perc=df_forobs_all.groupby(["time","leadtime","latitude","longitude"])[["precip_for","precip_obs","diff_forobs"]].quantile([0.1,0.25,0.5]).unstack().reset_index()
df_forobs_perc.columns = [f"{x}_{y}" if len(str(y))>0 else x for x, y in df_forobs_perc.columns  ]
```

```python
quant=0.5
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
df_forobs_perc_selm=df_forobs_perc[(df_forobs_perc.time.dt.month.isin([1,2]))]
g=sns.jointplot(data=df_forobs_perc_selm,y=f"diff_forobs_{quant}",x=f"precip_obs_{quant}", kind="hex",height=10,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_perc_selm[f"precip_obs_{quant}"].max()+20,20)
group = df_forobs_perc_selm.groupby(pd.cut(df_forobs_perc_selm[f"precip_obs_{quant}"], bins))
plot_centers = (bins [:-1] + bins [1:])/2
plot_values = group[f"diff_forobs_{quant}"].median()
g.ax_joint.plot(plot_centers,plot_values,color="#C25048",label="median")
g.set_axis_labels("Observed monthly precipitation (mm)", "Forecasted - Observed monthly precipitation (mm)", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
g.ax_joint.legend()
g.ax_joint.set_ylim(-400,280)
g.fig.suptitle("Bias plot of observed vs forecasted values for January and February")
g.fig.subplots_adjust(top=0.95)
```

Above we looked at the bias across all leadtimes and per cell. We now take the median of the error across all cells and months, but separated by leadtime. We can see that across all values the error is not that large, which was expected from the plots above


Questions: 
- better use mean or median to compute the difference between observation and forecast? As they give quite different results 
- Does using the CI make sense or better use percentiles? Due to the large number of values the CI becomes small (which can be a good thing), but at the same time the std is quite large

```python
def calc_diff_stats(observations,forecast):
    diff=forecast-observations
    diff_mean=diff.mean()
    diff_median=diff.median()
    diff_std=diff.std()
    diff_count=diff.count()
    ci95_hi = diff_mean + 1.95*diff_std/math.sqrt(diff_count)
    ci95_lo = diff_mean - 1.95*diff_std/math.sqrt(diff_count)
    return diff_median, diff_mean, diff_std, ci95_hi, ci95_lo

def compute_diff_cats(df,obs_col,for_col,threshold_list=None,adm_list=None,month_list=None):
    leadtimes=df.leadtime.sort_values().unique()
    df_diff=pd.DataFrame(leadtimes,columns=["leadtime"]).set_index('leadtime')
    for ilt, leadtime in enumerate(leadtimes):
        df_lt=df[df.leadtime==leadtime]
        med,me,std,ci_hi,ci_lo=calc_diff_stats(df_lt[obs_col],df_lt[for_col])
        df_diff.loc[leadtime,"all_med"]=med
        df_diff.loc[leadtime,"all_mean"]=me
        df_diff.loc[leadtime,"all_ci_lo"]=ci_lo
        df_diff.loc[leadtime,"all_ci_hi"]=ci_hi
        if threshold_list:
            for thresh in threshold_list:
                df_lt_thresh=df_lt[df_lt[obs_col]<=thresh]
                med,me,std,ci_hi,ci_lo = calc_diff_stats(df_lt_thresh[obs_col],df_lt_thresh[for_col])
                df_diff.loc[leadtime,f"thresh_{thresh}_med"]=med
        if month_list:
            for m in month_list:
                month_str="".join([calendar.month_abbr[i].lower() for i in m])
                df_lt_m=df_lt[df_lt.time.dt.month.isin(m)]
                med,me,std,ci_hi,ci_lo = calc_diff_stats(df_lt_m[obs_col],df_lt_m[for_col])
                df_diff.loc[leadtime,f"{month_str}_med"]=med
                df_diff.loc[leadtime,f"{month_str}_ci_lo"]=ci_lo
                df_diff.loc[leadtime,f"{month_str}_ci_hi"]=ci_hi
                
    return df_diff
```

```python
df_diff=compute_diff_cats(df_forobs_med,"precip_obs","precip_for",threshold_list=[210,180,170],month_list=[[m] for m in range(1,13)])
```

```python
fig, ax = plt.subplots()
ax.plot(df_diff.index,df_diff["all_mean"],label="mean")
ax.fill_between(df_diff.index, df_diff.all_ci_lo, df_diff.all_ci_hi,alpha=0.2,label="confidence interval")
ax.plot(df_diff.index, df_diff["all_med"],label="median")
ax.set_xlabel("leadtime [months]")
ax.set_ylabel("median bias")
ax.grid()
ax.legend(bbox_to_anchor=(1.05, 1));
```

When looking per month though, we can see that the average bias clearly differs per month. The months around the rainy season have the highest bias (nov, dec, jan, feb). This can either be because values are generally larger, or because the forecast has less skill. 

```python
px.line(df_diff.reset_index(), 
        x='leadtime', 
        y=[c for c in df_diff.columns if "thresh" not in c and "med" in c],
        labels={"value": "median bias"}
       )
```

We can also look at the bias for different thresholds. Here we see that there is not much difference. This is however partly caused by the fact that most months have very low values. If you would separate by month and threshold, you would probably get different patterns

```python
px.line(df_diff.reset_index(), 
        x='leadtime', 
        y=[c for c in df_diff.columns if "thresh" in c and "med" in c]+["all_med"],
        labels={"value": "median bias"}
       )
```

The values we saw in the CRPS plots are a lot larger than for the median bias plots. The bias can be lower because it sums both negative and positive errors, which can cancel each other out. With the CRPS the errors are squared.
