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
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
crps_path=os.path.join(monthly_precip_exploration_dir,"mwi_crps.csv")
```

### Read in forecast and observational data

```python
da_lt=processing.get_ecmwf_forecast_by_leadtime("mwi")
```

```python
da_obs=xr.load_dataset(chirps_monthly_mwi_path)
da_obs=da_obs.precip
```

```python
#interpolate forecast data such that it has the same resolution as the observed values
#using "nearest" as interpolation method and not "linear" because the forecasts are designed to have sharp edged and not be smoothed
da_forecast=da_lt.interp(latitude=da_obs["latitude"],longitude=da_obs["longitude"],method="nearest")
```

Small check to see if the data looks as expected.

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-01-01'

rf_list_slice = da_lt.sel(time=start,latitude=da_lt.latitude.values[10],longitude=da_lt.longitude.values[5])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

Question:
- Is there a method to already show the distribution of values here? 
    - We do this now after aggregating the ensemble members. Simply because without aggregation there are so many values that the code crashes when doing a histogram/boxplot


#### Compute the Continuous Ranked Probability Score (CRPS)

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. More information on the CRPS can for example be found [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjJoPGs44DyAhUlJMUKHadUDogQFjAAegQIBxAD&url=https%3A%2F%2Fconfluence.ecmwf.int%2Fdownload%2Fattachments%2F50042306%2Fhandout_v2.pdf%3Fversion%3D1%26modificationDate%3D1441745383382%26api%3Dv2&usg=AOvVaw0oiem39qkz_Yv_3LvbBnld)

```python
#only include times that are present in both datasets (assuming no missing data)
da_forecast=da_forecast.sel(time=slice(da_obs.time.min(), da_obs.time.max()))
da_obs=da_obs.sel(time=slice(da_forecast.time.min(), da_forecast.time.max()))
```

```python
# #only need to run if not updated data/categories. Takes about 15 minutes
# thresh_list=[210,180,170]
# sel_m=[[11,12,1,2,3,4],[1,2]]
# norm_meth=[None,"mean"]
# n_str_dict={None:"","mean":"n"}
# leadtimes = da_forecast.leadtime.values
# df_crps = pd.DataFrame(index=leadtimes)
# for n in norm_meth:
#     n_str=n_str_dict[n]
#     df_crps_all=processing.get_crps_ecmwf(da_obs,
#                                           da_forecast,
#                                           normalization=n
#                                          ).rename(columns={"crps":f"{n_str}crps"})
#     df_crps=pd.concat([df_crps,df_crps_all],axis=1)
#     for thresh in thresh_list:
#         df_crps_th=processing.get_crps_ecmwf(da_obs,
#                                              da_forecast,
#                                              normalization=n,
#                                              thresh=thresh
#                                             ).rename(columns={"crps":f"{n_str}crps_{thresh}"})
#         df_crps=pd.concat([df_crps,df_crps_th],axis=1)
#     for m in sel_m:
#         month_str="".join([calendar.month_abbr[i].lower() for i in m])
#         da_obs_m = da_obs.where(da_obs.time.dt.month.isin(m), drop=True)
#         da_forecast_m = da_forecast.where(da_forecast.time.dt.month.isin(m), drop=True)
#         df_crps_m=processing.get_crps_ecmwf(da_obs_m,
#                                             da_forecast_m,
#                                             normalization = n,
#                                            ).rename(columns={"crps":f"{n_str}crps_{month_str}"})
#         df_crps=pd.concat([df_crps,df_crps_m],axis=1)
#         for thresh in thresh_list:
#             df_crps_m=processing.get_crps_ecmwf(da_obs_m,
#                                                 da_forecast_m,
#                                                 thresh=thresh,
#                                                 normalization = n,
#                                                ).rename(columns={"crps":f"{n_str}crps_{month_str}_{thresh}"})
#             df_crps=pd.concat([df_crps,df_crps_m],axis=1)
# df_crps.to_csv(crps_path)
```

```python
df_crps=pd.read_csv(crps_path)
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

fig.suptitle("CRPS for different thresholds")
fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.9));
```

For the use of the forecast to predict dry spells, we are especially interested in the months of January and February, which have relatively low values. We therefore inspect the normalized crps for values which are not greater than 210 and 170 mm for these two months. It is important to note that the statistical significance becomes smaller here since we have less data points that meet these criteria. 

Nevertheleess, we can see that the results are not very promising. When only looking at January and February, the normalized CRPS becomes higher the lower we set the threshold. While the NCRPS for Jan and Feb was relatively lower compared to all months if we didn't set this threshold, it is significantly larger if we set the threshold. 

Moreover, the NCRPS for <=210 and <=170 mm during Jan and Feb is larger than across all months, indicating extra difficulty of prediction during those months

```python
fig,ax=plt.subplots()
ax.plot(df_crps["ncrps_janfeb_210"]*100,label="janfeb and <=210 mm")
ax.plot(df_crps["ncrps_janfeb_170"]*100,label="janfeb and <=170 mm")
ax.plot(df_crps["ncrps_janfeb"]*100,label="janfeb")
ax.plot(df_crps["ncrps"]*100,label="all")
ax.set_title("Normalized CRPS")
ax.set_ylabel("Normalized CRPS [% error]")
ax.set_xlabel("Leadtime [months]")
ax.grid()
ax.legend(bbox_to_anchor=(1.05,1))
```

### Compute the bias
While the CRPS gives a good indication of the skill across ensemble members, we want to understand better which direction the error has and whether it differs across ranges of precipitation.

To do so we aggregate the ensemble members to one number, for which we chose to use the median. 

We firstly look at the distribution of observed and forecasted values. Thereafter we compute the bias.

Often the bias is computed using the MPE. However, for very small values the MPE is not suitable as it disproportionally explodes. This is also the case for our data. We therefore instead solely focus on the difference between forecasted and observed values, instead of looking at the percentual difference. 

```python
#takes a minute to compute, due to taking the median
df_obs=da_obs.to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forec=da_forecast.median(dim="number").to_dataframe(name="precip").reset_index().drop("spatial_ref",axis=1)
df_forobs=df_forec.merge(df_obs,how="left",on=["time","latitude","longitude"],suffixes=("_for","_obs"))
df_forobs["diff_forobs"]=df_forobs["precip_for"]-df_forobs["precip_obs"]
df_forobs["month"]=df_forobs.time.dt.month
```

Below we plot the range of observed and forecasted values per month, but across all leadtimes. We can see that they both show the same pattern across the months. However, the observed data shows a broader range of values, and especially has more outliers. The medians are relatively in the same range for observed and forecasted values, though when zooming in you can see that they also differ. 

```python
fig, axes = plt.subplots(1,2,figsize=(20,8),sharey=True)
sns.boxplot(data=df_forobs,x="month",y="precip_obs",ax=axes[0],color=hdx_blue)
sns.boxplot(data=df_forobs,x="month",y="precip_for",ax=axes[1],color=hdx_blue)
axes[0].set_title("Observed")
axes[1].set_title("Forecasted")

for ax in axes:
    ax.set_ylabel("Monthly precipitation")
    ax.set_xlabel("Month number")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', labelleft=True)
fig.suptitle("Ranges of precipitation for different months");
```

Next we look at the bias. We firstly plot the observed vs forecasted-observed values across all leadtimes, dates, and cells. 
From this we can see that

- Most months with very low precipitation were correctly classified. 
- Months with less than 300mm have the tendency to be overpredicted, i.e. we see a positive bias
- Months with more than 300mm have the tendency to be underpredicted, i.e. we see a negative bias

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
g=sns.jointplot(data=df_forobs,y="diff_forobs",x="precip_obs", kind="hex",height=16,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs.precip_obs.max()+20,10)
group = df_forobs.groupby(pd.cut(df_forobs.precip_obs, bins))
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

Since our main months of interest are January and February, we zoom in on these months. 
From here we can see that the bias is a lot higher for these months compared to all months. Again for values up to 300 the forecast has a tendency to overpredict and for higher values to underpredict. 

It is hard to say whether this increased bias is due to the period or the range of precipitation, while these are heavily intertwined. 

```python
#plot the observed vs forecast-observed to get a feeling for the discrepancy between the two
df_forobs_selm=df_forobs[(df_forobs.time.dt.month.isin([1,2]))]
g=sns.jointplot(data=df_forobs_selm,y="diff_forobs",x="precip_obs", kind="hex",height=10,joint_kws={ 'bins':'log'})
#compute the average value of the difference between the forecasted and observed values
#do this in bins cause else very noisy mean
bins = np.arange(0,df_forobs_selm.precip_obs.max()+20,10)
group = df_forobs_selm.groupby(pd.cut(df_forobs_selm.precip_obs, bins))
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
df_diff=compute_diff_cats(df_forobs,"precip_obs","precip_for",threshold_list=[210,180,170],month_list=[[m] for m in range(1,13)])
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

Questions:
- The values we saw in the CRPS plots are a lot larger than for the median bias plots. Why is this the case? 
    - a median bias of around 6 across all values, compared to a CRPS of around 25
    - the mean bias is around 15 but this is still lower than the CRPS

```python

```
