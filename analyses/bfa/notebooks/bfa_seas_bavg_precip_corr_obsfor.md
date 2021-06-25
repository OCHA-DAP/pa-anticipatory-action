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

# Correlation of forecasted and observed lower tercile precipitation
This notebook explores the correlation between observed and forecasted below average precipitation. 
The area of interest for the pilot are 4 admin1 areas: Boucle de Mounhoun, Centre Nord, Sahel, and Nord. Therefore this analysis is mainly focussed on those areas

The proposal contains two triggers as outlined below. We therefore mainly focus on the stated seasons (=3month period). However, for a part of the analysis we include all seasons due to only having forecast data of the last 4 years. 

We solely experiment with the 40% threshold, as the 50% threshold was never met during the last 4 years. 
- Trigger #1 in March covering June-July-August. Threshold desired: 40%.
- Trigger #2 in July covering Aug-Sep-Oct. Threshold desired: 50%.

The forecasts are separately analyzed in `bfa_seas_bavg_iriforecast.md`, while the observed precipitation is being analyzed in `bfa_seas_bavg_precip_observed.md`


Based on the separate analysis of the forecast, the proposed trigger is that at least 10% of the area has a 40% probability of below average precipitation AND has a 5 percentage points higher probability than the above average tercile. For the observed precipitaiton, it was shown that 40% of the area experiencing below average precipitation is an event that happens 1 in 3 seasons (it is co-incidence that for the forecast and observed values we both use 40%). We therefore define this as a "true observed" event. 

These definitons are used to compute binary metrics, but we also analyze the distributions more broadly to understand if there might be other thresholds that would result in a better correlation. 

Resources
- [CHC's Early Warning Explorer](https://chc-ewx2.chc.ucsb.edu) is a nice resource to scroll through historically observed CHIRPS data

+++

## Correlate the observations with forecasts
Now that we have analyzed the observational data, we can investigate the correspondence between observed and forecasted values.  
With the forecasts there is an extra variable, namely the minimum probability of below average rainfall. Since a part of the trigger is based on this being {glue:text}`threshold_for_prob`%, this is also the value used in this analysis.

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
iri_exploration_dir=os.path.join(country_data_exploration_dir,"iri")
stats_reg_for_path=os.path.join(iri_exploration_dir,f"{country}_iri_seasonal_forecast_stats_{''.join(adm_sel_str)}.csv")
chirps_exploration_dir=os.path.join(country_data_exploration_dir,"chirps")
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

```{code-cell} ipython3
leadtime_mar=3
leadtime_jul=1
```

```{code-cell} ipython3
:tags: [remove_cell]

#load forecast data, computed in `bfa_iriforecast.md`
df_for=pd.read_csv(stats_reg_for_path,parse_dates=["F"])
def get_forecastmonth(pub_month,leadtime):
    return pub_month+relativedelta(months=+int(leadtime))
df_for["for_start"]=df_for.apply(lambda x: get_forecastmonth(x.F,x.L), axis=1)
df_for["for_start_month"]=df_for.for_start.dt.to_period("M")
df_for["for_end_month"]=df_for.apply(lambda x: get_forecastmonth(x.for_start,2), axis=1)
```

```{code-cell} ipython3
:tags: [remove_cell]

#only select values for below average rainfall
df_for_bavg=df_for[df_for.C==0]
```

```{code-cell} ipython3
#load the observed data
df_obs_bavg=pd.read_csv(stats_reg_observed_path,parse_dates=["start_month","end_month"])
df_obs_bavg["start_month"]=df_obs_bavg.start_month.dt.to_period("M")
df_obs_bavg["end_month"]=df_obs_bavg.end_month.dt.to_period("M")
```

```{code-cell} ipython3
:tags: [remove_cell]

#merge observed and forecasted
df_obsfor=df_obs_bavg.merge(df_for_bavg,left_on="start_month",right_on="for_start_month",suffixes=("_obs","_for"))
#add season mapping
df_obsfor["season"]=df_obsfor.for_end_month.apply(lambda x:month_season_mapping[x.month])
#used for plotting
df_obsfor["seas_year"]=df_obsfor.apply(lambda x: f"{x.season} {x.for_end_month.year}",axis=1)
```

```{code-cell} ipython3
:tags: [remove_cell]

df_obsfor_mar=df_obsfor[df_obsfor.L==leadtime_mar].dropna()
df_obsfor_jul=df_obsfor[df_obsfor.L==leadtime_jul].dropna()
```

```{code-cell} ipython3
:tags: [remove_cell]

df_obsfor_mar.head()
```

Show the values for the months that are included in the framework

```{code-cell} ipython3
df_obsfor_mar[df_obsfor_mar.for_start_month.dt.month==6][["for_start","40th_bavg_cell","bavg_cell"]]
```

```{code-cell} ipython3
df_obsfor_jul[df_obsfor_jul.for_start_month.dt.month==8][["for_start","40th_bavg_cell","bavg_cell"]]
```

As first comparison we can make a density plot of the area forecasted to have >=40% probability of below average precipitaiton, and the percentage of the area that observed below average precipitation.  We here focus onn the forecasts with a leadtime of 3 months, but this can easily be repeated for other leadtimes

As the plot below shows, these results are not very promissing. Only in a few seasons there was a >=40% probability of below average precipitation, and in most of those seasons, the percentage of the area that also observed the below average precipitation was relatively low.

For some months the rainfall is really low due to the dry season, this results in very small ranges between the terciles. It might therefore not be correct to treat all seasons similarly when computing the correlation. However due to the limited data this is the only method we have.

```{code-cell} ipython3
:tags: [remove_cell]

glue("pears_corr", df_obsfor_mar.corr().loc["bavg_cell","40th_bavg_cell"])
```

```{code-cell} ipython3
:tags: [remove_cell]

glue("pears_corr", df_obsfor_jul.corr().loc["bavg_cell","40th_bavg_cell"])
```

We can also capture the relation between the two variables in one number by looking at the Pearson correlation. This is found to be {glue:text}`pears_corr:.2f`. This indicates a weak and even negative correlation, which is the opposite from what we would expect.

```{code-cell} ipython3
:tags: [hide_input]

 #plot the observed vs forecast-observed for obs<=2mm
g=sns.jointplot(data=df_obsfor_mar,x="bavg_cell",y=f"{threshold_for_prob}percth_cell", kind="hex",height=12,marginal_kws=dict(bins=100),joint_kws=dict(gridsize=30),xlim=(0,100))
g.set_axis_labels("Percentage of area observed below average precipitation", "% of area forecasted >=40% probability below average", fontsize=12)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
cbar_ax = g.fig.add_axes([.85, .25, .05, .4])
plt.colorbar(cax=cbar_ax);
```

It would be possible that the observed and forecasted below average seasons don't fully overlap, but are close in time. To understand this better, we create a bar plot showing the percentage of the area with below average precipitation for the forecasted and observed values. We can see from here that the overlap is poor. 

Note:
1) the forecasted percentage, is the percentage of the area where the probability of below average >=40 & the difference between below and above average is at least 5 percentage points
2) some seasons are not included due to the dry mask defined by IRI (the dry mask is further explained [here](https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/methodology/)

```{code-cell} ipython3
:tags: [hide_input]

fig, ax = plt.subplots(figsize=(12,6))
tidy = df_obsfor_mar[["seas_year","for_start","40percth_cell","bavg_cell"]].rename(columns={"40percth_cell":"forecasted","bavg_cell":"observed"}).melt(id_vars=['for_start','seas_year'],var_name="data_source").sort_values("for_start")
tidy.rename(columns={"40percth_cell":"forecasted","bavg_cell":"observed"},inplace=True)
sns.barplot(x='seas_year', y='value', data=tidy, ax=ax,hue="data_source",palette={"observed":"#CCE5F9","forecasted":'#F2645A'})
sns.despine(fig)
x_dates = tidy.seas_year.unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right');
ax.set_ylabel("Percentage of area")
ax.set_ylim(0,100)
ax.set_title("Percentage of area meeting criteria for observed and forecasted below average precipitation");
```

```{code-cell} ipython3
:tags: [remove_cell]

for_thresh=10
occ_num=len(df_obsfor_mar[df_obsfor_mar["40th_bavg_cell"]>=for_thresh])
occ_perc=occ_num/len(df_obsfor_mar)*100
glue("for_thresh", for_thresh)
glue("occ_num", occ_num)
glue("occ_perc", occ_perc)
```

Despite the bad correlation, we do a bit further exploration to see if the geographical spread of the forecasted and observed below average area matters. 
Since it occurs so rarely that any part of the area is forecasted to have >=40% probability of below average rainfall, we define the forecast as meeting the criterium if at least {glue:text}`for_thresh`% of the area meets the 40% threshold. This occurred {glue:text}`occ_num` times between Apr 2017 and May 2021 (={glue:text}`occ_perc:.2f`% of the forecasts).

We then experiment with different thresholds of the area that had observed below average precipitation. As can be seen, with any threshold the miss and false alarm rate are really high, showing bad detection. 

Note: these numbers are not at all statistically significant!!

```{code-cell} ipython3
:tags: [remove_cell]

threshold_area_list=[1,5,50,20,35,40,45,43,for_thresh,60]
for t in threshold_area_list:
    df_obsfor_mar[f"obs_bavg_{t}"]=np.where(df_obsfor_mar.bavg_cell>=t,1,0)
    df_obsfor_mar[f"for_bavg_{t}"]=np.where(df_obsfor_mar["40th_bavg_cell"]>=t,1,0)
    df_obsfor_jul[f"obs_bavg_{t}"]=np.where(df_obsfor_jul.bavg_cell>=t,1,0)
    df_obsfor_jul[f"for_bavg_{t}"]=np.where(df_obsfor_jul["40th_bavg_cell"]>=t,1,0)
```

```{code-cell} ipython3
:tags: [remove_cell]

#compute tp,tn,fp,fn per threshold
y_predicted = np.where(df_obsfor_mar["40th_bavg_cell"]>=for_thresh,1,0)
threshold_list=np.arange(0,df_obsfor_mar.bavg_cell.max() +6,5)
df_pr_th=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_target = np.where(df_obsfor_mar.bavg_cell>=t,1,0)
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    #fn=not forecasted bavg but was observed
    tn,fp,fn,tp=cm.flatten()
    df_pr_th.loc[t,["month_miss_rate","month_false_alarm_rate"]]=fn/(tp+fn+0.00001)*100,fp/(tp+fp+0.00001)*100
    df_pr_th.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_th=df_pr_th.reset_index()
```

```{code-cell} ipython3
:tags: [hide_input]

fig,ax=plt.subplots()

df_pr_th.plot(x="threshold",y="month_miss_rate" ,figsize=(12, 6), color='#F2645A',legend=False,ax=ax,style='.-',label="bel.avg. rainfall was observed but not forecasted (misses)")
df_pr_th.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(12, 6), color='#66B0EC',legend=False,ax=ax,style='.-',label="no bel.avg. rainfall observed, but this was forecasted (false alarms)")

ax.set_xlabel("Observed below average area (%)", labelpad=20, weight='bold', size=20)
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=20)
sns.despine(bottom=True,left=True)
ax.set_ylim(0,100)

# Draw vertical axis lines
vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
```

We can see that the forecasts never correspond with an occurrence of observed below average precipitation, regardless of the threshold that is set. However, since we only have 4 years of data, these patterns are not at all statistically significant. 

However, to understand a bit better when extreme events of below average precipitation occur, we compute the confusion matrix per month as well as across all months. For this we set the threshold to 40% of the area having observed below average precipitaiton, since this is the 1 in 3 year return value. 

+++

Note: these numbers are not at all statistically significant!!

```{code-cell} ipython3
:tags: [remove_cell]

def compute_confusionmatrix(df,target_var,predict_var, ylabel,xlabel,col_var=None,colp_num=3,title=None,figsize=(20,15)):
    #number of dates with observed dry spell overlapping with forecasted per month
    if col_var is not None:
        num_plots = len(df[col_var].unique())
    else:
        num_plots=1
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=figsize)
    if col_var is not None:
        for i, m in enumerate(df.sort_values(by=col_var)[col_var].unique()):
            ax = fig.add_subplot(rows,colp_num,i+1)
            y_target =    df.loc[df[col_var]==m,target_var]
            y_predicted = df.loc[df[col_var]==m,predict_var]
            cm = confusion_matrix(y_target=y_target, 
                                  y_predicted=y_predicted)

            plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{col_var}={m}")
    else:
        ax = fig.add_subplot(rows,colp_num,1)
        y_target =    df[target_var]
        y_predicted = df[predict_var]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
```

```{code-cell} ipython3
:tags: [hide_input]

cm_thresh=compute_confusionmatrix(df_obsfor_mar,f"obs_bavg_40",f"for_bavg_10","Observed bel. avg. rainfall","Forecasted bel. avg. rainfall",figsize=(5,5))
```

```{code-cell} ipython3
:tags: [hide_input]

cm_thresh=compute_confusionmatrix(df_obsfor_jul,f"obs_bavg_40",f"for_bavg_10","Observed bel. avg. rainfall","Forecasted bel. avg. rainfall",figsize=(5,5))
```

```{code-cell} ipython3
:tags: [hide_input]

#divide by season to see if there is a pattern, but too limited data
cm_thresh=compute_confusionmatrix(df_obsfor_mar,f"obs_bavg_50",f"for_bavg_10","Observed bel. avg. rainfall","Forecasted bel. avg. rainfall",col_var="season",colp_num=5)
```

#### Metrics 
- To compute metrics we have to binarise the observed values, I set it as an event if at least 40% of the area saw below average precipitation, since this is a 1 in 3 year event.
- While we would optimally only focus on the publication months that are included in the trigger, we decided to also compute the metrics across all seasons due to the very limited data availability. 
- Across all months, the accuracy is pretty bad.. With a leadtime of 3 months, the trigger would have been met 5 times, and with a leadtime of 1 month only 2 times. However, all those occurrences didn’t correspond with a 1 in 3 year event (which were 9 in total). So there were 9 misses and 5 or 2 false alarms.  (experimented with lower observed numbers but then it doesn’t get better)
- For Mar, the trigger was only met in 2017, and in that season 7% of the area had observed bel avg rainfall
- For Jul, the trigger was never met. In 2017, 24% of the area observed bel avg precipitation, in the other 3 years this was 0%.


+++

## Conclusion

+++

The forecasted and observed values don't show a great overlap for our threshold and area of interest.    
One limitation of these numbers is the low statistical significance due to very limited data availability.   
If we want to continue understanding the suitability of this trigger, we therefore might want to look for ideas on how we could make them statistically significant. One idea could be to do the analysis at the raster cell level instead of aggregating to the area of interst.

+++ {"tags": ["remove_cell"]}

## Archive

```{code-cell} ipython3
:tags: [remove_cell]

#inspect consistency forecasts
fig,ax=plt.subplots(figsize=(10,10))
sns.lineplot(data=df_obsfor, x="for_start", y="40th_bavg_cell", hue="L",ax=ax)
# sns.lineplot(data=df_obsfor, x="time",y="bavg_cell",ax=ax,linestyle="--",marker="o")
```

```{code-cell} ipython3
:tags: [remove_cell]

#inspect distribution observed depending on whether trigger was met
#note: way less datapoints where trigger was met --> not fair method of comparison
df_obsfor["for_trigger_met"]=np.where(df_obsfor["40th_bavg_cell"]>=10,1,0)
df_obsfor[f"max_cell_{threshold_for_prob}"]=np.where(df_obsfor.max_cell_for>=threshold_for_prob,1,0)
#plot distribution precipitation with and without observed belowavg precip
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_obsfor[df_obsfor.C==0],x="L",y="bavg_cell",ax=ax,color="#66B0EC",hue="for_trigger_met",palette={0:"#CCE5F9",1:'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Leadtime")
ax.get_legend().set_title("Trigger met")
```

```{code-cell} ipython3

```
