### Evaluating the forecast skill of ECMWF seasonal forecast per admin1 region of Malawi
This notebook is assesses the forecast skill of ECMWF's seasonal forecast per admin1 region of Malawi at different lead times. We use the monthly total precipitation that is forecasted by this forecast. 
We compare the forecasted and observed values visually. Thereafter we compute the bias, and more specifically look at the areas, leadtimes, and months that are of interest for the trigger.  

The notebook `07_ecmwf_monthly_skill_general.md` investigates the skill for separate raster cells, while `09_ecmwf_monthly_skill_dryspells` evaluates the skill specifically focussed on dry spells

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
import glob
import itertools

import math
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
from src.utils_general.statistics import calc_mpe
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
```

```python
import plotly
import plotly.graph_objects as go
```

#### Set config values

```python
use_incorrect_area_coords = False
interpolate=False
```

```python
country_iso3="mwi"
config=Config()
parameters = config.parameters(country_iso3)

country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / country_iso3

monthly_precip_path=Path(country_data_processed_dir) / "chirps" / "chirps_monthly_total_precipitation_admin1.csv"
```

```python
#using the mean value of the admin
if use_incorrect_area_coords:
    aggr_meth="mean_cell"
else:
    aggr_meth = "mean_ADM1_PCODE"
```

```python
#set plot colors
ds_color='#F2645A'
no_ds_color='#CCE5F9'
no_ds_color_dark='#66B0EC'
```

### Define functions

```python
def calc_mpe(observations,forecast):
    return (((forecast-observations)/observations).sum()) / len(observations) * 100

def compute_mpe_cats(df,obs_col,for_col,leadtime_list,threshold_list=None,adm_list=None,month_list=None):
    df_mpe=pd.DataFrame(leadtime_list,columns=["leadtime"]).set_index('leadtime')
    for ilt, leadtime in enumerate(df.leadtime.sort_values().unique()):
        df_lt=df[df.leadtime==leadtime]
        df_mpe.loc[leadtime,"all"]=calc_mpe(df_lt[obs_col],df_lt[for_col])
        df_mpe.loc[leadtime,"all_count"]=len(df_lt)
        if threshold_list:
            for thresh in threshold_list:
                df_lt_thresh=df_lt[df_lt[obs_col]<=thresh]
                df_mpe.loc[leadtime,f"<={thresh}mm"]=calc_mpe(df_lt_thresh[obs_col],df_lt_thresh[for_col])
                df_mpe.loc[leadtime,f"<={thresh}mm_count"]=len(df_lt_thresh)
        if adm_list:
            for adm in adm_list:
                df_lt_adm=df_lt[df_lt.ADM1_EN==adm]
                df_mpe.loc[leadtime,f"{adm.lower()}"]=calc_mpe(df_lt_adm[obs_col],df_lt_adm[for_col])
                df_mpe.loc[leadtime,f"{adm.lower()}_count"]=len(df_lt_adm)
        if month_list:
            for m in month_list:
                df_lt_m=df_lt[df_lt.date_month.dt.month==m]
                df_mpe.loc[leadtime,calendar.month_abbr[m].lower()]=calc_mpe(df_lt_m[obs_col],df_lt_m[for_col])
                df_mpe.loc[leadtime,f"{calendar.month_abbr[m].lower()}_count"]=len(df_lt_m)
                
    return df_mpe
```

```python
def calc_ci(df_grouped):
    stats = df_grouped.agg(['mean', 'count', 'std','min','max'])
    ci95_hi = []
    ci95_lo = []

    for i in stats.index:
        m, c, s, mi, ma = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats=stats.reset_index()
    return stats
```

### Load data

```python
#define selection parameters that are used in part of the analysis
#these parameters are based on the end goal, namely forecasting dry spelsl
sel_adm=["Southern"]
sel_months=[1,2]
sel_leadtime=[1,2,3,4,5,6]
#for some plots we can only show one leadtime, set that here
sel_lt_plt=4
start_year=2000
end_year=2020
#just locking the date to keep the analysis the same even though data is added
#might wanna delete again later
end_date="5-1-2021"
seas_years=range(start_year,end_year)

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])
```

#### Forecast
For this notebook we directly load the statistics per admin1 and therafter aggregate across the ensemble members. These stats are computed with `src/mwi/get_ecmwf_seasonal_data.py` 
In other notebooks we explored the skill per raster cell and ensemble member. 
Note: The statistics over the whole admin region per ensemble member were first computed, after which we combine the ensemble models with different percentile thresholds. While we think this methodology makes sense, one could also argue to first group by the ensemble members and then aggregating to the admin. This was also tested and no large differences were found. 

```python
#read the ecmwf forecast per adm1 per date and concat all dates
# the mwi_seasonal-monthly-single-levels_v5_interp*.csv contain results when interpolating the forecasts to be more granular
# but results actually worsen with this
date_list=pd.date_range(start=f'1-1-{start_year}', end=end_date, freq='MS')
all_files=[processing.get_stats_filepath(country_iso3,config,date,interpolate=interpolate,adm_level=1,use_incorrect_area_coords=use_incorrect_area_coords) for date in date_list]

df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
df_for   = pd.concat(df_from_each_file, ignore_index=True)
```

```python
#set the value to the 50% percentile across all ensemble members. This is further explained in 08_ecmwf_monthly_skill_dryspells.md
probability=0.5
#for earlier dates, the model included less members --> values for those members are nan --> remove those rows
df_for = df_for[df_for[aggr_meth].notna()]
#start month of the rainy season
start_rainy_seas=10
#season approx indicates the year during which the rainy season started
#this is done because it can start during one year and continue the next calendar year
#we therefore prefer to group by rainy season instead of by calendar year
df_for["season_approx"]=np.where(df_for.date.dt.month>=start_rainy_seas,df_for.date.dt.year,df_for.date.dt.year-1)
```

```python
#aggregate across ensemble members by taking the 50% probability
df_for_quant=df_for.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(probability)
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
```

#### Observational

```python
#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
# df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=start_rainy_seas,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)
```

#### Merge the two datasets

```python
stat_col_forec = f"{aggr_meth}_forec"
df_for_quant.rename(columns={aggr_meth:stat_col_forec},inplace=True)
stat_col_obs = "mean_cell_obs"
df_obs_month.rename(columns={"mean_cell":stat_col_obs},inplace=True)
```

```python
#merge forecast and observed
df_obsfor=df_for_quant.merge(df_obs_month,how="left",on=["date_month","ADM1_EN"],suffixes=("_forec","_obs"))
```

```python
df_obsfor["diff_forecobs"]=df_obsfor[stat_col_forec]-df_obsfor[stat_col_obs]
```

Create one df with only the admins, months, and leadtimes of interest for the trigger. This is further explained in `08_ecmwf_monthly_skill_dryspells.md`

```python
df_obsfor_sel=df_obsfor[(df_obsfor.ADM1_EN.isin(sel_adm))&(df_obsfor.date_month.dt.month.isin(sel_months))&(df_obsfor.leadtime.isin(sel_leadtime))&(df_obsfor.season_approx.isin(seas_years))]
```

### Analysis

```python
def compute_confusionmatrix_leadtime(df,target_var,predict_var, ylabel,xlabel,colp_num=3,title=None):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df.leadtime.unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(15,8))
    for i, m in enumerate(df.sort_values(by="leadtime").leadtime.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df.leadtime==m,target_var]
        y_predicted = df.loc[df.leadtime==m,predict_var]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax,class_names=["No","Yes"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Leadtime={m}")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
```

```python
thresh_obs=210
thresh_for=210
lt_rp=4
month_rp=2
```

```python
df_obsfor_lt=df_obsfor_sel[(df_obsfor_sel.leadtime==lt_rp)&(df_obsfor_sel.date_forec.dt.month==month_rp)]
print(f"Occurrences observed <={thresh_obs} in {calendar.month_name[month_rp]}: "
      f"{len(df_obsfor_lt[df_obsfor_lt.mean_cell_obs<=thresh_obs])} (of {len(df_obsfor_lt)}={len(df_obsfor_lt[df_obsfor_lt.mean_cell_obs<=thresh_obs])/len(df_obsfor_lt)*100:.2f}%)")
print(f"Occurrences forecasted <={thresh_obs} in {calendar.month_name[month_rp]} with leadtime {lt_rp}: "
      f"{len(df_obsfor_lt[df_obsfor_lt[stat_col_forec]<=thresh_for])} (of {len(df_obsfor_lt)}={len(df_obsfor_lt[df_obsfor_lt[stat_col_forec]<=thresh_for])/len(df_obsfor_lt)*100:.2f}%)")
```

```python
df_obsfor[f"mean_cell_obs_{thresh_obs}"]=np.where(df_obsfor.mean_cell_obs<=thresh_obs,1,0)
df_obsfor[f"mean_cell_forec_{thresh_for}"]=np.where(df_obsfor[stat_col_forec]<=thresh_for,1,0)
df_obsfor_sel[f"mean_cell_obs_{thresh_obs}"]=np.where(df_obsfor_sel.mean_cell_obs<=thresh_obs,1,0)
df_obsfor_sel[f"mean_cell_forec_{thresh_for}"]=np.where(df_obsfor_sel[stat_col_forec]<=thresh_for,1,0)
cm_th=compute_confusionmatrix_leadtime(df_obsfor_sel,f"mean_cell_obs_{thresh_obs}",f"mean_cell_forec_{thresh_for}",f"Observed <={thresh_obs}",f"Forecasted <={thresh_for}",title=f"Confusion matrices of below threshold monthly precipitation during January and February in the Southern region of Malawi")
```

```python
df_for_sel_plot
```

```python

df_for_sel_plot=df_for[(df_for.leadtime==sel_lt_plt)&(df_for.ADM1_EN.isin(sel_adm))]
df_obs_sel_plot=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))]

df_for_perc50=df_for_sel_plot.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(0.5)
df_for_perc25=df_for_sel_plot.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(0.25)
df_for_perc75=df_for_sel_plot.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(0.75)
fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(
    x=df_for_perc50.date, 
    y=df_for_perc50[aggr_meth], 
    name='Forecasted median',
    line=dict(color='firebrick', width=4)
))
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=df_for_perc75.date,
    y=df_for_perc75[aggr_meth],
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Forecasted 25-75 percentile',
    x=df_for_perc25.date,
    y=df_for_perc25[aggr_meth],
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty',
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=df_obs_sel_plot.date, 
    y=df_obs_sel_plot[stat_col_obs], 
    name = 'Observed',
    line=dict(color='royalblue', width=4)
))



fig.update_layout(
    title=f"Forecasted and observed precipitation in the Southern region with {sel_lt_plt} months leadtime",
    xaxis_title="Year",
    yaxis_title="Monthly precipitation (mm)",
)
```

From the above graph we can see that the forecasts generally follow the trend quite well. However it is often wrong around dec-mar. During this period it predicts relatively smooth transitions from one month to another while the observed patterns are more spikey. This is problematic for our goal of predicting dry spells, where we are especially interested in the abnormally low months


While above we inspected the values for one leadtime, we are also interested in the range of forecasted values across all leadtimes, which is shown below. In this case the grey area doesn't indicate the confidence interval across ensemble members, but instead the min and max forecasted 50% values across all leadtimes

```python
df_for_ci_lt=calc_ci(df_for_quant.groupby(['ADM1_EN','date'])[stat_col_forec])
stats_lt_sel_plot=df_for_ci_lt[df_for_ci_lt.ADM1_EN.isin(sel_adm)]
df_obs_sel_plot=df_obs_month[df_obs_month.ADM1_EN.isin(sel_adm)]
fig = go.Figure()
# Create and style traces

fig.add_trace(go.Scatter(
    name='Minimum',
    x=stats_lt_sel_plot.date,
    y=stats_lt_sel_plot['min'],
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    name='Min-Max leadtimes',
    x=stats_lt_sel_plot.date,
    y=stats_lt_sel_plot['max'],
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty',
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=df_obs_sel_plot.date, 
    y=df_obs_sel_plot[stat_col_obs], 
    name = 'Observed',
    line=dict(color='royalblue', width=4)
))
fig.add_trace(go.Scatter(
    x=stats_lt_sel_plot.date, 
    y=stats_lt_sel_plot["mean"], 
    name='Forecasted mean',
    line=dict(color='firebrick', width=4)
))

fig.update_layout(
    title=f"Forecasted and observed precipitation in the Southern region across leadtimes",
    xaxis_title="Year",
    yaxis_title="Monthly precipitation (mm)",
)
```

From the above graph we can see that forecasted values differ across leadtimes, especially around the rainy season. However in several years during none of the leadtimes the observed values were forecasted. Moreover, it should be noted that often the forecast with 1 month leadtimes "dares" to forecast relatively more extreme values. 

This is not surprising but less useful for our use case as the 1 month leadtime is only published mid-way the month it is forecasting and is thus providing very little real leadtime for our purposes. 

```python
def plot_mpe(df_mpe,title=None):
    fig, ax = plt.subplots()
    for c in df_mpe.columns:
        if not "count" in c:
            ax.plot(list(df_mpe.index), df_mpe[c],label=f"{c} (n={int(df_mpe.loc[1,f'{c}_count'])})")
    ax.axhline(y=0, c='k', ls=':')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Leadtime [months]')
    ax.set_ylabel('% bias')
    ax.legend(bbox_to_anchor=(1.05, 1))
    if title:
        ax.set_title(title)

leadtime_list=range(1,7)
threshold_list=[210,180,170]
```

#### Compute bias
We compute the bias by looking at the Mean Percentage Error (MPE). We separate by different criteria to better understand the factors contributing to the error. 
Since the MPE can explode when looking at very small values, we only select months that had at least 20mm of rain. 


From the below graphs we can see that
- The bias across all data points is pretty large, around 30%
- The bias across all data points doesn't depend much on the leadtime, surprisingly only getting smaller as leadtime increases
- The bias is positive, meaning that the forecast is on average overpredicting
- The lower the maximum cap, the larger the bias. This is bad news as we are especially interesting in those months. However, this worse performance might also be caused by the percentual nature of the MPE. 
- The bias is the lowest for the Southern region, which is also the region of interest. 
- The bias depends on the month, but is during most months of the rainy season lower than across the whole year. 

```python
df_mpe_thresh=compute_mpe_cats(df_obsfor[df_obsfor[stat_col_forec]>=20],stat_col_obs,stat_col_forec,leadtime_list,threshold_list)
plot_mpe(df_mpe_thresh,title="Bias for different thresholds")
```

```python
df_mpe_adm=compute_mpe_cats(df_obsfor[df_obsfor[stat_col_forec]>=20],stat_col_obs,stat_col_forec,leadtime_list,adm_list=list(df_obsfor.ADM1_EN.unique()))
plot_mpe(df_mpe_adm,title="Bias for different admins")
```

```python
df_mpe_adm=compute_mpe_cats(df_obsfor[df_obsfor[stat_col_forec]>=20],stat_col_obs,stat_col_forec,leadtime_list,adm_list=list(df_obsfor.ADM1_EN.unique()))
plot_mpe(df_mpe_adm,title="Bias for different admins")
```

```python
df_mpe_month=compute_mpe_cats(df_obsfor[df_obsfor[stat_col_forec]>=20],stat_col_obs,stat_col_forec,leadtime_list,month_list=[11,12,1,2,3])
plot_mpe(df_mpe_month,title="Bias for different months")
```

The main datapoints of interest are those in January and February in the Southern region. As we can see from the graph below, the bias for these points is not very promising. The bias is slighly lower with a leadtime of 1 and 4 months. Since the leadtime of 1 month is noot very useful for our purpose, the leadtime of 4 months might be the most interesting. 

Note however that there are not many data points here, which means that the statistical significance is not high. 

```python
df_mpe_sel=compute_mpe_cats(df_obsfor_sel,stat_col_obs,stat_col_forec,leadtime_list,threshold_list)
plot_mpe(df_mpe_sel,title="Bias for the data points evaluated in the trigger")
```
