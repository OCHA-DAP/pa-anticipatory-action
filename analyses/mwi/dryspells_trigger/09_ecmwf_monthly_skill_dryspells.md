### Evaluating the forecast skill of ECMWF seasonal forecast for dry spells in Malawi 
This notebook is assesses the forecast skill of ECMWF's seasonal forecast for dry spells at various lead times. We use the monthly total precipitation that is forecasted by this forecast.    
The goal of this analysis is to determine the suitability of the monthly forecast to be used as a trigger for dry spells.    

To determine the skill for dry spells, two parameters have to be set. Namely the cap of forecasted mm of precipitation during the month, and the probability of the precipitation being below this cap. This notebook explores several combinations of these two parameters.   

Note that due to the small number of historically observed dry spells, the statistical significance is low.    

The dry spells are determined in [this script](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/1589debf38eee928d323414e254f7d811d577108/analyses/mwi/scripts/mwi_chirps_dry_spell_detection.R) and ECMWF's forecast can be processed by [this script](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/e7ad2ac3a250912b713ab55fe45ed995d944ffc7/analyses/malawi/notebooks/read_in_data.py). [This notebook](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/mwi-corranalys/analyses/malawi/notebooks/mwi_monthlytotal_corr_dryspells_adm1.md) explores the relation between observed monthly precipitation and observed dry spells

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
reload(processing)

mpl.rcParams['figure.dpi'] = 200
pd.options.mode.chained_assignment = None
font = {'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
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

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")

monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells", f"v{parameters['version']}", "monthly_precipitation")
dry_spells_processed_dir = os.path.join(country_data_processed_dir, "dry_spells", f"v{parameters['version']}")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(dry_spells_processed_dir,"full_list_dry_spells_2000_2021.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","chirps_monthly_total_precipitation_admin1.csv")
```

```python
#using the mean value of the admin
if use_incorrect_area_coords:
    aggr_meth="mean_cell"
    plots_seasonal_dir = Path(plots_seasonal_dir) / "unrounded-coords"
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
def det_rate(tp,fn,epsilon):
    return tp/(tp+fn+epsilon)*100
def tn_rate(tn,fp,epsilon):
    return tn/(tn+fp+epsilon)*100
def miss_rate(fn,tp,epsilon):
    return fn/(tp+fn+epsilon)*100
def false_alarm_rate(fp,tp,epsilon):
    return fp/(tp+fp+epsilon)*100

def compute_miss_false_leadtime(df,target_var,predict_var):
    df_pr=pd.DataFrame(list(df.leadtime.unique()),columns=["leadtime"]).set_index('leadtime')
    #TODO: also account for different adm1's
    a="Southern"
    for i, m in enumerate(df.sort_values(by="leadtime").leadtime.unique()):
        y_target =    df.loc[df.leadtime==m,target_var]
        y_predicted = df.loc[df.leadtime==m,predict_var]
        
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        tn,fp,fn,tp=cm.flatten()
        df_pr.loc[m,["month_ds"]] = det_rate(tp,fn,epsilon)
        df_pr.loc[m,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
        df_pr.loc[m,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
        df_pr.loc[m,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
        df_pr.loc[m,["tn","tp","fp","fn"]]=tn,tp,fp,fn
    df_pr=df_pr.reset_index()
    return df_pr
```

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
def label_ds(row,obs_col,for_col):
    if row[obs_col]==1 and row[for_col]==1:
        return 3
    elif row[obs_col]==1:
        return 2
    elif row[for_col]==1:
        return 1
    else:
        return 0
    
def refactor_data_hm(df,obs_col,for_col):
    #we have a R script to plot heatmaps nicely, so refactor the df to the expected format of that R script
    #convert monthly dates to dateranges with an entry per day
    df["first_date"]=pd.to_datetime(df.date)
    df["last_date"]=df.date.dt.to_period("M").dt.to_timestamp("M")

    df_obs=df[df[obs_col]==1]
    df_obs_res=df_obs.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_obs_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_obs_daterange=df_obs_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_obs_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed below threshold monthly precipitation, so add that information
    df_obs_daterange[obs_col]=1

    df_for=df[df[for_col]==1]
    df_for_res=df_for.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_for_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_for_daterange=df_for_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_for_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an forecasted below threshold monthly precipitation, so add that information
    df_for_daterange[for_col]=1

    #merge the observed and forecasted daterange
    df_daterange_comb=df_obs_daterange.merge(df_for_daterange,on=["date","pcode"],how="outer")

    df_alldatesrange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df.pcode.unique())),columns=['date','pcode'])
    df_alldatesrange_sel=df_alldatesrange[df_alldatesrange.date.dt.month.isin(sel_months)]

    df_daterange_comb=df_daterange_comb.merge(df_alldatesrange_sel,on=["date","pcode"],how="right")

    df_daterange_comb[obs_col]=df_daterange_comb[obs_col].replace(np.nan,0)
    df_daterange_comb[for_col]=df_daterange_comb[for_col].replace(np.nan,0)

    #encode dry spells and whether it was none, only observed, only forecasted, or both
    #R visualization code is expecting the column "dryspell_match"
    df_daterange_comb["dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row,obs_col,for_col),axis=1)
    
    return df_daterange_comb
```

```python
def load_dryspell_data(ds_path,min_ds_days_month=7,min_adm_ds_month=3,ds_adm_col="pcode",shp_adm_col="ADM1_EN",ds_date_cols=["dry_spell_first_date","dry_spell_last_date"]):
    df_ds_all=pd.read_csv(ds_path,parse_dates=ds_date_cols)
    
    #get list of all dates that were part of a dry spell
    df_ds_res=df_ds_all.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_res[ds_date_cols].values]
    #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
    df_ds_daterange=df_ds_res[[ds_adm_col]].join(pd.DataFrame(a)).set_index([ds_adm_col]).stack().droplevel(-1).reset_index()
    df_ds_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed dry spell, so add that information
    df_ds_daterange["dryspell_obs"]=1
    df_ds_daterange["date_month"]=df_ds_daterange.date.dt.to_period("M")
    
    #count the number of days within a year-month combination that had were part of a dry spell
    df_ds_countmonth=df_ds_daterange.groupby([ds_adm_col,"date_month"],as_index=False).sum()
    
    df_ds_month=df_ds_countmonth[df_ds_countmonth.dryspell_obs>=min_ds_days_month]
    
    #TODO: this is not really generalizable
    if shp_adm_col not in df_ds_month.columns:
        df_adm2=gpd.read_file(adm2_bound_path)
        df_ds_month=df_ds_month.merge(df_adm2[["ADM2_PCODE","ADM2_EN","ADM1_EN"]],left_on=ds_adm_col,right_on="ADM2_PCODE")
        
    df_ds_month_adm1=df_ds_month.groupby([shp_adm_col,"date_month"],as_index=False).count()
    df_ds_month_adm1["dry_spell"]=np.where(df_ds_month_adm1.dryspell_obs>=min_adm_ds_month,1,0)
    
    return df_ds_month_adm1
```

### Load the forecast data
And select the data of interest


Note: the forecast data is an ensemble model. The statistics over the whole admin region per ensemble member were first computed, after which we combine the ensemble models with different percentile thresholds. While we think this methodology makes sense, one could also argue to first group by the ensemble members and then aggregating to the admin. This was also tested and no large differences were found. 

```python
start_year=2000
end_year=2020
#just locking the date to keep the analysis the same even though data is added
#might wanna delete again later
end_date="5-1-2021"
```

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
#this should be the number of years*number of months FORECASTED i.e. 
print(len(all_files))
```

```python
#number of years*months till latest forecast + 6
21*12+5+6
```

```python
#for now using mean cell as this requires one variable less to be set (else need to set percentage of cells)
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
sel_adm=["Southern"]
sel_months=[1,2]
sel_leadtime=[1,2,3,4,5,6]
seas_years=range(start_year,end_year)

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for this analysis we are only interested in the southern region during a few months that the dry spells have the biggest impact
df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.leadtime.isin(sel_leadtime))&(df_for.season_approx.isin(seas_years))]
```

### Load observational data

```python
#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=10,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)

#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
```

```python
df_ds=load_dryspell_data(all_dry_spells_list_path)
```

```python
#check that same number of months in observed and forecasted
print("number of months in forecasted data",len(df_for_sel.date.unique()))
print("number of months in observed data",len(df_obs_month_sel.date.unique()))
```

### set threshold based on x% probability

```python
probability=0.5
```

```python
#compute the value for which x% of members forecasts the precipitation to be below or equal to that value
df_for_quant=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(probability)
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
```

```python
df_for_quant[["date","leadtime",aggr_meth]].head()
```

```python
#plot the distribution of precipitation. Can clearly see that for leadtime=1 the values are more spread (but leadtime=1 is really the month that is currently occurring so less useful)
g = sns.FacetGrid(df_for_quant, height=5, col="leadtime",col_wrap=3)
g.map_dataframe(sns.histplot, aggr_meth,common_norm=False,kde=True,alpha=1,binwidth=10)

for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
```

```python
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_for=df_ds.merge(df_for_quant,how="right",on=["ADM1_EN","date_month"])
df_ds_for.loc[:,"dry_spell"]=df_ds_for.dry_spell.replace(np.nan,0).astype(int)
```

```python
df_ds_for["month"]=df_ds_for.date_month.dt.month
df_ds_for["month_name"]=df_ds_for.month.apply(lambda x: calendar.month_name[x])
df_ds_for["month_abbr"]=df_ds_for.month.apply(lambda x: calendar.month_abbr[x])
df_ds_for_labels=df_ds_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
```

```python
#for some reason facetgrid doesn't want to show the values if there is only one occurence (i.e. in January..)
g = sns.FacetGrid(df_ds_for_labels, height=5, col="leadtime",row="month_name",hue="dry_spell",palette={"no":no_ds_color,"yes":ds_color})
g.map_dataframe(sns.histplot, aggr_meth,common_norm=False,alpha=1,binwidth=10)

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
```

```python
#plot distribution precipitation with and withoud dry spell
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_ds_for_labels,x="leadtime",y=aggr_meth,ax=ax,hue="dry_spell",palette={"no":no_ds_color,"yes":ds_color})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Leadtime")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_boxplot_formonth_dsobs_perlt_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
df_ds_for_labels
```

```python
for m in df_ds_for_labels.month_name.unique():#plot distribution precipitation with and withoud dry spell
    df_ds_for_labels_m=df_ds_for_labels[(df_ds_for_labels.month_name==m)&(df_ds_for_labels.leadtime.isin([2,4]))]
    fig,ax=plt.subplots(figsize=(10,6))
    g=sns.boxplot(data=df_ds_for_labels_m,x="leadtime",y=aggr_meth,ax=ax,hue="dry_spell",palette={"no":no_ds_color,"yes":ds_color})
    ax.set_ylabel("Monthly precipitation")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Leadtime")
    ax.get_legend().set_title("Dry spell occurred")
    ax.set_title(f"Month={m}")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_boxplot_formonth_dsobs_perlt_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
#compute tp,tn,fp,fn per threshold
y_target =  df_ds_for.dry_spell
threshold_list=np.arange(0,df_ds_for[aggr_meth].max() - df_ds_for_labels[aggr_meth].max()%10,10)
df_pr_th=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
#to prevent division by zero
epsilon=0.00001
for t in threshold_list:
    y_predicted = np.where(df_ds_for[aggr_meth]<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    df_pr_th.loc[t,["month_ds"]]= det_rate(tp,fn,epsilon)
    df_pr_th.loc[t,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
    df_pr_th.loc[t,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
    df_pr_th.loc[t,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
    df_pr_th.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_th=df_pr_th.reset_index()
```

```python
fig,ax=plt.subplots()

df_pr_th.plot(x="threshold",y="month_ds" ,figsize=(16, 8), color=ds_color,style='.-',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_th.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color=no_ds_color_dark,style='.-',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold")

ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=20)
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=20)
ax.set_ylim(0,100)
sns.despine(bottom=True,left=True)

# Draw vertical axis lines
vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_dsobs_missfalse_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
fig,ax=plt.subplots()

df_pr_th.plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), color=ds_color,legend=False,ax=ax,style='.-',label="dry spell occurred and monthly precipitation above threshold (misses)")
df_pr_th.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), color=no_ds_color_dark,legend=False,ax=ax,style='.-',label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=20)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=20)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

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

```python
print("point of intersection")
#this is across all leadtimes, so might not be the best point for leadtimes of interest
df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1)
```

```python
pr_list=[]
threshold_list=np.arange(0,df_ds_for[aggr_meth].max() - df_ds_for[aggr_meth].max()%10,10)
unique_lt=df_ds_for.leadtime.unique()

for m in unique_lt:
    df_pr_perlt=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
    df_ds_for_lt=df_ds_for[df_ds_for.leadtime==m]
    y_target =  df_ds_for_lt.dry_spell
    
    for t in threshold_list:
        y_predicted = np.where(df_ds_for_lt[aggr_meth]<=t,1,0)

        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        df_pr_perlt.loc[t,["precision","recall","num_trig","detection_rate"]]=tp/(tp+fp+0.00001)*100,tp/(tp+fn)*100,tp+fp,tp/(tp+fn)*100
        df_pr_perlt.loc[t,["month_ds"]]= det_rate(tp,fn,epsilon)
        df_pr_perlt.loc[t,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
        df_pr_perlt.loc[t,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
        df_pr_perlt.loc[t,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
        df_pr_perlt.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
        df_pr_perlt.loc[t,"leadtime"]=m
    df_pr_perlt=df_pr_perlt.reset_index()
    pr_list.append(df_pr_perlt)
```

```python
df_pr_sep_lt=pd.concat(pr_list).sort_values(["leadtime","threshold"])
df_pr_sep_lt.threshold=df_pr_sep_lt.threshold.astype(int)
```

```python
#same but now also separated by month instead of only by leadtime
pr_list=[]
threshold_list=np.arange(0,df_ds_for[aggr_meth].max() - df_ds_for[aggr_meth].max()%10,10)
unique_lt=df_ds_for.leadtime.unique()
unique_months=df_ds_for.month_name.unique()

for l in unique_lt:
    for m in unique_months:
        df_pr_perlt_m=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
        df_ds_for_lt_m=df_ds_for[(df_ds_for.leadtime==l)&(df_ds_for.month_name==m)]
        y_target =  df_ds_for_lt_m.dry_spell

        for t in threshold_list:
            y_predicted = np.where(df_ds_for_lt_m[aggr_meth]<=t,1,0)

            cm = confusion_matrix(y_target=y_target, 
                                  y_predicted=y_predicted)
            tn,fp,fn,tp=cm.flatten()
            df_pr_perlt_m.loc[t,["precision","recall","num_trig","detection_rate"]]=tp/(tp+fp+0.00001)*100,tp/(tp+fn)*100,tp+fp,tp/(tp+fn)*100
            df_pr_perlt_m.loc[t,["month_ds"]]= det_rate(tp,fn,epsilon)
            df_pr_perlt_m.loc[t,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
            df_pr_perlt_m.loc[t,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
            df_pr_perlt_m.loc[t,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
            df_pr_perlt_m.loc[t,"num_dates"]=len(y_predicted)
            df_pr_perlt_m.loc[t,"perc_trig"]=df_pr_perlt_m.loc[t,"num_trig"]/df_pr_perlt_m.loc[t,"num_dates"]
            df_pr_perlt_m.loc[t,"rp"]=round(1/(df_pr_perlt_m.loc[t,"perc_trig"]+0.0001))
            df_pr_perlt_m.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
            df_pr_perlt_m.loc[t,"leadtime"]=int(l)
            df_pr_perlt_m.loc[t,"month"]=m
        df_pr_perlt_m=df_pr_perlt_m.reset_index()
        pr_list.append(df_pr_perlt_m)
df_pr_sep_lt_m=pd.concat(pr_list).sort_values(["leadtime","threshold"])
df_pr_sep_lt_m.threshold=df_pr_sep_lt_m.threshold.astype(int)
df_pr_sep_lt_m.leadtime=df_pr_sep_lt_m.leadtime.astype(int)
df_pr_sep_lt_m["detection_rate"]=df_pr_sep_lt_m.apply(lambda x: f"{math.ceil(x.month_ds)}% ({int(x.tp)}/{int(x.tp+x.fn)})",axis=1)
df_pr_sep_lt_m["false_alarm_rate"]=df_pr_sep_lt_m.apply(lambda x: f"{math.ceil(x.month_false_alarm_rate)}% ({int(x.fp)}/{int(x.tp+x.fp)})",axis=1)
df_pr_sep_lt_m["return_period"]=df_pr_sep_lt_m.apply(lambda x: f"1/{int(x.rp)} years ({int(x.num_trig)}/{int(x.num_dates)})",axis=1)
```

```python
df_pr_sep_lt_m_sel=df_pr_sep_lt_m[(df_pr_sep_lt_m.threshold>=160)&(df_pr_sep_lt_m.threshold<=220)&(df_pr_sep_lt_m.leadtime.isin([2,4]))][["month","threshold","leadtime","detection_rate","false_alarm_rate","return_period"]].sort_values(["month","leadtime","threshold"])
```

```python
df_pr_sep_lt_m_sel.head()
```

```python
# df_pr_sep_lt_m_sel=df_pr_sep_lt_m_sel.replace("1/10000 years (0/20)","-")
# df_pr_sep_lt_m_sel.to_csv(os.path.join(monthly_precip_exploration_dir,f"mwi_detect_falsealarm_thresholds_perc_{int(probability*100)}_{adm_str}_{month_str}.csv"),index=False)
```

```python
#compute the statistics per year. 
# It is only looked at if during any of the months in the year a dry spell was observed 
# and/or forecasted precipitation was below the threshold
# i.e. the months don't have to match
df_ds_for["year"]=df_ds_for.date_month.dt.year
pr_list=[]
threshold_list=np.arange(0,df_ds_for[aggr_meth].max() - df_ds_for[aggr_meth].max()%10,10)
unique_lt=df_ds_for.leadtime.unique()

for m in unique_lt:
    df_pr_year_perlt=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
    df_ds_for_lt=df_ds_for[df_ds_for.leadtime==m]
    
    
    for t in threshold_list:
        df_ds_for_lt["below_th"]=np.where(df_ds_for_lt[aggr_meth]<=t,1,0)
        df_ds_for_lt_year=df_ds_for_lt.groupby("year").max()
        y_target =  df_ds_for_lt_year.dry_spell
        y_predicted = df_ds_for_lt_year.below_th

        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        df_pr_year_perlt.loc[t,["precision","recall","num_trig","detection_rate"]]=tp/(tp+fp+0.00001)*100,tp/(tp+fn)*100,tp+fp,tp/(tp+fn)*100
        df_pr_year_perlt.loc[t,["month_ds"]]= det_rate(tp,fn,epsilon)
        df_pr_year_perlt.loc[t,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
        df_pr_year_perlt.loc[t,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
        df_pr_year_perlt.loc[t,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
        df_pr_year_perlt.loc[t,"num_dates"]=len(y_predicted)
        df_pr_year_perlt.loc[t,"perc_trig"]=df_pr_year_perlt.loc[t,"num_trig"]/df_pr_year_perlt.loc[t,"num_dates"]
        df_pr_year_perlt.loc[t,"rp"]=round(1/(df_pr_year_perlt.loc[t,"perc_trig"]+0.0001))
        df_pr_year_perlt.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
        df_pr_year_perlt.loc[t,"leadtime"]=m
    df_pr_year_perlt=df_pr_year_perlt.reset_index()
    pr_list.append(df_pr_year_perlt)
df_pr_year_sep_lt=pd.concat(pr_list).sort_values(["leadtime","threshold"])
df_pr_year_sep_lt.threshold=df_pr_year_sep_lt.threshold.astype(int)
df_pr_year_sep_lt.leadtime=df_pr_year_sep_lt.leadtime.astype(int)
df_pr_year_sep_lt["detection_rate"]=df_pr_year_sep_lt.apply(lambda x: f"{math.ceil(x.month_ds)}% ({int(x.tp)}/{int(x.tp+x.fn)})",axis=1)
df_pr_year_sep_lt["false_alarm_rate"]=df_pr_year_sep_lt.apply(lambda x: f"{math.ceil(x.month_false_alarm_rate)}% ({int(x.fp)}/{int(x.tp+x.fp)})",axis=1)
df_pr_year_sep_lt["return_period"]=df_pr_year_sep_lt.apply(lambda x: f"1/{int(x.rp)} years ({int(x.num_trig)}/{int(x.num_dates)})",axis=1)
```

```python
df_pr_year_sep_lt_sel=df_pr_year_sep_lt[(df_pr_year_sep_lt.threshold>=160)&(df_pr_year_sep_lt.threshold<=220)&(df_pr_year_sep_lt.leadtime.isin([2,4]))][["threshold","leadtime","detection_rate","false_alarm_rate","return_period"]].sort_values(["leadtime","threshold"])
```

```python
df_pr_year_sep_lt_sel
```

```python
# df_pr_year_sep_lt_sel.leadtime=df_pr_year_sep_lt_sel.leadtime-1
# df_pr_year_sep_lt_sel.to_csv(os.path.join(monthly_precip_exploration_dir,f"mwi_detect_falsealarm_thresholds_perc_{int(probability*100)}_{adm_str}_peryear.csv"),index=False)
```

```python
#same but now also separated by month instead of only by leadtime
pr_list=[]
threshold_list=np.arange(0,df_ds_for[aggr_meth].max() - df_ds_for[aggr_meth].max()%10,10)
unique_lt=df_ds_for.leadtime.unique()
unique_months=df_ds_for.month_name.unique()

for l in unique_lt:
    for m in unique_months:
        df_pr_perlt_m=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
        df_ds_for_lt_m=df_ds_for[(df_ds_for.leadtime==l)&(df_ds_for.month_name==m)]
        y_target =  df_ds_for_lt_m.dry_spell

        for t in threshold_list:
            y_predicted = np.where(df_ds_for_lt_m[aggr_meth]<=t,1,0)

            cm = confusion_matrix(y_target=y_target, 
                                  y_predicted=y_predicted)
            tn,fp,fn,tp=cm.flatten()
            df_pr_perlt_m.loc[t,["precision","recall","num_trig","detection_rate"]]=tp/(tp+fp+0.00001)*100,tp/(tp+fn)*100,tp+fp,tp/(tp+fn)*100
            df_pr_perlt_m.loc[t,["month_ds"]]= det_rate(tp,fn,epsilon)
            df_pr_perlt_m.loc[t,["month_no_ds"]]= tn_rate(tn,fp,epsilon)
            df_pr_perlt_m.loc[t,["month_miss_rate"]]= miss_rate(fn,tp,epsilon)
            df_pr_perlt_m.loc[t,["month_false_alarm_rate"]]= false_alarm_rate(fp,tp,epsilon)
            df_pr_perlt_m.loc[t,"num_dates"]=len(y_predicted)
            df_pr_perlt_m.loc[t,"perc_trig"]=df_pr_perlt_m.loc[t,"num_trig"]/df_pr_perlt_m.loc[t,"num_dates"]
            df_pr_perlt_m.loc[t,"rp"]=round(1/(df_pr_perlt_m.loc[t,"perc_trig"]+0.0001))
            df_pr_perlt_m.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
            df_pr_perlt_m.loc[t,"leadtime"]=int(l)
            df_pr_perlt_m.loc[t,"month"]=m
        df_pr_perlt_m=df_pr_perlt_m.reset_index()
        pr_list.append(df_pr_perlt_m)
df_pr_sep_lt_m=pd.concat(pr_list).sort_values(["leadtime","threshold"])
df_pr_sep_lt_m.threshold=df_pr_sep_lt_m.threshold.astype(int)
df_pr_sep_lt_m.leadtime=df_pr_sep_lt_m.leadtime.astype(int)
df_pr_sep_lt_m["detection_rate"]=df_pr_sep_lt_m.apply(lambda x: f"{math.ceil(x.month_ds)}% ({int(x.tp)}/{int(x.tp+x.fn)})",axis=1)
df_pr_sep_lt_m["false_alarm_rate"]=df_pr_sep_lt_m.apply(lambda x: f"{math.ceil(x.month_false_alarm_rate)}% ({int(x.fp)}/{int(x.tp+x.fp)})",axis=1)
df_pr_sep_lt_m["return_period"]=df_pr_sep_lt_m.apply(lambda x: f"1/{int(x.rp)} years ({int(x.num_trig)}/{int(x.num_dates)})",axis=1)
```

#### Set threshold and compute performance
Now that we have a rough feel of the performance across thresholds, we can set a threshold and more thoroughly inspect the performance for that threshold. 
In first instance, the threshold was set by using the intersection point between the percentage of hits, and months with no dry spell and no precipitation <=threshold. 

However, this intersection point is computed across all leadtimes. After closer inspection per leadtime, it was decided to also manually test thresholds. Of which 210 and 180 mm give relatively the best performance (still poor) for leadtimes 2 and 4, depending on whether precision or recall is optimized. 

```python
#threshold based on intersection point of the two lines
# threshold_perc=df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1).threshold.values[0]
#for easily testing different thresholds
threshold_perc=210 #180
```

```python
threshold_perc
```

```python
df_ds_for["for_below_th"]=np.where(df_ds_for[aggr_meth]<=threshold_perc,1,0)
```

```python
# df_ds_for[["ADM1_EN","date_month","pcode","leadtime","dry_spell","for_below_th"]].to_csv(os.path.join(monthly_precip_exploration_dir,f"mwi_list_dsobs_forblw_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{month_str}.csv"),index=False)
```

```python
df_pr_ds=compute_miss_false_leadtime(df_ds_for,"dry_spell","for_below_th")
```

```python
fig_cm=compute_confusionmatrix_leadtime(df_ds_for,"dry_spell","for_below_th",ylabel="Dry spell",xlabel=f"{int(probability*100)}% probability <={threshold_perc} mm")
# fig_cm.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_dsobs_cm_lt123456_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
#focus on leadtimes of interest
lt_sub=[2,4]
lt_sub_str=lt_str="".join([str(l) for l in lt_sub])
#cm per month
for m in sel_months:
    fig_cm=compute_confusionmatrix_leadtime(df_ds_for[(df_ds_for.leadtime.isin(lt_sub))&(df_ds_for.date_month.dt.month==m)],"dry_spell","for_below_th",ylabel="Dry spell",xlabel=f"{int(probability*100)}% probability <={threshold_perc} mm",title=f"Month = {calendar.month_name[m]}",colp_num=2)
#     fig_cm.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_dsobs_cm_lt{lt_sub_str}_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{calendar.month_abbr[m].lower()}.png"))
```

```python
df_ds_for["pcode"]="MW3"
for l in [2,4]:#df_ds_for.leadtime.unique():
    df_ds_for_lt=df_ds_for[df_ds_for.leadtime==l]
    df_hm_daterange_lt=refactor_data_hm(df_ds_for_lt,"dry_spell","for_below_th")

    lt_str_sel=str(l)
#     df_hm_daterange_lt.to_csv(os.path.join(monthly_precip_exploration_dir,f"monthly_precip_dsobs_formonth_lt{lt_str_sel}_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{month_str}.csv"))
```

```python
#for the trigger now focussing on janfeb for leadtime 2,4, so print those numbers
df_pr_sel=compute_miss_false_leadtime(df_ds_for[df_ds_for.date.dt.month.isin([1,2])],"dry_spell","for_below_th")
df_pr_sel[df_pr_sel.leadtime.isin([2,4])]
```

### Determine skill based on set threshold, with varying probability
**NOTE: from here on the code hasn't been kept up-to-date, so might not work anymore**
From here on different methods of defining the threshold and probability are experimented with. However, we chose to go with the first method that was presented above.    

Threshold is set based on the analysis of observed monthly precipitation and dry spells. 
The probability is set by equalling the forecasted months that the threshold would have been met, equal to the number of months the observed precip was below the threshold

```python
#max mm precip/month for which the month is classified as "dry spell likely"
#based on analysis with overlap dry spells
threshold=170
```

```python
#set all rows for which below threshold was forecasted
df_for_sel.loc[:,"below_threshold"]=np.where(df_for_sel.loc[:,aggr_meth]<=threshold,1,0)
```

```python
#compute the percentage of ensemble members that were below the threshold for each date-adm-leadtime combination
df_sel_date=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).agg(mem_below=("below_threshold","sum"),mem_num=("below_threshold","count"))
df_sel_date["perc_below"]=df_sel_date["mem_below"]/df_sel_date["mem_num"]*100
```

```python
#plot the % of members that are below the threshold per leadtime
num_plots = len(df_sel_date.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,10))
for i, m in enumerate(df_sel_date.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.ecdfplot(data=df_sel_date[df_sel_date.leadtime==m],x="perc_below",stat="count",complementary=True,ax=ax)

    ax.set_xlabel(f"% of members forecast below {threshold} mm", labelpad=20, weight='bold', size=12)
    ax.set_ylabel("Number of months", labelpad=20, weight='bold', size=12)
    ax.set_title(f"leadtime = {m} months")
    ax.set_xlim(0,100)
    sns.despine(left=True, bottom=True)


fig.suptitle(f"Number of months for which x% of the members forecasts below {threshold} mm",size=20)
fig.tight_layout()
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_percmembers_th{threshold}_{adm_str}_{month_str}.png"))
```

```python
#merge observed and forecasted data
df_obs_for=df_obs_month_sel.merge(df_sel_date,on=["date","ADM1_EN"],how="right")[["date","leadtime","ADM1_EN",aggr_meth,"perc_below"]]
df_obs_for["date_month"]=df_obs_for.date.dt.to_period("M")
```

```python
df_obs_for=df_obs_for.merge(df_ds[["ADM1_EN","date_month","dry_spell"]],how="left",on=["ADM1_EN","date_month"])
df_obs_for.loc[:,"dry_spell"]=df_obs_for.dry_spell.replace(np.nan,0).astype(int)
```

```python
df_obs_for.head()
```

```python
# for i in df_obs_for.leadtime.unique():
#     for a in df_obs_for.ADM1_EN.unique():
#         print(i)
#         print(df_obs_for[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a)].corr())
```

```python
#check that threshold didn't change with some code
threshold
```

```python
#indicate months with observed precip below threshold
df_obs_for.loc[:,"obs_below_th"]=np.where(df_obs_for.loc[:,aggr_meth]<=threshold,1,0)
```

```python
#plot distribution of probability with and without dry spell
g = sns.FacetGrid(df_obs_for, height=5, col="leadtime",hue="dry_spell",palette={0:no_ds_color,1:ds_color})
g.map_dataframe(sns.histplot, "perc_below",common_norm=False,kde=True,alpha=1,binwidth=10)

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel(f"Probability <={threshold} mm")
g.fig.suptitle(f"Probability <={threshold} mm separated by dry spell occurred")
g.fig.tight_layout()
```

```python
g = sns.FacetGrid(df_obs_for, height=5, col="leadtime",hue="obs_below_th",palette={0:no_ds_color,1:ds_color})
g.map_dataframe(sns.histplot, "perc_below",common_norm=False,kde=True,alpha=1,binwidth=10)

g.add_legend(title=f"<={threshold} mm occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel(f"Probability <={threshold} mm")
g.fig.suptitle(f"Probability <={threshold} mm separated by observed <={threshold} mm")
g.fig.tight_layout()
```

```python
#compute number of months for which we would want to trigger
#can either set this to the number of months with observed below threshold monthly precipitation
#but from experimentation, saw that with number of months with dry spell, get very bad results, i.e. trigger is too strict
#or number of months with observed dry spell 
#compute number of observed months below threshold
num_obs_months_belowth=len(df_obs_month_sel[df_obs_month_sel[aggr_meth]<=threshold])
#compute number of months with observed dry spell
# num_obs_months_belowth=len(df_obs_for[(df_obs_for.leadtime==1)&(df_obs_for.ADM1_EN=="Southern")&(df_obs_for.dry_spell==1)])
num_obs_months_belowth
```

```python
#compute threshold of percentage of ensemble members forecasting to be below mm/month threshold
#compute the percentage threshold, by choosing the percentage such that the number of forecasted months below mm/month threshold are closest to the observed months with below threshold mm/month
#compute cumulative sum of months wit at least x% of members forecasting below the threshold
df_sel_perc=df_sel_date.sort_values("perc_below",ascending=False).groupby(["leadtime","ADM1_EN","perc_below"],sort=False).agg(num_months=("date","count"))
df_sel_perc["cum_months_below"]=df_sel_perc.groupby(level=0,sort=False).cumsum()
df_sel_perc=df_sel_perc.reset_index()
```

```python
#gap between forecasted months meeting the threshold and observed
df_sel_perc["gap_months_obs_for"]=(df_sel_perc['cum_months_below']-num_obs_months_belowth).abs()
#choose probability thresholds with smallest gap
df_perc_threshold=df_sel_perc.loc[df_sel_perc.groupby(["leadtime","ADM1_EN"]).gap_months_obs_for.idxmin()]
```

```python
df_perc_threshold
```

```python
ax = df_perc_threshold[["leadtime","perc_below"]].plot(kind='bar',x="leadtime",y="perc_below", figsize=(10, 8), color='#86bf91', zorder=2, width=0.85,legend=False)

# Draw vertical axis lines
vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

ax.set_xlabel("Leadtime", labelpad=20, weight='bold', size=12)
ax.set_ylabel(f"Percentage of members", labelpad=20, weight='bold', size=12)
sns.despine(left=True,bottom=True)

plt.title(f"Threshold of percentage of members forecasting below {threshold} mm per leadtime")
# plt.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_percth_th{threshold}_{adm_str}_{month_str}.png"))
```

```python
#set forecasted months that meet criteria
for i in df_obs_for.leadtime.unique():
    for a in df_obs_for.ADM1_EN.unique():
        df_obs_for.loc[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a),"for_below_th"]=np.where(df_obs_for.loc[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a),"perc_below"]>=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==i)].perc_below.values[0],1,0)
```

```python
# for i in df_obs_for.leadtime.unique():
#     for a in df_obs_for.ADM1_EN.unique():
#         print(i)
#         print(df_obs_for[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a)].corr())
```

```python
df_pr=compute_miss_false_leadtime(df_obs_for,"obs_below_th","for_below_th")
```

```python
fig,ax=plt.subplots()

df_pr.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color=ds_color,legend=True,ax=ax,label="observed below and forecasted above threshold (misses)")
df_pr.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color=no_ds_color_dark,legend=True,ax=ax,label="observed above and forecasted below threshold (false alarms)")

ax.set_xlabel("Leadtime (months)", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
ax.set_ylim(0,100)
sns.despine(left=True,bottom=True)

vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of misses and false alarms compared to observed monthly precipitation per leadtime")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_obsmonth_missfalse_th{threshold}_percvar_{adm_str}_{month_str}.png"))
```

```python
#compute metrics comparing to dry spells
df_pr_ds=compute_miss_false_leadtime(df_obs_for,"dry_spell","for_below_th")
```

```python
fig,ax=plt.subplots()

df_pr_ds.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color=ds_color,legend=True,ax=ax,label="observed dry spell and forecasted above threshold (misses)")
df_pr_ds.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color=no_ds_color_dark,legend=True,ax=ax,label="observed dry spell and forecasted below threshold (false alarms)") 

ax.set_xlabel("Leadtime (months)", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
sns.despine(left=True,bottom=True)
ax.set_ylim(0,100)

vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of misses and false alarms compared to observed dry spells per leadtime")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_ds_missfalse_th{threshold}_percvar_{adm_str}_{month_str}.png"))
```

```python
fig_cm=compute_confusionmatrix_leadtime(df_obs_for,"obs_below_th","for_below_th",ylabel=f"Observed precipitation <={threshold}",xlabel=f">=x% ensemble members <={threshold}")
# fig_cm.savefig(os.path.join(plots_seasonal_dir,f"mwi_cm_formonth_obsmonth_th{threshold}_percvar_{adm_str}_{month_str}.png"))
```

```python
fig_cm_ds=compute_confusionmatrix_leadtime(df_obs_for,"dry_spell","for_below_th",ylabel="Observed dry spell",xlabel=f">=x% ensemble members <={threshold}")
# fig_cm_ds.savefig(os.path.join(plots_seasonal_dir,f"mwi_cm_formonth_ds_th{threshold}_percvar_{adm_str}_{month_str}.png"))
```

### Compute based on set percentage and set threshold

```python
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_obs_for,x="leadtime",y="perc_below",ax=ax,color=no_ds_color_dark,hue="dry_spell",palette={0:no_ds_color,1:ds_color})
ax.set_ylabel(f"Probability of <={int(threshold)}mm")
sns.despine()
ax.set_xlabel("Lead time")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_boxplot_{month_str}_southern_ds7_adm1.png"))
```

```python
perc=40
```

```python
df_obs_for["for_below_th_perc"]=np.where(df_obs_for.perc_below>=perc,1,0)
```

```python
#set forecasted months that meet criteria
for i in df_obs_for.leadtime.unique():
    for a in df_obs_for.ADM1_EN.unique():
        df_obs_for.loc[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a),"for_below_th"]=np.where(df_obs_for.loc[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a),"perc_below"]>=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==i)].perc_below.values[0],1,0)
```

```python
df_pr_ds=compute_miss_false_leadtime(df_obs_for,"dry_spell","for_below_th_perc")
```

```python
fig,ax=plt.subplots()

df_pr_ds.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color=ds_color,legend=True,ax=ax,label="observed dry spell and forecasted above threshold (misses)")
df_pr_ds.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color=no_ds_color_dark,legend=True,ax=ax,label="observed dry spell and forecasted below threshold (false alarms)") 

ax.set_xlabel("Leadtime (months)", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
sns.despine(left=True,bottom=True)
ax.set_ylim(0,100)

vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of misses and false alarms per leadtime")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_ds_miss_falsealarms.png"))
# 
```

```python
cm_perc=compute_confusionmatrix_leadtime(df_obs_for,"dry_spell","for_below_th_perc",ylabel="Observed dry spell",xlabel=f">={perc}% below {threshold} mm")
# fig_cm.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_contigencymatrices.png"))
```

```python
cm_perc=compute_confusionmatrix_leadtime(df_obs_for,"dry_spell","for_below_th_perc",ylabel="Observed dry spell",xlabel=f">={perc}% below {threshold} mm")
# fig_cm.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_contigencymatrices.png"))
```

```python
# df_obs_for["pcode"]="MW3"
# adm_str="".join([a.lower() for a in sel_adm])
# month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
# for l in df_obs_for.leadtime.unique():
#     df_obs_for_lt=df_obs_for[df_obs_for.leadtime==l]
#     df_hm_daterange_lt=refactor_data_hm(df_obs_for_lt,"dry_spell","for_below_th_perc")

#     lt_str=str(l)
#     df_hm_daterange_lt.to_csv(os.path.join(monthly_precip_exploration_dir,f"monthly_precip_dsobs_formonth_lt{lt_str}_th{int(threshold)}_perc_{int(perc)}_{adm_str}_{month_str}.csv"))
```
