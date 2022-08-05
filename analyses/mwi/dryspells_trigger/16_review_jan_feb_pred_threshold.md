# Analysis of Malawi Trigger Framework

## Reviewing the possibility of having two trigger values for January and February

The framework was developed using one predictive trigger value for both the months of January and February. The trigger is above the average rainfall value for February. This exercise is to try and review the trigger value to create separate values for each month.

The analysis is done on the Southern Region and the dry spell definition is 14 days with at most 2mm of rain.


```python
from importlib import reload
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math

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


```python
source_cds=False
use_unrounded_area_coords = False
interpolate=False
resolution=None #0.05
all_touched=False #True

#cannot have unrounded area coords when source_cds=False so check and set to False
if not source_cds:
    use_unrounded_area_coords = False
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
if use_unrounded_area_coords:
    plots_seasonal_dir = Path(plots_seasonal_dir) / "unrounded-coords"
aggr_meth = "mean_ADM1_PCODE"
```


```python
#set plot colors
ds_color='#F2645A'
no_ds_color='#CCE5F9'
no_ds_color_dark='#66B0EC'
```


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
def compute_confusionmatrix_leadtime(df,target_var,predict_var, ylabel,xlabel,colp_num=3,title=None,figsize=(15,8)):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df.leadtime.unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=figsize)
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


```python
start_year=2000
end_year=2020
#just locking the date to keep the analysis the same even though data is added
#might wanna delete again later
end_date="2-1-2020"
```


```python
#read the ecmwf forecast per adm1 per date and concat all dates
# the mwi_seasonal-monthly-single-levels_v5_interp*.csv contain results when interpolating the forecasts to be more granular
# but results actually worsen with this
date_list=pd.date_range(start=f'1-1-{start_year}', end=end_date, freq='MS')
all_files=[processing.get_stats_filepath(country_iso3,config,date,resolution=resolution,
                                         adm_level=1,use_unrounded_area_coords=use_unrounded_area_coords,
                                         source_cds=source_cds,all_touched=all_touched) for date in date_list]

df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
df_for   = pd.concat(df_from_each_file, ignore_index=True)

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
df_for
```

### Adding Region and Months

#### Both January and February

Unsure why the threshold of 210mm cannot be replicated.


```python
#define areas, months, years, and leadtimes of interest for the trigger
sel_adm=["Southern"]
sel_months=[1,2]
seas_years=range(start_year,end_year)
sel_leadtime=[3]

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for this analysis we are only interested in the southern region during a few months that the dry spells have the biggest impact
#df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))&(df_for.leadtime.isin(sel_leadtime))]
df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))]
df_for_sel
```


```python

#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=10,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)

#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
df_ds=load_dryspell_data(all_dry_spells_list_path)
probability=0.5
#compute the value for which x% of members forecasts the precipitation to be below or equal to that value
df_for_quant=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(probability)
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
df_for_quant[["date","leadtime",aggr_meth]].head()

```


```python
# df_for_quant=df_for_quant[df_for_quant['leadtime'] <= 1]
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_for=df_ds.merge(df_for_quant,how="right",on=["ADM1_EN","date_month"])
df_ds_for.loc[:,"dry_spell"]=df_ds_for.dry_spell.replace(np.nan,0).astype(int)
#extract month and names for plotting
df_ds_for["month"]=df_ds_for.date_month.dt.month
df_ds_for["month_name"]=df_ds_for.month.apply(lambda x: calendar.month_name[x])
df_ds_for["month_abbr"]=df_ds_for.month.apply(lambda x: calendar.month_abbr[x])
df_ds_for_labels=df_ds_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
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
print("point of intersection")
#this is across all leadtimes, so might not be the best point for leadtimes of interest
#df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1)
threshold_perc=df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1).threshold.values[0]
threshold_perc
```

#### For January

It is not possible to compute the thresholds separately with confidence without understanding why the 210mm threshold is not replicable.


```python
#define areas, months, years, and leadtimes of interest for the trigger
sel_adm=["Southern"]
sel_months=[1]
seas_years=range(start_year,end_year)
sel_leadtime=[3]

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for this analysis we are only interested in the southern region during a few months that the dry spells have the biggest impact
#df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))&(df_for.leadtime.isin(sel_leadtime))]
df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))]

#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=10,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)

#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
df_ds=load_dryspell_data(all_dry_spells_list_path)
probability=0.5
#compute the value for which x% of members forecasts the precipitation to be below or equal to that value
df_for_quant=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(probability)
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
df_for_quant[["date","leadtime",aggr_meth]].head()
# df_for_quant=df_for_quant[df_for_quant['leadtime'] <= 1]
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_for=df_ds.merge(df_for_quant,how="right",on=["ADM1_EN","date_month"])
df_ds_for.loc[:,"dry_spell"]=df_ds_for.dry_spell.replace(np.nan,0).astype(int)
#extract month and names for plotting
df_ds_for["month"]=df_ds_for.date_month.dt.month
df_ds_for["month_name"]=df_ds_for.month.apply(lambda x: calendar.month_name[x])
df_ds_for["month_abbr"]=df_ds_for.month.apply(lambda x: calendar.month_abbr[x])
df_ds_for_labels=df_ds_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
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
print("point of intersection")
#this is across all leadtimes, so might not be the best point for leadtimes of interest
#df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1)
threshold_perc=df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1).threshold.values[0]
threshold_perc
```

#### For February


```python
#define areas, months, years, and leadtimes of interest for the trigger
sel_adm=["Southern"]
sel_months=[2]
seas_years=range(start_year,end_year)
sel_leadtime=[3]

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for this analysis we are only interested in the southern region during a few months that the dry spells have the biggest impact
#df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))&(df_for.leadtime.isin(sel_leadtime))]
df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))]

#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=10,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)

#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
df_ds=load_dryspell_data(all_dry_spells_list_path)
probability=0.5
#compute the value for which x% of members forecasts the precipitation to be below or equal to that value
df_for_quant=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(probability)
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
df_for_quant[["date","leadtime",aggr_meth]].head()
# df_for_quant=df_for_quant[df_for_quant['leadtime'] <= 1]
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_for=df_ds.merge(df_for_quant,how="right",on=["ADM1_EN","date_month"])
df_ds_for.loc[:,"dry_spell"]=df_ds_for.dry_spell.replace(np.nan,0).astype(int)
#extract month and names for plotting
df_ds_for["month"]=df_ds_for.date_month.dt.month
df_ds_for["month_name"]=df_ds_for.month.apply(lambda x: calendar.month_name[x])
df_ds_for["month_abbr"]=df_ds_for.month.apply(lambda x: calendar.month_abbr[x])
df_ds_for_labels=df_ds_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
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
print("point of intersection")
#this is across all leadtimes, so might not be the best point for leadtimes of interest
#df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1)
threshold_perc=df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1).threshold.values[0]
threshold_perc
```
