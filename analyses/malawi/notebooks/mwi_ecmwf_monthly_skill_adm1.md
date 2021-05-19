```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
import glob
import itertools
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import math
import geopandas as gpd


import read_in_data as rd
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
iso3=parameters["iso3_code"]

country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,iso3)
ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",iso3)
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")
country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,iso3)

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])

all_dry_spells_list_path=os.path.join(country_data_processed_dir,"dry_spells","full_list_dry_spells.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
```

```python
#max mm precip/month for which the month is classified as "dry spell likely"
#based on analysis with overlap dry spells
threshold=170
```

```python
#read the ecmwf forecast per adm1 per date and concat all dates
# the mwi_seasonal-monthly-single-levels_v5_interp*.csv contain results when interpolating the forecasts to be more granular, but results actually worsen with this
all_files = glob.glob(os.path.join(ecmwf_country_data_processed_dir, "mwi_seasonal-monthly-single-levels_v5_2*.csv"))

df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
df_for   = pd.concat(df_from_each_file, ignore_index=True)
```

```python
print(len(all_files))
```

```python
#for now using mean cell as this requires one variable less to be set (else need to set percentage of cells)
aggr_meth="mean_cell"
#for earlier dates, the model included less members --> values for those members are nan --> remove those rows
df_for = df_for[df_for[aggr_meth].notna()]
df_for["season_approx"]=np.where(df_for.date.dt.month>=10,df_for.date.dt.year,df_for.date.dt.year-1)
```

```python
sel_adm=["Southern"]
sel_months=[12,1,2]
sel_leadtime=[1,2,3,4,5,6]
seas_years=range(2000,2020)

#for this analysis we are only interested in the southern region during the DecJanFeb period, since this is most sensitive to dry spells
df_for_sel=df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.leadtime.isin(sel_leadtime))&(df_for.season_approx.isin(seas_years))]

```

```python
#set all rows for which below threshold was forecasted
df_for_sel.loc[:,"below_threshold"]=np.where(df_for_sel[aggr_meth]<=threshold,1,0)
```

```python
#compute the percentage of ensemble members that were below the threshold for each date-adm-leadtime combination
df_sel_date=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).agg(mem_below=("below_threshold","sum"),mem_num=("below_threshold","count"))
df_sel_date["perc_below"]=df_sel_date["mem_below"]/df_sel_date["mem_num"]*100
```

```python
#plot the % of members that are below the threshold
#combining all adms/dates/leadtimes, but can distinguish with hue
fig,ax=plt.subplots(figsize=(8,4))
sns.ecdfplot(data=df_sel_date,x="perc_below",stat="count",complementary=True,ax=ax)#,hue="month",palette={12:"#CCE5F9",1:'#F2645A',2:"green"})#,color='#66B0EC')
# Set x-axis label
ax.set_xlabel("At least x% of members below threshold", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Number of months", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Number of months for which x% of the members is below the threshold")
```

```python
#plot the % of members that are below the threshold
#combining all adms/dates/leadtimes, but can distinguish with hue
num_plots = len(df_sel_date.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,12))
for i, m in enumerate(df_sel_date.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.ecdfplot(data=df_sel_date[df_sel_date.leadtime==m],x="perc_below",stat="count",complementary=True,ax=ax)#,hue="month",palette={12:"#CCE5F9",1:'#F2645A',2:"green"})#,color='#66B0EC')
    # Set x-axis label
    ax.set_xlabel(f"% of members forecast below {threshold} mm", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Number of months", labelpad=20, weight='bold', size=12)
    ax.set_title(f"leadtime = {m} months")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

fig.suptitle(f"Number of months for which x% of the members forecasts below {threshold} mm",size=20)
fig.tight_layout()
fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_forecast_percmembers_below_{threshold}.png"))
```

### Load observational data

```python
#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=10,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)
```

```python
#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
```

```python
#check that threshold didn't change with some code
threshold
```

```python
#indicate months with observed precip below threshold
df_obs_month_sel.loc[:,"obs_below_th"]=np.where(df_obs_month_sel.loc[:,aggr_meth]<=threshold,1,0)
```

```python
#check that same number of months in observed and forecasted
print("number of months in forecasted data",len(df_for_sel.date.unique()))
print("number of months in observed data",len(df_obs_month_sel.date.unique()))
```

```python
#compute number of observed months below threshold
num_obs_months_belowth=len(df_obs_month_sel[df_obs_month_sel[aggr_meth]<=threshold])
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
df_sel_perc["gap_months_obs_for"]=(df_sel_perc['cum_months_below']-num_obs_months_belowth).abs()
```

```python
df_perc_threshold=df_sel_perc.loc[df_sel_perc.groupby(["leadtime","ADM1_EN"]).gap_months_obs_for.idxmin()]
```

```python
df_perc_threshold#[["leadtime","perc_below"]]
```

```python
ax = df_perc_threshold[["leadtime","perc_below"]].plot(kind='bar',x="leadtime",y="perc_below", figsize=(10, 8), color='#86bf91', zorder=2, width=0.85,legend=False)

# Draw vertical axis lines
vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Leadtime", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel(f"Percentage of members", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Threshold of percentage of members forecasting below {threshold} mm per leadtime")
plt.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_forecast_percmembers_threshold.png"))
```

```python
df_perc_threshold[(df_perc_threshold.ADM1_EN=="Southern")&(df_perc_threshold.leadtime==1)].perc_below.values
```

```python
#set forecasted months that meet criteria
for i in df_sel_perc.leadtime.unique():
    for a in df_sel_perc.ADM1_EN.unique():
        df_sel_date.loc[(df_sel_date.leadtime==i)&(df_sel_date.ADM1_EN==a),"for_below_th"]=np.where(df_sel_date.loc[(df_sel_date.leadtime==i)&(df_sel_date.ADM1_EN==a),"perc_below"]>=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==i)].perc_below.values[0],1,0)
```

```python
#merge observed and forecasted data
df_obs_for=df_obs_month_sel.merge(df_sel_date,on=["date","ADM1_EN"],how="right")[["date","leadtime","ADM1_EN","obs_below_th","for_below_th",aggr_meth,"perc_below"]]
```

```python
df_obs_for=df_obs_for.dropna(subset=["obs_below_th","for_below_th"],how="any")
```

```python
df_obs_for
```

```python
# for i in df_obs_for.leadtime.unique():
#     for a in df_obs_for.ADM1_EN.unique():
#         print(i)
#         print(df_obs_for[(df_obs_for.leadtime==i)&(df_obs_for.ADM1_EN==a)].corr())
```

```python
#TODO: also account for different adm1's

def compute_miss_false_leadtime(df,target_var,predict_var):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df.leadtime.unique())
    colp_num=3
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(10,8))
    df_pr=pd.DataFrame(list(df.leadtime.unique()),columns=["leadtime"]).set_index('leadtime')
    for i, m in enumerate(df.sort_values(by="leadtime").leadtime.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df.leadtime==m,target_var]
        y_predicted = df.loc[df.leadtime==m,predict_var]
        a="Southern"
        perc_th_value=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==m)].perc_below.values[0]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
        ax.set_ylabel(f"Observed precipitation <={threshold}")
        ax.set_xlabel(f">={perc_th_value:.2f}% ensemble members <={threshold}")
        ax.set_title(f"Leadtime={m}")

        tn,fp,fn,tp=cm.flatten()
        df_pr.loc[m,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tp+fp)*100
        df_pr.loc[m,["tn","tp","fp","fn"]]=tn,tp,fp,fn
    fig.tight_layout()
    df_pr=df_pr.reset_index()
    # fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_contigencymatrices.png"))
    return df_pr, fig
```

```python
df_pr,fig_cm=compute_miss_false_leadtime(df_obs_for,"obs_below_th","for_below_th")
fig_cm.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_contigencymatrices.png"))
```

```python
fig,ax=plt.subplots()

df_pr.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="observed below and forecasted above threshold (misses)")
df_pr.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="observed above and forecasted below threshold (false alarms)") #["#18998F","#FCE0DE"]

# Set x-axis label
ax.set_xlabel("Leadtime (months)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_ylim(0,100)

vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of misses and false alarms per leadtime")
fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_miss_falsealarms.png"))
# 
```

## Prepare for heatmap visualization

```python
#TODO: account for different adm1's
# df_obs_for["ADM1_EN"]="Southern"
df_obs_for["pcode"]="MW3"
```

```python
def label_ds(row):
    if row["obs_below_th"]==1 and row["for_below_th"]==1:
        return 3
    elif row["obs_below_th"]==1:
        return 2
    elif row["for_below_th"]==1:
        return 1
    else:
        return 0

#convert monthly dates to dateranges with an entry per day
df_obs_for["first_date"]=pd.to_datetime(df_obs_for.date)
df_obs_for["last_date"]=df_obs_for.date.dt.to_period("M").dt.to_timestamp("M")

for l in df_obs_for.leadtime.unique():
    df_obs_for_lt=df_obs_for[df_obs_for.leadtime==l]
    df_obs_bel=df_obs_for_lt[df_obs_for_lt.obs_below_th==1]
    df_obs_bel_res=df_obs_bel.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_obs_bel_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_obs_bel_daterange=df_obs_bel_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_obs_bel_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed below threshold monthly precipitation, so add that information
    df_obs_bel_daterange["obs_below_th"]=1

    df_for_bel=df_obs_for_lt[df_obs_for_lt.for_below_th==1]
    df_for_bel_res=df_for_bel.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_for_bel_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_for_bel_daterange=df_for_bel_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_for_bel_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an forecasted below threshold monthly precipitation, so add that information
    df_for_bel_daterange["for_below_th"]=1

    #merge the observed and forecasted daterange
    df_daterange_comb=df_obs_bel_daterange.merge(df_for_bel_daterange,on=["date","pcode"],how="outer")

    df_alldatesrange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df_obs_for_lt.pcode.unique())),columns=['date','pcode'])
    df_alldatesrange_sel=df_alldatesrange[df_alldatesrange.date.dt.month.isin([12,1,2])]
    df_daterange_comb=df_daterange_comb.merge(df_alldatesrange_sel,on=["date","pcode"],how="right")

    df_daterange_comb.obs_below_th=df_daterange_comb.obs_below_th.replace(np.nan,0)
    df_daterange_comb.for_below_th=df_daterange_comb.for_below_th.replace(np.nan,0)

    #encode dry spells and whether it was none, only observed, only forecasted, or both
    #R visualization code is expecting the column "dryspell_match"
    df_daterange_comb["dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row),axis=1)

    #todo make decjanfeb variable
    adm_str="".join([a.lower() for a in sel_adm])
#     lt_str="".join([str(l) for l in sel_leadtime])
    lt_str=str(l)
    
    perc_th_value=df_perc_threshold[(df_perc_threshold.ADM1_EN=="Southern")&(df_perc_threshold.leadtime==l)].perc_below.values[0]
    df_daterange_comb.to_csv(os.path.join(monthly_precip_exploration_dir,f"monthly_precip_obsfor_lt{lt_str}_th{threshold}_perc_{int(perc_th_value)}_{adm_str}_decjanfeb.csv"))
```

### Forecasted monthly vs observed dry spells

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

```python
df_ds=load_dryspell_data(all_dry_spells_list_path)
```

```python
df_obs_for["date_month"]=df_obs_for.date.dt.to_period("M")
```

```python
df_obs_for_ds=df_obs_for.merge(df_ds[["ADM1_EN","date_month","dry_spell"]],how="left",on=["ADM1_EN","date_month"])
df_obs_for_ds.loc[:,"dry_spell"]=df_obs_for_ds.dry_spell.replace(np.nan,0).astype(int)
```

```python
df_obs_for_ds
```

```python
df_pr_ds,fig_cm_ds=compute_miss_false_leadtime(df_obs_for_ds,"dry_spell","for_below_th")
```

```python
def plot_cm_ds(df,target_var,predict_var):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df.leadtime.unique())
    colp_num=3
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(10,8))
    df_pr=pd.DataFrame(list(df.leadtime.unique()),columns=["leadtime"]).set_index('leadtime')
    for i, m in enumerate(df.sort_values(by="leadtime").leadtime.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df.leadtime==m,target_var]
        y_predicted = df.loc[df.leadtime==m,predict_var]
        a="Southern"
        perc_th_value=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==m)].perc_below.values[0]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
        ax.set_ylabel(f"Observed dry spell")
        ax.set_xlabel(f">={perc_th_value:.2f}% ensemble members <={threshold}")
        ax.set_title(f"Leadtime={m}")

        tn,fp,fn,tp=cm.flatten()
        df_pr.loc[m,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tn+fp)*100
        df_pr.loc[m,["tn","tp","fp","fn"]]=tn,tp,fp,fn
    fig.tight_layout()
    return fig
```

```python
fig_cm_ds=plot_cm_ds(df_obs_for_ds,"dry_spell","for_below_th")
```

```python
fig_cm_ds.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_ds_contigencymatrices.png"))
```

```python
fig,ax=plt.subplots()

df_pr_ds.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="observed dry spell and forecasted above threshold (misses)")
df_pr_ds.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="observed dry spell and forecasted below threshold (false alarms)") #["#18998F","#FCE0DE"]

# Set x-axis label
ax.set_xlabel("Leadtime (months)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_ylim(0,100)

vals = ax.get_yticks()
for tick in vals:
    ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

plt.title(f"Percentage of misses and false alarms per leadtime")
fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_ds_miss_falsealarms.png"))
# 
```

### set percentage threshold based on number of months with a dry spell

```python
df_obs_for_ds
```

```python
num_obs_months_belowth=7
```

```python

```

```python

```

### set threshold based on 50% probability

```python
df_for_quant=df_for_sel.groupby(["date","ADM1_EN","leadtime"],as_index=False).quantile(0.5)
```

```python
df_for_quant[["date","leadtime","mean_cell"]]
```

```python
g = sns.FacetGrid(df_for_quant, height=5, col="leadtime",col_wrap=3)
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=1,binwidth=10)#x="mean_cell",hue="dry_spell")

# g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
# g.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_distribution_facet_decjanfeb_southern_ds7.png"))
```

```python
#using all dry spells, also those outside the rainy season
#since we are only focussing on dec,jan,feb. Thus to prevent that we are missing dry spells due to late onset rainy season according to our definition
df_ds_all=pd.read_csv(all_dry_spells_list_path,parse_dates=["dry_spell_first_date","dry_spell_last_date"])
```

```python
df_ds_res=df_ds_all.reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_daterange=df_ds_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
df_ds_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_daterange["dryspell_obs"]=1
df_ds_daterange["date_month"]=df_ds_daterange.date.dt.to_period("M")
```

```python
#count the number of days within a year-month combination that had were part of a dry spell
df_ds_countmonth=df_ds_daterange.groupby(["pcode","date_month"],as_index=False).sum()
```

```python
df_ds_month=df_ds_countmonth[df_ds_countmonth.dryspell_obs>=7]
```

```python
df_adm2=gpd.read_file(adm2_bound_path)
df_ds_month=df_ds_month.merge(df_adm2[["ADM2_PCODE","ADM1_EN"]],left_on="pcode",right_on="ADM2_PCODE")
```

```python
df_ds_month_adm1=df_ds_month.groupby(["ADM1_EN","date_month"]).count()
```

```python
df_for_quant["date_month"]=df_for_quant.date.dt.to_period("M")
```

```python
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_for=df_ds_month_adm1.merge(df_for_quant,how="right",on=["ADM1_EN","date_month"])
```

```python
#dryspell_obs is number of adm2s in which a dry spell is observed in the given date_month
#select all date_months with at least one adm2 having a dry spell
df_ds_for["dry_spell"]=np.where(df_ds_for.dryspell_obs>2,1,0)
```

```python
df_ds_for["month"]=df_ds_for.date_month.dt.month
df_ds_for_labels=df_ds_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
```

```python
#very ugly but working, only used for plotting
df_ds_for_labels[" month"]=df_ds_for_labels.month.apply(lambda x: calendar.month_name[x])
```

```python
g = sns.FacetGrid(df_ds_for_labels, height=5, col="leadtime",row=" month",hue="dry_spell",palette={"no":"#CCE5F9","yes":'#F2645A'})
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=1,binwidth=10)#x="mean_cell",hue="dry_spell")

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
# g.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_distribution_facet_decjanfeb_southern_ds7.png"))
```

```python
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_ds_for_labels,x="leadtime",y="mean_cell",ax=ax,color="#66B0EC",hue="dry_spell",palette={"no":"#CCE5F9","yes":'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Total monthly precipitation (mm)")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_boxplot_decjanfeb_southern_ds7_adm1.png"))
```

```python
#compute tp,tn,fp,fn
y_target =  df_ds_for.dry_spell
threshold_list=np.arange(0,df_ds_for.mean_cell.max() - df_ds_for_labels.mean_cell.max()%10,10)
df_pr_decjanfeb=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_ds_for.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    df_pr_decjanfeb.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tp+fp)*100
    df_pr_decjanfeb.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_decjanfeb=df_pr_decjanfeb.reset_index()
```

```python
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
```

```python
fig,ax=plt.subplots()

df_pr_decjanfeb.plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',style='.-',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_decjanfeb.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',style='.-',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold")

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
# fig.tight_layout()
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_categorized_adm1_dryspells.png"))
```

```python
fig,ax=plt.subplots()

df_pr_decjanfeb.plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=False,ax=ax,style='.-',label="dry spell occurred and monthly precipitation above threshold (misses)")
df_pr_decjanfeb.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=False,ax=ax,style='.-',label="no dry spell occurred and monthly precipitation below threshold (false alarms)") #["#18998F","#FCE0DE"]

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
# fig.tight_layout()
fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_categorized_adm1_dryspells.png"))
```

```python
print("point of intersection")
df_pr_decjanfeb[df_pr_decjanfeb.month_ds>=df_pr_decjanfeb.month_no_ds].head(1)
```

```python
pr_list=[]
threshold_list=np.arange(0,df_ds_for.mean_cell.max() - df_ds_for.mean_cell.max()%10,10)
unique_lt=df_ds_for.leadtime.unique()

for m in unique_lt:
    df_pr_perlt=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
    df_ds_for_lt=df_ds_for[df_ds_for.leadtime==m]
    y_target =  df_ds_for_lt.dry_spell
    
    for t in threshold_list:
        y_predicted = np.where(df_ds_for_lt.mean_cell<=t,1,0)

        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        df_pr_perlt.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tp+fp+0.000001)*100
        df_pr_perlt.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
        df_pr_perlt.loc[t,"leadtime"]=m
    df_pr_perlt=df_pr_perlt.reset_index()
#     print(df_pr_permonth)
    pr_list.append(df_pr_perlt)
```

```python
df_pr_sep_lt=pd.concat(pr_list).sort_values(["leadtime","threshold"])
```

```python
num_plots = len(df_pr_sep_lt.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure()#figsize=(25,10))
for i, m in enumerate(df_pr_sep_lt.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_ds" ,figsize=(20, 9), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_no_ds" ,figsize=(20, 9), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold") #["#18998F","#FCE0DE"]

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"Leadtime = {int(m)} months")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
# plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7_adm1.png"))
```

```python
num_plots = len(df_pr_sep_lt.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure()#figsize=(25,10))
for i, m in enumerate(df_pr_sep_lt.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_miss_rate" ,figsize=(20, 9), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation above threshold (misses)")
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_false_alarm_rate" ,figsize=(20, 9), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"Leadtime = {int(m)} months")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
# plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7_adm1.png"))
```

```python
num_plots = len(df_pr_sep_lt.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure()#figsize=(25,10))
for i, m in enumerate(df_pr_sep_lt.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot.bar(x="threshold",y="fn" ,figsize=(20, 9), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation above threshold (misses)")
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot.bar(x="threshold",y="fp" ,figsize=(20, 9), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"Leadtime = {int(m)} months")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
# plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7_adm1.png"))
```

```python
#TODO: also account for different adm1's

def compute_miss_false_leadtime_ds(df,target_var,predict_var):
    #number of dates with observed dry spell overlapping with forecasted per month
    num_plots = len(df.leadtime.unique())
    colp_num=3
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=(10,8))
    df_pr=pd.DataFrame(list(df.leadtime.unique()),columns=["leadtime"]).set_index('leadtime')
    for i, m in enumerate(df.sort_values(by="leadtime").leadtime.unique()):
        ax = fig.add_subplot(rows,colp_num,i+1)
        y_target =    df.loc[df.leadtime==m,target_var]
        y_predicted = df.loc[df.leadtime==m,predict_var]
        a="Southern"
        perc_th_value=df_perc_threshold[(df_perc_threshold.ADM1_EN==a)&(df_perc_threshold.leadtime==m)].perc_below.values[0]
        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)

        plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax) #,class_names=["No","Yes"])
        ax.set_ylabel(f"Dry spell")
        ax.set_xlabel(f">=50% ensemble members <={threshold}")
        ax.set_title(f"Leadtime={m}")

        tn,fp,fn,tp=cm.flatten()
        df_pr.loc[m,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tn+fp)*100
        df_pr.loc[m,["tn","tp","fp","fn"]]=tn,tp,fp,fn
    fig.tight_layout()
    df_pr=df_pr.reset_index()
    # fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_forecast_contigencymatrices.png"))
    return df_pr
```

```python
# threshold=200
```

```python
threshold=df_pr_decjanfeb[df_pr_decjanfeb.month_false_alarm_rate>=df_pr_decjanfeb.month_miss_rate].head(1).threshold.values[0]
```

```python
threshold
```

```python
df_ds_for["for_below_th"]=np.where(df_ds_for.mean_cell<=threshold,1,0)
```

```python
df_pr_ds=compute_miss_false_leadtime_ds(df_ds_for,"dry_spell","for_below_th")
```

```python
def label_ds(row):
    if row["dry_spell"]==1 and row["for_below_th"]==1:
        return 3
    elif row["dry_spell"]==1:
        return 2
    elif row["for_below_th"]==1:
        return 1
    else:
        return 0

df_ds_for["pcode"]="MW3"
#convert monthly dates to dateranges with an entry per day
df_ds_for["first_date"]=pd.to_datetime(df_ds_for.date)
df_ds_for["last_date"]=df_ds_for.date.dt.to_period("M").dt.to_timestamp("M")
threshold=df_pr_decjanfeb[df_pr_decjanfeb.month_false_alarm_rate>=df_pr_decjanfeb.month_miss_rate].head(1).threshold.values[0]
# df_th=df_ds_for[df_ds_for.mean_cell<=threshold]


for l in df_ds_for.leadtime.unique():
    df_ds_for_lt=df_ds_for[df_ds_for.leadtime==l]
#     df_ds_for_lt.rename(columns={"below_threshold","for_bel"})
    df_ds_bel=df_ds_for_lt[df_ds_for_lt.dry_spell==1]
    df_ds_bel_res=df_ds_bel.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_bel_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_ds_daterange=df_ds_bel_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_ds_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed below threshold monthly precipitation, so add that information
    df_ds_daterange["dry_spell"]=1

    df_for_bel=df_ds_for_lt[df_ds_for_lt.for_below_th==1]
    df_for_bel_res=df_for_bel.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_for_bel_res[['first_date', 'last_date']].values]
    #join the daterange with the adm1, which create a column per date, then stack to have each adm1-date combination
    #not really needed now cause only one adm1, but for future compatability
    df_for_bel_daterange=df_for_bel_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
    df_for_bel_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an forecasted below threshold monthly precipitation, so add that information
    df_for_bel_daterange["for_below_th"]=1

    #merge the observed and forecasted daterange
    df_daterange_comb=df_ds_daterange.merge(df_for_bel_daterange,on=["date","pcode"],how="outer")

    df_alldatesrange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df_ds_for_lt.pcode.unique())),columns=['date','pcode'])
    df_alldatesrange_sel=df_alldatesrange[df_alldatesrange.date.dt.month.isin([12,1,2])]
    df_daterange_comb=df_daterange_comb.merge(df_alldatesrange_sel,on=["date","pcode"],how="right")

    df_daterange_comb.dry_spell=df_daterange_comb.dry_spell.replace(np.nan,0)
    df_daterange_comb.for_below_th=df_daterange_comb.for_below_th.replace(np.nan,0)

    #encode dry spells and whether it was none, only observed, only forecasted, or both
    #R visualization code is expecting the column "dryspell_match"
    df_daterange_comb["dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row),axis=1)

    #todo make decjanfeb variable
    adm_str="".join([a.lower() for a in sel_adm])
#     lt_str="".join([str(l) for l in sel_leadtime])
    lt_str=str(l)
    
    perc_th_value=df_perc_threshold[(df_perc_threshold.ADM1_EN=="Southern")&(df_perc_threshold.leadtime==l)].perc_below.values[0]
#     df_daterange_comb.to_csv(os.path.join(monthly_precip_exploration_dir,f"monthly_precip_dsobs_formonth_lt{lt_str}_th{int(threshold)}_perc_50_{adm_str}_decjanfeb.csv"))
```

```python
df_daterange_comb
```

```python

```
