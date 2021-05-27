### Evaluating the forecast skill of ECMWF seasonal forecast in Malawi
This notebook is to compare the forecast skill of ECMWF's seasonal forecast for various lead times. We use the monthly total precipitation that is forecasted by this forecast.    

The first part looks at the general skill of the forecast, which is compared to CHIRPS observations. 
The second and larger part of the analysis assessess the suitability of the monthly forecast to be used as a trigger for dry spells.    

To determine the skill for dry spells, two parameters have to be set. Namely the cap of forecasted mm of precipitation during the month, and the probability of the precipitation being below this cap. This notebook explores several combinations of these two parameters.   

Note that due to the small number of historically observed dry spells, the statistical significance is low.    

The dry spells are determined in [this script](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/1589debf38eee928d323414e254f7d811d577108/analyses/malawi/scripts/mwi_chirps_dry_spell_detection.R) and ECMWF's forecast can be processed by [this script](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/e7ad2ac3a250912b713ab55fe45ed995d944ffc7/analyses/malawi/notebooks/read_in_data.py). [This notebook](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/mwi-corranalys/analyses/malawi/notebooks/mwi_monthlytotal_corr_dryspells_adm1.md) explores the relation between observed monthly precipitation and observed dry spells

```python
%load_ext autoreload
%autoreload 2
```

```python
from importlib import reload
from pathlib import Path
import os
import sys

import rioxarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xskillscore as xs
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
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
chirps_country_data_exploration_dir= os.path.join(config.DATA_DIR,config.PUBLIC_DIR, "exploration", country_iso3,'chirps')

chirps_monthly_mwi_path=os.path.join(chirps_country_data_exploration_dir,"chirps_mwi_monthly.nc")
ecmwf_country_data_processed_dir = os.path.join(country_data_processed_dir,"ecmwf")
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(country_data_processed_dir,"dry_spells","full_list_dry_spells.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
```

# Determine general skill per leadtime


### Read in forecast and observational data

```python
# #only needed if stats files haven't been computed before and are not on the gdrive
# #takes several hours
# processing.compute_stats_per_admin(country)
```

```python
da = processing.get_ecmwf_forecast()
```

```python
da_lt=processing.get_ecmwf_forecast_by_leadtime()
```

```python
da_obs=rioxarray.open_rasterio(chirps_monthly_mwi_path,masked=True)
#only select the years for which we also identified dry spells
da_obs=da_obs.sel(time=da_obs.time.dt.year.isin(range(2000,2021)))
```

```python
#interpolate forecast data such that it has the same resolution as the observed values
#using "nearest" as interpolation method and not "linear" because the forecasts are designed to have sharp edged and not be smoothed
da_lt_interp=da_lt.interp(latitude=da_obs["y"],longitude=da_obs["x"],method="nearest")
```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2020-06-01'
# end = '2020-10-31'

rf_list_slice = da_lt.sel(time=start,latitude=da_lt.latitude.values[5],longitude=da_lt.longitude.values[5])

rf_list_slice.dropna("leadtime").plot.line(label='Historical', c='grey',hue="number",add_legend=False)
rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(label='Historical', c='red',hue="number",add_legend=False)
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts.

```python
df_crps=pd.DataFrame(columns=['leadtime', 'crps'])

#rainy includes the months to select
#thresh the thresholds
subset_dict={"rainy":[12,1,2],"thresh":[170]}

for leadtime in da_lt_interp.leadtime:
    forecast = da_lt_interp.sel(
    leadtime=leadtime.values).dropna(dim='time')
    observations = da_obs.reindex({'time': forecast.time}).precip
    # For all dates
    crps = xs.crps_ensemble(observations, forecast,member_dim='number')
    append_dict = {'leadtime': leadtime.values,
                          'crps': crps.values,
                           'std': observations.std().values,
                           'mean': observations.mean().values,
              }

    if "rainy" in subset_dict:
        # For rainy season only
        observations_rainy = observations.where(observations.time.dt.month.isin(subset_dict['rainy']), drop=True)
        crps_rainy = xs.crps_ensemble(
            observations_rainy,
            forecast.where(forecast.time.dt.month.isin(subset_dict['rainy']), drop=True),
            member_dim='number')
        append_dict.update({
                f'crps_rainy': crps_rainy.values,
                f'std_rainy': observations_rainy.std().values,
                f'mean_rainy': observations_rainy.mean().values
            })
    
    if "thresh" in subset_dict:
        for thresh in subset_dict["thresh"]:
            crps_thresh = xs.crps_ensemble(observations.where(observations<=thresh), forecast.where(observations<=thresh), member_dim='number')
            append_dict.update({
                f'crps_{thresh}': crps_thresh.values,
                f'std_{thresh}': observations.where(observations<=thresh).std().values,
                f'mean_{thresh}': observations.where(observations<=thresh).mean().values
            })
        df_crps= df_crps.append([append_dict], ignore_index=True)
```

```python
def plot_skill(df_crps, division_key=None,
              ylabel="CRPS [mm]"):
    fig, ax = plt.subplots()
    df = df_crps.copy()
    for i, subset in enumerate([k for k in df_crps.keys() if "crps" in k]):
        y = df[subset]
        if division_key is not None:
            dkey = f'{division_key}_{subset.split("_")[-1]}' if subset!="crps" else division_key
            y /= df[dkey]
        ax.plot(df['leadtime'], y, ls="-", c=f'C{i}')
    ax.plot([], [], ls="-", c='k')
    # Add colours to legend
    for i, subset in enumerate([k for k in append_dict.keys() if "crps" in k]):
        ax.plot([], [], c=f'C{i}', label=subset)
    ax.set_title("ECMWF forecast skill in Malawi:\n 2000-2020 forecast")
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(ylabel)
    ax.legend()
```

```python
#performance pretty bad.. especially looking at the mean values, it is about 20% off on average for decjanfeb..
# Plot absolute skill
plot_skill(df_crps)

# Rainy season performs the worst, but this is likely because 
# the values during this time period are higher. Try using 
# reduced skill (dividing by standard devation).
plot_skill(df_crps, division_key='std', ylabel="RCRPS")

#This is perhpas not exactly what we want because we know this 
#data comes from the same location and the dataset has the same properties, 
#but we are splitting it up by mean value. Therefore try normalizing using mean
plot_skill(df_crps, division_key='mean', ylabel="NCRPS (CRPS / mean)")

```

# Determine skill for predicting dry spells


### Define functions

```python
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
        df_pr.loc[m,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tp+fp)*100
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

```python
#read the ecmwf forecast per adm1 per date and concat all dates
# the mwi_seasonal-monthly-single-levels_v5_interp*.csv contain results when interpolating the forecasts to be more granular, but results actually worsen with this
all_files = glob.glob(os.path.join(ecmwf_country_data_processed_dir, "mwi_seasonal-monthly-single-levels_v5_2*.csv"))

df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
df_for   = pd.concat(df_from_each_file, ignore_index=True)
```

```python
#this should be the number of years*number of months
print(len(all_files))
```

```python
#number of years*months till may 2021
21*12+5
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

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for this analysis we are only interested in the southern region during the DecJanFeb period, since this is most sensitive to dry spells
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
df_for_quant[["date","leadtime","mean_cell"]].head()
```

```python
#plot the distribution of precipitation. Can clearly see that for leadtime=1 the values are more spread (but leadtime=1 is really the month that is currently occurring so less useful)
g = sns.FacetGrid(df_for_quant, height=5, col="leadtime",col_wrap=3)
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=1,binwidth=10)

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
g = sns.FacetGrid(df_ds_for_labels, height=5, col="leadtime",row="month_name",hue="dry_spell",palette={"no":"#CCE5F9","yes":'#F2645A'})
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,alpha=1,binwidth=10)

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
```

```python
#plot distribution precipitation with and withoud dry spell
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_ds_for_labels,x="leadtime",y="mean_cell",ax=ax,color="#66B0EC",hue="dry_spell",palette={"no":"#CCE5F9","yes":'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Leadtime")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_boxplot_formonth_dsobs_perlt_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
#compute tp,tn,fp,fn per threshold
y_target =  df_ds_for.dry_spell
threshold_list=np.arange(0,df_ds_for.mean_cell.max() - df_ds_for_labels.mean_cell.max()%10,10)
df_pr_th=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_ds_for.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    df_pr_th.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn+0.00001)*100,tn/(tn+fp+0.00001)*100,fn/(tp+fn+0.00001)*100,fp/(tp+fp+0.00001)*100
    df_pr_th.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_th=df_pr_th.reset_index()
```

```python
fig,ax=plt.subplots()

df_pr_th.plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',style='.-',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_th.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',style='.-',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold")

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
```

```python
fig,ax=plt.subplots()

df_pr_th.plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=False,ax=ax,style='.-',label="dry spell occurred and monthly precipitation above threshold (misses)")
df_pr_th.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=False,ax=ax,style='.-',label="no dry spell occurred and monthly precipitation below threshold (false alarms)") #["#18998F","#FCE0DE"]

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
df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1)
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
        df_pr_perlt.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate","precision","recall","num_trig","detection_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tp+fp+0.000001)*100,tp/(tp+fp+0.00001)*100,tp/(tp+fn)*100,tp+fp,tp/(tp+fn)*100
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
num_plots = len(df_pr_sep_lt.leadtime.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure()#figsize=(25,10))
for i, m in enumerate(df_pr_sep_lt.leadtime.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_miss_rate" ,figsize=(20, 9), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation above threshold (miss rate)")
    df_pr_sep_lt[df_pr_sep_lt.leadtime==m].plot(x="threshold",y="month_false_alarm_rate" ,figsize=(20, 9), color='#66B0EC',legend=False,ax=ax,label="monthly precipitation below threshold but no dry spell occurred (false alarms)")

    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"Leadtime = {int(m)} months")
    sns.despine(left=True,bottom=True)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_dsobs_missfalse_perlt_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
threshold_perc=df_pr_th[df_pr_th.month_ds>=df_pr_th.month_no_ds].head(1).threshold.values[0]
#for easily testing different thresholds
# threshold_perc=180
```

```python
threshold_perc
```

```python
df_ds_for["for_below_th"]=np.where(df_ds_for.mean_cell<=threshold_perc,1,0)
```

```python
df_pr_ds=compute_miss_false_leadtime(df_ds_for,"dry_spell","for_below_th")
```

```python
fig_cm=compute_confusionmatrix_leadtime(df_ds_for,"dry_spell","for_below_th",ylabel="Dry spell",xlabel=f">={int(probability*100)}% ensemble members <={threshold_perc}")
# fig_cm.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_formonth_dsobs_cm_lt{lt_str}_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{month_str}.png"))
```

```python
#focus on leadtimes of interest
lt_sub=[2,4]
lt_sub_str=lt_str="".join([str(l) for l in lt_sub])
#cm per month
for m in sel_months:
    fig_cm=compute_confusionmatrix_leadtime(df_ds_for[(df_ds_for.leadtime.isin(lt_sub))&(df_ds_for.date_month.dt.month==m)],"dry_spell","for_below_th",ylabel="Dry spell",xlabel=f">={int(probability*100)}% ensemble members <={threshold_perc}",title=f"Month = {calendar.month_name[m]}",colp_num=2)
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
g = sns.FacetGrid(df_obs_for, height=5, col="leadtime",hue="dry_spell",palette={0:"#CCE5F9",1:'#F2645A'})
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
g = sns.FacetGrid(df_obs_for, height=5, col="leadtime",hue="obs_below_th",palette={0:"#CCE5F9",1:'#F2645A'})
g.map_dataframe(sns.histplot, "perc_below",common_norm=False,kde=True,alpha=1,binwidth=10)#x="mean_cell",hue="dry_spell")

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

df_pr.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="observed below and forecasted above threshold (misses)")
df_pr.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="observed above and forecasted below threshold (false alarms)")

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

df_pr_ds.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="observed dry spell and forecasted above threshold (misses)")
df_pr_ds.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="observed dry spell and forecasted below threshold (false alarms)") #["#18998F","#FCE0DE"]

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
g=sns.boxplot(data=df_obs_for,x="leadtime",y="perc_below",ax=ax,color="#66B0EC",hue="dry_spell",palette={0:"#CCE5F9",1:'#F2645A'})
ax.set_ylabel(f"Probability of <={int(threshold)}mm")
sns.despine()
ax.set_xlabel("Lead time")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_boxplot_decjanfeb_southern_ds7_adm1.png"))
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

df_pr_ds.plot(x="leadtime",y="month_miss_rate" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="observed dry spell and forecasted above threshold (misses)")
df_pr_ds.plot(x="leadtime",y="month_false_alarm_rate" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="observed dry spell and forecasted below threshold (false alarms)") #["#18998F","#FCE0DE"]

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