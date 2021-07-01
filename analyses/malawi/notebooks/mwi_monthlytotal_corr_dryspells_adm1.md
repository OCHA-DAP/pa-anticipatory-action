### The correlation of monthly precipitation with dry spells


### set general variables and functions

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math
import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
from datetime import timedelta
import itertools
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
#TODO: clean-up paths nog being used
country="malawi"
config=Config()
parameters = config.parameters(country)
iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,iso3)
data_exploration_dir= os.path.join(config.DATA_DIR, config.PUBLIC_DIR, "exploration")
dry_spells_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")

plots_dir=os.path.join(country_data_processed_dir,"plots","dry_spells")
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
#includes all dry spells also outside rainy season
all_dry_spells_list_path=os.path.join(dry_spells_processed_dir,"full_list_dry_spells.csv")
#only includes dry spells within rainy season
all_dry_spells_4mm_list_path=os.path.join(dry_spells_processed_dir,"daily_mean_dry_spells_details_4mm_2000_2020.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation_admin1.csv")
```

#### Load data
Load the dry spell and observed monthly precipitation data. 

Merge the two, classifying a month as having experienced a dry spell if at least during **7** days of that month a dry spell was observed in at least **3** adm2's

```python
def load_monthly_dryspell_precip(ds_path,precip_path,min_ds_days_month=7,min_adm_ds_month=3,include_seas=range(2000,2020),ds_adm_col="pcode",precip_adm_col="ADM1_EN",ds_date_cols=["dry_spell_first_date","dry_spell_last_date"]):
    df_ds_all=pd.read_csv(ds_path,parse_dates=ds_date_cols)
    #get list of all dates that were part of a dry spell
    df_ds_res=df_ds_all.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
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
    if precip_adm_col not in df_ds_month.columns:
        df_adm2=gpd.read_file(adm2_bound_path)
        df_ds_month=df_ds_month.merge(df_adm2[["ADM2_PCODE","ADM2_EN","ADM1_EN"]],left_on=ds_adm_col,right_on="ADM2_PCODE")
        
    df_ds_month_adm1=df_ds_month.groupby([precip_adm_col,"date_month"],as_index=False).count()
    #load the monthly precipitation data
    df_total_month=pd.read_csv(precip_path)
    #remove day part of date (day doesnt indicate anything with this data and easier for merge)
    df_total_month.date_month=pd.to_datetime(df_total_month.date_month).dt.to_period("M")
    
    #include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
    df_comb_countmonth=df_ds_month_adm1.merge(df_total_month,how="outer",on=[precip_adm_col,"date_month"])
    
    #dryspell_obs is number of adm2s in which a dry spell is observed in the given date_month
    #select all date_months with at least min_adm_ds_month adm2 having a dry spell
    df_comb_countmonth["dry_spell"]=np.where(df_comb_countmonth.dryspell_obs>=min_adm_ds_month,1,0)
    
    df_comb_countmonth["month"]=df_comb_countmonth.date_month.dt.month
    df_comb_countmonth["season_approx"]=np.where(df_comb_countmonth.month>=10,df_comb_countmonth.date_month.dt.year,df_comb_countmonth.date_month.dt.year-1)
    
    #only select the seasons for which dry spells were computed! 
    df_comb_countmonth=df_comb_countmonth[df_comb_countmonth.season_approx.isin(include_seas)]
    
    return df_comb_countmonth
```

### Analysis on Admin1


#### Merge precipitation data on adm1 with dry spells on adm2

```python
#select only dec,jan,feb as those are the months we are focussing on
months_sel=[1,2]
```

```python
min_ds_days_month=7
min_adm_ds_month=3
df_comb_countmonth=load_monthly_dryspell_precip(all_dry_spells_list_path,monthly_precip_path,min_ds_days_month=min_ds_days_month,min_adm_ds_month=min_adm_ds_month)
```

```python
df_comb_countmonth_labels=df_comb_countmonth.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
#very ugly but working, only used for plotting
df_comb_countmonth_labels[" month"]=df_comb_countmonth_labels.month.apply(lambda x: calendar.month_name[x])
```

```python
#since almost all dry spells occur in the southern region, we solely look at this region for this analysis
#if combining all the regions, this can clutter results
df_comb_countmonth_southern=df_comb_countmonth[df_comb_countmonth.ADM1_EN=="Southern"]
df_comb_countmonth_labels_southern=df_comb_countmonth_labels[df_comb_countmonth_labels.ADM1_EN=="Southern"].sort_values(["month","dry_spell"])
```

```python
df_southern_countmonth_decjanfeb=df_comb_countmonth_southern[df_comb_countmonth_southern.month.isin(months_sel)]
df_southern_countmonth_labels_decjanfeb=df_comb_countmonth_labels_southern[df_comb_countmonth_labels_southern.month.isin(months_sel)]
```

```python
#we only look at decjanfeb since these months are most sensitive to dry spells
df_comb_ds_southern_decjanfeb=df_southern_countmonth_decjanfeb[(df_southern_countmonth_decjanfeb.dry_spell==1)]
```

```python
df_comb_ds_southern_decjanfeb[["date_month","dryspell_obs","mean_cell","season_approx"]]
```

```python
print("return period dry spell:",len(df_comb_ds_southern_decjanfeb.season_approx.unique())/len(df_comb_countmonth_southern.season_approx.unique()))
```

```python
#plot distribution monthly precip with and without ds, bars
#TODO: somehow not showing if only one drysepll during a month.. Has to do smth with facetgrid I believe
g = sns.FacetGrid(df_comb_countmonth_labels_southern, height=5, col=" month",hue="dry_spell",col_wrap=3,palette={"no":"#CCE5F9","yes":'#F2645A'},col_order=["December","January","February"])
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=0.5,binwidth=10)#x="mean_cell",hue="dry_spell")

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
# g.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_distribution_facet_decjanfeb_southern_ds{min_ds_days_month}{min_adm_ds_month}_adm1.png"))
```

```python
# #plot observed values with threshold line, to know where the threshold falls in the distribution
# #this threshold was set with later work on the correlation with forecasts
# df_comb_janfeb=df_comb_countmonth_labels_southern[df_comb_countmonth_labels_southern.month.isin([1,2])]
# perc_210=sum(np.where(df_comb_janfeb.mean_cell<=210,1,0))/len(df_comb_janfeb)*100
# print(f"{perc_210:.0f}% of the months had no more than 210 mm observed rainfall")
# fig, ax = plt.subplots(figsize=(16,8))
# g=sns.histplot(df_comb_countmonth_labels_southern[df_comb_countmonth_labels_southern.month.isin([1,2])],x="mean_cell",ax=ax,kde=True,color="#CCCCCC",binwidth=5)
# plt.axvline(210,color="#18998F",label="threshold")
# perc=np.percentile(df_comb_countmonth_labels_southern[df_comb_countmonth_labels_southern.month.isin([1,2])].mean_cell, 33)
# plt.axvline(perc,color="#C25048",label="below average")
# plt.legend()
# sns.despine()
# g.set(xlabel="Monthly precipitation (mm)")
# plt.title("Distribution of monthly precipitation in Jan+Feb in the Southern region from 2000-2020");
```

```python
#plot distribution monthly precip with and without ds, boxplot
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_southern_countmonth_labels_decjanfeb,x=" month",y="mean_cell",ax=ax,color="#66B0EC",hue="dry_spell",order=["December","January","February"],palette={"no":"#CCE5F9","yes":'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Total monthly precipitation (mm)")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_boxplot_decjanfeb_southern_ds{min_ds_days_month}{min_adm_ds_month}_adm1.png"))
```

```python
#compute tp,tn,fp,fn for different thresholds
def compute_miss_false_thresholds(df):
    y_target =  df.dry_spell
    threshold_list=np.arange(0,df.mean_cell.max() - df.mean_cell.max()%10,10)
    df_pr=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
    for t in threshold_list:
        y_predicted = np.where(df.mean_cell<=t,1,0)

        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        df_pr.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tn+fp)*100
        df_pr.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
    df_pr=df_pr.reset_index()
    return df_pr
```

```python
df_pr_decjanfeb=compute_miss_false_thresholds(df_southern_countmonth_decjanfeb)
```

```python
#plot perc of months correctly categorized with and without ds for different thresholds
fig,ax=plt.subplots()

df_pr_decjanfeb.plot(x="threshold",y="month_ds" ,figsize=(16, 8),style='.-', color='#F2645A',legend=True,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_decjanfeb.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8),style='.-', color='#66B0EC',legend=True,ax=ax,label="no dry spell occurred and monthly precipitation above threshold") #["#18998F","#FCE0DE"]

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_threshold_categorized_ds{min_ds_days_month}{min_adm_ds_month}_adm1.png"))
```

```python
#plot miss and false alarm rate for different thresholds
fig,ax=plt.subplots()

df_pr_decjanfeb.plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), style='.-',color='#F2645A',legend=True,ax=ax,label="dry spell occurred and monthly precipitation above threshold (misses)")
df_pr_decjanfeb.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), style='.-',color='#66B0EC',legend=True,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_threshold_missfalse_ds{min_ds_days_month}{min_adm_ds_month}_adm1.png"))
```

```python
#TODO: choose if we prefer this point to be the last on the left or the first on the right of the intersection
print("point of intersection false alarm and miss rate")
df_pr_decjanfeb[df_pr_decjanfeb.month_false_alarm_rate>=df_pr_decjanfeb.month_miss_rate].head(1)
```

```python
#compute confusion matrix for intersection threshold
y_target =  df_southern_countmonth_decjanfeb.dry_spell

t_intersection=df_pr_decjanfeb[df_pr_decjanfeb.month_ds>=df_pr_decjanfeb.month_no_ds].head(1).threshold.values[0]
y_predicted = np.where(df_southern_countmonth_decjanfeb.mean_cell<=t_intersection,1,0)

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN1 during month")
ax.set_xlabel(f"<={int(t_intersection)} mm precipitation during month")
plt.show()
fig.tight_layout()
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_confusionmatrix_ds{min_ds_days_month}{min_adm_ds_month}_th{int(t_intersection)}_adm1.png"))
```

##### Compute misses/false alarms per month

```python
def compute_miss_false_thresholds_permonth(df):
    pr_list=[]
    threshold_list=np.arange(0,df.mean_cell.max() - df.mean_cell.max()%10,10)
    unique_months=df.month.unique()

    for m in unique_months:
        df_pr_permonth=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
        df_month=df[df.month==m]
        y_target =  df_month.dry_spell

        for t in threshold_list:
            y_predicted = np.where(df_month.mean_cell<=t,1,0)

            cm = confusion_matrix(y_target=y_target, 
                                  y_predicted=y_predicted)
            tn,fp,fn,tp=cm.flatten()
            df_pr_permonth.loc[t,["month_ds","month_no_ds","month_miss_rate","month_false_alarm_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fn/(tp+fn)*100,fp/(tn+fp)*100
            df_pr_permonth.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
            df_pr_permonth.loc[t,"month"]=m
        df_pr_permonth=df_pr_permonth.reset_index()
        pr_list.append(df_pr_permonth)
    df_pr=pd.concat(pr_list).sort_values(["month","threshold"])
    return df_pr
```

```python
df_pr_sep_month=compute_miss_false_thresholds_permonth(df_southern_countmonth_decjanfeb)
```

```python
num_plots = len(df_pr_sep_month.month.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(16,1))
for i, m in enumerate(months_sel):#df_pr_sep_month.month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), style='.-',color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation above threshold (misses)")
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), style='.-',color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"month = {calendar.month_name[int(m)]}")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_threshold_missfalse_ds{min_ds_days_month}{min_adm_ds_month}_adm1_permonth.png"))
```

```python
num_plots = len(df_pr_sep_month.month.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(16,1))
for i, m in enumerate(months_sel):#df_pr_sep_month.month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_ds" ,figsize=(16, 8), style='.-',color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold (match)")
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), style='.-',color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false positive match)")

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"month = {calendar.month_name[int(m)]}")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_threshold_missfalse_ds{min_ds_days_month}{min_adm_ds_month}_adm1_permonth.png"))
```

```python
num_plots = len(df_pr_sep_month.month.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(16,1))
for i, m in enumerate(months_sel):#df_pr_sep_month.month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and observed monthly precipitation below threshold")
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and observed monthly precipitation above threshold") #["#18998F","#FCE0DE"]

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"month = {calendar.month_name[int(m)]}")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7_adm1.png"))
```

##### Prepare for viz heatmap

```python
#prep data for heatmap visualization in R
def label_ds(row):
    if row["dry_spell"]==1 and row["precip_below"]==1:
        return 3
    elif row["dry_spell"]==1:
        return 2
    elif row["precip_below"]==1:
        return 1
    else:
        return 0
    
def refactor_data_hm(df,threshold):
    df_th=df[df.mean_cell<=threshold]
    df_th.loc[:,"first_date"]=pd.to_datetime(df_th.date)
    df_th.loc[:,"last_date"]=df_th.date_month.dt.to_timestamp("M")
    df_th_res=df_th.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_th_res[['first_date', 'last_date']].values]
    #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
    df_precip_daterange=df_th_res[["ADM1_EN"]].join(pd.DataFrame(a)).set_index(["ADM1_EN"]).stack().droplevel(-1).reset_index()
    df_precip_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed dry spell, so add that information
    df_precip_daterange["precip_below"]=1

    df_ds=df[df.dry_spell==1]
    df_ds.loc[:,"first_date"]=pd.to_datetime(df_ds.date)
    df_ds.loc[:,"last_date"]=df_ds.date_month.dt.to_timestamp("M")
    df_ds_res=df_ds.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_res[['first_date', 'last_date']].values]
    #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
    df_ds_daterange=df_ds_res[["ADM1_EN"]].join(pd.DataFrame(a)).set_index(["ADM1_EN"]).stack().droplevel(-1).reset_index()
    df_ds_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed dry spell, so add that information
    df_ds_daterange.loc[:,"dry_spell"]=1

    df_daterange_comb=df_ds_daterange.merge(df_precip_daterange,on=["date","ADM1_EN"],how="outer")
    df_daterange_comb.dry_spell=df_daterange_comb.dry_spell.replace(np.nan,0).astype(int)
    df_daterange_comb.precip_below=df_daterange_comb.precip_below.replace(np.nan,0).astype(int)

    df_daterange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df.ADM1_EN.unique())),columns=['date','ADM1_EN'])
    df_daterange_comb=df_daterange.merge(df_daterange_comb,on=["date","ADM1_EN"],how="left")
    
    #encode dry spells and whether it was none, only observed, only forecasted, or both
    df_daterange_comb.loc[:,"dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row),axis=1)
    
    #the R visualization code expects an entry per adm2
    df_adm2=gpd.read_file(adm2_bound_path)
    df_daterange_comb=df_daterange_comb.merge(df_adm2[["ADM2_EN","ADM2_PCODE","ADM1_EN"]])
    df_daterange_comb.rename(columns={"ADM2_PCODE":"pcode"},inplace=True)
    
    return df_daterange_comb
```

```python
#select value where %false alarms increases more than %hits increases
threshold=int(df_pr_decjanfeb[df_pr_decjanfeb.month_false_alarm_rate>=df_pr_decjanfeb.month_miss_rate].head(1)["threshold"].values[0])
```

```python
threshold
```

```python
df_daterange_comb=refactor_data_hm(df_comb_countmonth_southern,threshold)
```

```python
df_daterange_comb_southern_decjanfeb = df_daterange_comb[(df_daterange_comb.ADM1_EN=="Southern")&(df_daterange_comb.date.dt.month.isin(months_sel))]
df_daterange_comb_southern_decjanfeb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_ds{min_ds_days_month}{min_adm_ds_month}_adm1_th{threshold}_southern_decjanfeb.csv"))
```

### ADMIN1 with definition of dry spell of <=4mm/day

```python
df_countmonth_4mm=load_monthly_dryspell_precip(all_dry_spells_4mm_list_path,monthly_precip_path)
```

```python
df_countmonth_4mm_labels=df_countmonth_4mm.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
#very ugly but working, only used for plotting
df_countmonth_4mm_labels[" month"]=df_countmonth_4mm_labels.month.apply(lambda x: calendar.month_name[x])
```

```python
#since almost all dry spells occur in the southern region, we solely look at this region for this analysis
#if combining all the regions, this can clutter results
df_countmonth_4mm_southern=df_countmonth_4mm[df_countmonth_4mm.ADM1_EN=="Southern"]
df_countmonth_4mm_labels_southern=df_countmonth_4mm_labels[df_countmonth_4mm_labels.ADM1_EN=="Southern"].sort_values(["month","dry_spell"])
```

```python
#select only dec,jan,feb as those are the months we are focussing on
df_countmonth_4mm_southern_decjanfeb=df_countmonth_4mm_southern[df_countmonth_4mm_southern.month.isin(months_sel)]
df_countmonth_4mm_labels_southern_decjanfeb=df_countmonth_4mm_labels_southern[df_countmonth_4mm_labels_southern.month.isin(months_sel)]
```

```python
#we only look at decjanfeb since these months are most sensitive to dry spells
df_comb_ds_4mm_southern_decjanfeb=df_countmonth_4mm_southern_decjanfeb[(df_countmonth_4mm_southern_decjanfeb.dry_spell==1)]
```

```python
df_comb_ds_4mm_southern_decjanfeb[["date_month","dryspell_obs","mean_cell","season_approx"]]
```

```python
print("return period dry spell:",len(df_comb_ds_4mm_southern_decjanfeb.season_approx.unique())/len(df_countmonth_4mm_southern_decjanfeb.season_approx.unique()))
```

```python
g = sns.FacetGrid(df_countmonth_4mm_labels_southern.sort_values("mean_cell",ascending=False), height=5, col=" month",hue="dry_spell",col_wrap=3,palette={"no":"#CCE5F9","yes":'#F2645A'},col_order=["December","January","February"])
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=1,binwidth=10,thresh=0,pthresh=0)#x="mean_cell",hue="dry_spell")

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
# g.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_distribution_facet_decjanfeb_southern_ds7.png"))
```

```python
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_countmonth_4mm_labels_southern_decjanfeb,x=" month",y="mean_cell",ax=ax,color="#66B0EC",hue="dry_spell",order=["December","January","February"],palette={"no":"#CCE5F9","yes":'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Total monthly precipitation (mm)")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_boxplot_decjanfeb_southern_ds7.png"))
```

```python
#compute tp,tn,fp,fn
y_target =  df_southern_countmonth_decjanfeb.dry_spell
threshold_list=np.arange(0,df_southern_countmonth_decjanfeb.mean_cell.max() - df_southern_countmonth_decjanfeb.mean_cell.max()%10,10)
df_pr_decjanfeb=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_southern_countmonth_decjanfeb.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    df_pr_decjanfeb.loc[t,["month_ds","month_no_ds","month_miss_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fp/(tn+fp)*100
    df_pr_decjanfeb.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_decjanfeb=df_pr_decjanfeb.reset_index()
```

```python
df_pr_4mm=compute_miss_false_thresholds(df_countmonth_4mm_southern_decjanfeb)
```

```python
fig,ax=plt.subplots()

df_pr_4mm.plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',style=".-",legend=True,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_4mm.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',style=".-",legend=True,ax=ax,label="no dry spell occurred and monthly precipitation above threshold")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_categorized.png"))
```

```python
#plot miss and false alarm rate for different thresholds
fig,ax=plt.subplots()

df_pr_4mm.plot(x="threshold",y="month_miss_rate" ,figsize=(16, 8), style='.-',color='#F2645A',legend=True,ax=ax,label="dry spell occurred and monthly precipitation above threshold (misses)")
df_pr_4mm.plot(x="threshold",y="month_false_alarm_rate" ,figsize=(16, 8), style='.-',color='#66B0EC',legend=True,ax=ax,label="no dry spell occurred and monthly precipitation below threshold (false alarms)")

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_threshold_missfalse_ds{min_ds_days_month}{min_adm_ds_month}_4mm_adm1.png"))
```

```python
#TODO: choose if we prefer this point to be the last on the left or the first on the right of the intersection
print("point of intersection false alarm and miss rate")
df_pr_4mm[df_pr_4mm.month_false_alarm_rate>=df_pr_4mm.month_miss_rate].head(1)
```

```python
#compute confusion matrix for intersection threshold
y_target =  df_countmonth_4mm_southern_decjanfeb.dry_spell

t_intersection=df_pr_4mm[df_pr_4mm.month_false_alarm_rate>=df_pr_4mm.month_miss_rate].head(1).threshold.values[0]
y_predicted = np.where(df_countmonth_4mm_southern_decjanfeb.mean_cell<=t_intersection,1,0)

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN1 during month")
ax.set_xlabel(f"<={int(t_intersection)} mm precipitation during month")
plt.show()
fig.tight_layout()
# fig.savefig(os.path.join(plots_seasonal_dir,f"mwi_plot_monthly_precipitation_confusionmatrix_ds{min_ds_days_month}{min_adm_ds_month}_th{int(t_intersection)}_4mm_adm1.png"))
```

##### Compute per month

```python
df_pr_sep_month_4mm=compute_miss_false_thresholds_permonth(df_countmonth_4mm_southern_decjanfeb)
```

```python
num_plots = len(df_pr_sep_month_4mm.month.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(16,1))
for i, m in enumerate(months_sel):#df_pr_sep_month.month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_month_4mm[df_pr_sep_month_4mm.month==m].plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
    df_pr_sep_month_4mm[df_pr_sep_month_4mm.month==m].plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold") #["#18998F","#FCE0DE"]

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"month = {calendar.month_name[int(m)]}")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
    # fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7.png"))
```

##### Prepare for viz heatmap

```python
#select value where %false alarms increases more than %hits increases
threshold_4mm=int(df_pr_4mm[df_pr_4mm.month_false_alarm_rate>=df_pr_4mm.month_miss_rate].head(1)["threshold"].values[0])
```

```python
df_daterange_comb_4mm=refactor_data_hm(df_countmonth_4mm_southern,threshold_4mm)
```

```python
df_daterange_comb_4mm_southern_decjanfeb = df_daterange_comb_4mm[(df_daterange_comb_4mm.ADM1_EN=="Southern")&(df_daterange_comb_4mm.date.dt.month.isin(months_sel))]
df_daterange_comb_4mm_southern_decjanfeb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_4mm_ds{min_ds_days_month}{min_adm_ds_month}_adm1_th{threshold_4mm}_southern_decjanfeb.csv"))
```

## Archive

```python
def load_monthly_dryspell_precip_count(ds_path,precip_path,min_ds_days_month=7,min_adm_ds_month=3,include_seas=range(2000,2020),ds_adm_col="pcode",precip_adm_col="ADM1_EN",ds_date_cols=["dry_spell_first_date","dry_spell_last_date"],print_numds=True):
    df_ds_all=pd.read_csv(ds_path,parse_dates=ds_date_cols)
    if print_numds:
        df_adm2=gpd.read_file(adm2_bound_path)
        df_ds_all_nums=df_ds_all.merge(df_adm2[["ADM2_PCODE","ADM2_EN","ADM1_EN"]],left_on=ds_adm_col,right_on="ADM2_PCODE")
        df_ds_all_nums=df_ds_all_nums[(df_ds_all_nums.ADM1_EN=="Southern")&((df_ds_all_nums.dry_spell_first_date.dt.month.isin(months_sel))|(df_ds_all_nums.dry_spell_first_date.dt.month.isin(months_sel)))]
        print("num adm2-date combs with dry spell:",len(df_ds_all_nums))
    #get list of all dates that were part of a dry spell
    df_ds_res=df_ds_all.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
    #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
    df_ds_daterange=df_ds_res[[ds_adm_col]].join(pd.DataFrame(a)).set_index([ds_adm_col]).stack().droplevel(-1).reset_index()
    df_ds_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed dry spell, so add that information
    df_ds_daterange["dryspell_obs"]=1
    df_ds_daterange["date_month"]=df_ds_daterange.date.dt.to_period("M")
    
    #count the number of days within a year-month combination that had were part of a dry spell
    df_ds_countmonth=df_ds_daterange.groupby([ds_adm_col,"date_month"],as_index=False).sum()
    
    df_ds_month=df_ds_countmonth[df_ds_countmonth.dryspell_obs>=min_ds_days_month]
    df_ds_month=df_ds_month.merge(df_adm2[["ADM2_PCODE","ADM2_EN","ADM1_EN"]],left_on=ds_adm_col,right_on="ADM2_PCODE")
    print("num adm2-month combs with dry spell:",len(df_ds_month[(df_ds_month.ADM1_EN=="Southern")&(df_ds_month.date_month.dt.month.isin(months_sel))].groupby(["date_month","pcode"],as_index=False).count()))
    
    #TODO: this is not really generalizable
    if precip_adm_col not in df_ds_month.columns:
        df_adm2=gpd.read_file(adm2_bound_path)
        df_ds_month=df_ds_month.merge(df_adm2[["ADM2_PCODE","ADM2_EN","ADM1_EN"]],left_on=ds_adm_col,right_on="ADM2_PCODE")
        
    df_ds_month_adm1=df_ds_month.groupby([precip_adm_col,"date_month"],as_index=False).count()
    print("num ds at adm1:", len(df_ds_month_adm1[(df_ds_month_adm1.ADM1_EN=="Southern")&(df_ds_month_adm1.date_month.dt.month.isin(months_sel))]))
    
    #load the monthly precipitation data
    df_total_month=pd.read_csv(precip_path)
    #remove day part of date (day doesnt indicate anything with this data and easier for merge)
    df_total_month.date_month=pd.to_datetime(df_total_month.date_month).dt.to_period("M")
    
    #include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
    df_comb_countmonth=df_ds_month_adm1.merge(df_total_month,how="outer",on=[precip_adm_col,"date_month"])
    
    #dryspell_obs is number of adm2s in which a dry spell is observed in the given date_month
    #select all date_months with at least min_adm_ds_month adm2 having a dry spell
    df_comb_countmonth["dry_spell"]=np.where(df_comb_countmonth.dryspell_obs>=min_adm_ds_month,1,0)
    print(f"num ds adm1 >={min_adm_ds_month}adm2s:",len(df_comb_countmonth[(df_comb_countmonth.dry_spell==1)&(df_comb_countmonth.ADM1_EN=="Southern")&(df_comb_countmonth.date_month.dt.month.isin(months_sel))]))
    
    df_comb_countmonth["month"]=df_comb_countmonth.date_month.dt.month
    df_comb_countmonth["season_approx"]=np.where(df_comb_countmonth.month>=10,df_comb_countmonth.date_month.dt.year,df_comb_countmonth.date_month.dt.year-1)
    
    #only select the seasons for which dry spells were computed! 
    df_comb_countmonth=df_comb_countmonth[df_comb_countmonth.season_approx.isin(include_seas)]
    
    return df_comb_countmonth
```

```python
min_ds_days_month=7
min_adm_ds_month=3
df_comb_countmonth=load_monthly_dryspell_precip_count(all_dry_spells_list_path,monthly_precip_path,min_ds_days_month=min_ds_days_month,min_adm_ds_month=min_adm_ds_month)
```

```python
min_ds_days_month=7
min_adm_ds_month=3
df_comb_countmonth=load_monthly_dryspell_precip_count(all_dry_spells_4mm_list_path,monthly_precip_path,min_ds_days_month=min_ds_days_month,min_adm_ds_month=min_adm_ds_month)
```

### ADMIN2

```python
def load_monthly_dryspell_precip_adm2(ds_path,precip_path,min_ds_days_month=7,include_seas=range(2000,2020),ds_adm_col="pcode",precip_adm_col="ADM2_PCODE",ds_date_cols=["dry_spell_first_date","dry_spell_last_date"]):
    df_ds_all=pd.read_csv(ds_path,parse_dates=ds_date_cols)
    
    #get list of all dates that were part of a dry spell
    df_ds_res=df_ds_all.reset_index(drop=True)
    a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
    #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
    df_ds_daterange=df_ds_res[[ds_adm_col]].join(pd.DataFrame(a)).set_index([ds_adm_col]).stack().droplevel(-1).reset_index()
    df_ds_daterange.rename(columns={0:"date"},inplace=True)
    #all dates in this dataframe had an observed dry spell, so add that information
    df_ds_daterange["dryspell_obs"]=1
    df_ds_daterange["date_month"]=df_ds_daterange.date.dt.to_period("M")
    
    #count the number of days within a year-month combination that had were part of a dry spell
    df_ds_countmonth=df_ds_daterange.groupby([ds_adm_col,"date_month"],as_index=False).sum()
    
    #load the monthly precipitation data
    df_total_month=pd.read_csv(precip_path)
    #remove day part of date (day doesnt indicate anything with this data and easier for merge)
    df_total_month.date_month=pd.to_datetime(df_total_month.date_month).dt.to_period("M")
    
    #include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
    df_comb_countmonth=df_ds_countmonth.merge(df_total_month,how="outer",left_on=[ds_adm_col,'date_month'],right_on=[precip_adm_col,"date_month"])
    
    #Assign month to dry spell if at least 7 days of that month experienced dry spell
    #dryspell_obs is count of numbers of days within a dry spell in the given date_month and adm2
    #for date_months with no dry spell days this is 0
    df_comb_countmonth["dry_spell"]=np.where(df_comb_countmonth.dryspell_obs>=min_ds_days_month,1,0)
    df_comb_countmonth["month"]=df_comb_countmonth.date_month.dt.month
    
    return df_comb_countmonth
```

```python
df_comb_countmonth=load_monthly_dryspell_precip_adm2(all_dry_spells_list_path,os.path.join(country_data_processed_dir,"chirps","seasonal","chirps_monthly_total_precipitation.csv"))
```

```python
df_comb_countmonth_labels=df_comb_countmonth.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
```

```python
#very ugly but working, only used for plotting
df_comb_countmonth_labels[" month"]=df_comb_countmonth_labels.month.apply(lambda x: calendar.month_name[x])
```

```python
#since almost all dry spells occur in the southern region, we solely look at this region for this analysis
#if combining all the regions, this can clutter results
df_comb_countmonth_southern=df_comb_countmonth[df_comb_countmonth.ADM1_EN=="Southern"]
df_comb_countmonth_labels_southern=df_comb_countmonth_labels[df_comb_countmonth_labels.ADM1_EN=="Southern"].sort_values(["month","dry_spell"])
```

```python
g = sns.FacetGrid(df_comb_countmonth_labels_southern, height=5, col=" month",hue="dry_spell",col_wrap=3,palette={"no":"#CCE5F9","yes":'#F2645A'},col_order=["December","January","February"])
g.map_dataframe(sns.histplot, "mean_cell",common_norm=False,kde=True,alpha=1,binwidth=10)#x="mean_cell",hue="dry_spell")

g.add_legend(title="Dry spell occurred")  
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Total monthly precipitation (mm)")
# g.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_distribution_facet_decjanfeb_southern_ds7.png"))
```

```python
#select only dec,jan,feb as those are the months we are focussing on
df_southern_countmonth_decjanfeb=df_comb_countmonth_southern[df_comb_countmonth_southern.month.isin(months_sel)]
df_southern_countmonth_labels_decjanfeb=df_comb_countmonth_labels_southern[df_comb_countmonth_labels_southern.month.isin(months_sel)]
```

```python
fig,ax=plt.subplots(figsize=(10,10))
g=sns.boxplot(data=df_southern_countmonth_labels_decjanfeb,x=" month",y="mean_cell",ax=ax,color="#66B0EC",hue="dry_spell",order=["December","January","February"],palette={"no":"#CCE5F9","yes":'#F2645A'})
ax.set_ylabel("Monthly precipitation")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Total monthly precipitation (mm)")
ax.get_legend().set_title("Dry spell occurred")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_boxplot_decjanfeb_southern_ds7.png"))
```

```python
# #statistics shown in boxplot by month
# for m in months_sel:
#     print(f"month={m}")
#     print(df_southern_countmonth_labels_decjanfeb.loc[df_southern_countmonth_labels_decjanfeb.month==m,"mean_cell"].describe())
```

#### Compute tp,tn,fp,fn over dec,jan,feb
We now compute per adm2, whether this adm2 had a dry spell and/or below threshold monthly precipitation
We could also choose to define this more loosely, by e.g. checking if **any** adm2 had below threshold precip at time any adm2 had a dry spell
Or e.g. at least 50% of adm2's had below precip during time any adm2 had a dry spell

NOTE: for forecasts, we have to look at admin1, so we do have to go to an approach of percentage of adm1, in order to set a threshold.. 

```python
#compute tp,tn,fp,fn
y_target =  df_southern_countmonth_decjanfeb.dry_spell
threshold_list=np.arange(0,df_southern_countmonth_decjanfeb.mean_cell.max() - df_southern_countmonth_decjanfeb.mean_cell.max()%10,10)
df_pr_decjanfeb=pd.DataFrame(threshold_list,columns=["threshold"]).set_index('threshold')
for t in threshold_list:
    y_predicted = np.where(df_southern_countmonth_decjanfeb.mean_cell<=t,1,0)

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    df_pr_decjanfeb.loc[t,["month_ds","month_no_ds","month_miss_rate"]]=tp/(tp+fn)*100,tn/(tn+fp)*100,fp/(tn+fp)*100
    df_pr_decjanfeb.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
df_pr_decjanfeb=df_pr_decjanfeb.reset_index()
```

```python
fig,ax=plt.subplots()

df_pr_decjanfeb.plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',legend=True,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
df_pr_decjanfeb.plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',legend=True,ax=ax,label="no dry spell occurred and monthly precipitation above threshold") #["#18998F","#FCE0DE"]

# Set x-axis label
ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.title(f"Percentage of months that are correctly categorized for the given threshold")
# fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_categorized.png"))
```

```python
print("point of intersection")
df_pr_decjanfeb[df_pr_decjanfeb.month_ds>=df_pr_decjanfeb.month_no_ds].head(1)
```

##### Compute per month

```python
pr_list=[]
threshold_list=np.arange(0,df_southern_countmonth_decjanfeb.mean_cell.max() - df_southern_countmonth_decjanfeb.mean_cell.max()%10,10)
unique_months=df_southern_countmonth_decjanfeb.month.unique()

for m in unique_months:
    df_pr_permonth=pd.DataFrame(threshold_list,columns=["threshold"]).set_index(['threshold'])
    df_southern_month=df_southern_countmonth_decjanfeb[df_southern_countmonth_decjanfeb.month==m]
    y_target =  df_southern_month.dry_spell
    
    for t in threshold_list:
        y_predicted = np.where(df_southern_month.mean_cell<=t,1,0)

        cm = confusion_matrix(y_target=y_target, 
                              y_predicted=y_predicted)
        tn,fp,fn,tp=cm.flatten()
        df_pr_permonth.loc[t,["month_ds","month_no_ds","month_miss_rate"]]=tp/(tp+fn+0.000001)*100,tn/(tn+fp+0.000001)*100,fp/(tn+fp+0.000001)*100
        df_pr_permonth.loc[t,["tn","tp","fp","fn"]]=tn,tp,fp,fn
        df_pr_permonth.loc[t,"month"]=m
    df_pr_permonth=df_pr_permonth.reset_index()
#     print(df_pr_permonth)
    pr_list.append(df_pr_permonth)
```

```python
df_pr_sep_month=pd.concat(pr_list).sort_values(["month","threshold"])
```

```python
num_plots = len(df_pr_sep_month.month.unique())
colp_num=3
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(16,1))
for i, m in enumerate(months_sel):#df_pr_sep_month.month.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_ds" ,figsize=(16, 8), color='#F2645A',legend=False,ax=ax,label="dry spell occurred and monthly precipitation below threshold")
    df_pr_sep_month[df_pr_sep_month.month==m].plot(x="threshold",y="month_no_ds" ,figsize=(16, 8), color='#66B0EC',legend=False,ax=ax,label="no dry spell occurred and monthly precipitation above threshold") #["#18998F","#FCE0DE"]

    # Set x-axis label
    ax.set_xlabel("Monthly rainfall threshold (mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Percentage", labelpad=20, weight='bold', size=12)
    ax.set_title(f"month = {calendar.month_name[int(m)]}")
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
plt.gcf().set_size_inches(15,5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout(rect=(0,0,1,0.9))
    # fig.savefig(os.path.join(plots_seasonal_dir,"mwi_plot_monthly_precipitation_threshold_decjanfeb_southern_ds7.png"))
```

```python

```

```python
df_southern_decjanfeb=df_comb_countmonth_southern[df_comb_countmonth_southern.month.isin(months_sel)]
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
df_ds_all=df_ds_all.merge(gdf_adm2[["ADM2_PCODE","ADM2_EN"]],how="left",left_on="pcode",right_on="ADM2_PCODE")
```

```python
#prep data for heatmap visualization in R
#select value where %false alarms increases more than %hits increases
# threshold=int(df_pr_month[df_pr_month.month_ds>=df_pr_month.month_no_ds].head(1)["threshold"].values[0])
threshold=100
df_comb_month_be=df_comb_countmonth_southern[df_comb_countmonth_southern.mean_cell<=threshold]
df_comb_month_be["first_date"]=pd.to_datetime(df_comb_month_be.date)
df_comb_month_be["last_date"]=df_comb_month_be.date_month.dt.to_timestamp("M")
df_comb_month_be_res=df_comb_month_be.reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_comb_month_be_res[['first_date', 'last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_precip_daterange=df_comb_month_be_res[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_precip_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_precip_daterange["precip_below"]=1

df_ds_res=df_ds_all.reset_index(drop=True)
a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_daterange=df_ds_res[["ADM2_EN"]].join(pd.DataFrame(a)).set_index(["ADM2_EN"]).stack().droplevel(-1).reset_index()
df_ds_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_daterange["dryspell_obs"]=1
df_daterange=pd.DataFrame(list(itertools.product(pd.date_range("2000-01-01","2020-12-31",freq="D"),df_ds_all.ADM2_EN.unique())),columns=['date','ADM2_EN'])
df_daterange_ds=df_daterange.merge(df_ds_daterange,on=["date","ADM2_EN"],how="left")
# df_daterange_comb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_th100.csv"))

df_daterange_comb=df_daterange_ds.merge(df_precip_daterange,on=["date","ADM2_EN"],how="left")
df_daterange_comb.dryspell_obs=df_daterange_comb.dryspell_obs.replace(np.nan,0)
df_daterange_comb.precip_below=df_daterange_comb.precip_below.replace(np.nan,0)
```

```python
def label_ds(row):
    if row["dryspell_obs"]==1 and row["precip_below"]==1:
        return 3
    elif row["dryspell_obs"]==1:
        return 2
    elif row["precip_below"]==1:
        return 1
    else:
        return 0
```

```python
#encode dry spells and whether it was none, only observed, only forecasted, or both
df_daterange_comb["dryspell_match"]=df_daterange_comb.apply(lambda row:label_ds(row),axis=1)
```

```python
df_adm2=gpd.read_file(adm2_bound_path)
df_daterange_comb=df_daterange_comb.merge(df_adm2[["ADM2_EN","ADM2_PCODE","ADM1_EN"]])
df_daterange_comb.rename(columns={"ADM2_PCODE":"pcode"},inplace=True)
# df_daterange_comb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_th{threshold}.csv"))
```

```python
df_daterange_comb_southern_decjanfeb = df_daterange_comb[(df_daterange_comb.ADM1_EN=="Southern")&(df_daterange_comb.date.dt.month.isin(months_sel))]
# df_daterange_comb_southern_decjanfeb.to_csv(os.path.join(country_data_processed_dir,"dry_spells","seasonal",f"monthly_dryspellobs_th{threshold}_southern_decjanfeb.csv"))
```

#### ADMIN1: percentage of cells

```python
dry_spells_percentage_adm1_path = os.path.join(country_data_processed_dir,"dry_spells","ds_counts_per_pixel_adm1.csv")
```

```python
df_ds_adm1=pd.read_csv(dry_spells_percentage_adm1_path,parse_dates=["date"])
```

```python
threshold=30
```

```python
df_ds_adm1[f"gt{threshold}"]=np.where(df_ds_adm1.perc_ds_cells>=threshold,1,0)
```

```python
#compute periods where 14 consecutive days more than x% of adm1 had dry spell
df_roll=df_ds_adm1.set_index("date").groupby("ADM1_EN",as_index=False).gt30.rolling(14,min_periods=14).sum()
df_roll=df_roll.reset_index()
df_roll[f"14d_gt{threshold}"]=np.where(df_roll[f"gt{threshold}"]>=14,1,0)
```

```python
df_roll_ds=df_roll[df_roll[f"14d_gt{threshold}"]==1]
#assign ID to dates that were part of one dry spell
df_roll_ds["ID"]=df_roll_ds.sort_values(["ADM1_EN","date"]).groupby("ADM1_EN").date.diff().dt.days.ne(1).cumsum()
```

```python
#compute start and end date of each dry spell
df_ds_perc_list=df_roll_ds.groupby(["ID","ADM1_EN"],as_index=False).agg(confirmation_date=("date","min"),dry_spell_last_date=("date","max"))
df_ds_perc_list["dry_spell_first_date"]=df_ds_perc_list.confirmation_date-timedelta(days=13)
```

```python
df_ds_perc_list
```

```python
# #to merge based on percentage, but for now decided to just aggregate adm2 to adm1
# #because percentage data doesn't include dry spells outside rainy season (december) and don't have percentage list for <=4mm /day
# adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
# df_adm1=gpd.read_file(adm1_bound_path)
# df_ds_perc_list=df_ds_perc_list.merge(df_adm1[["ADM1_EN","ADM1_PCODE"]])
# df_ds_perc_list.rename(columns={"ADM1_PCODE":"pcode"},inplace=True)
# df_ds_res=df_ds_perc_list.reset_index(drop=True)
# a = [pd.date_range(*r, freq='D') for r in df_ds_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
# #join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
# df_ds_daterange_perc=df_ds_res[["ADM1_EN"]].join(pd.DataFrame(a)).set_index(["ADM1_EN"]).stack().droplevel(-1).reset_index()
# df_ds_daterange_perc.rename(columns={0:"date"},inplace=True)
# #all dates in this dataframe had an observed dry spell, so add that information
# df_ds_daterange_perc["dryspell_obs"]=1
# df_ds_daterange_perc["date_month"]=df_ds_daterange_perc.date.dt.to_period("M")
# #count the number of days within a year-month combination that had were part of a dry spell
# df_ds_countmonth=df_ds_daterange.groupby(["ADM1_EN","date_month"],as_index=False).sum()
```
