### The correlation of observed precipitation data with dry spells
Note: this notebook is in a very experimental state. Since based on this analysis it was decided to not move forward with below average precipitation forecasts, it was also decided to not improve the quality of this notebook for the time being. 

This notebook explores the correlation between observed below average precipitation and dry spells. 
The goal of the analysis is to see if, given perfect forecasting skill, there is information in the forecasted quantities for forecasting dryspells.
The reason why the focus is on below average precipitaiton, is because most forecasts are published in this format. 
First seasonal (=3month) below average monthly precipitation is analysed, and thereafter monthly below average precipitation. 

For the seasonal precipitation as well as for the dry spells, CHIRPS is used as data source. The occurences of below average precipitation are computed in `02_mwi_chirps_observed_monthly_seasonal_precipitation.ipynb`, and the dry spells are computed in `mwi_chirps_dry_spell_detection.R`

Methodological notes
- A dry spell is assigned to a season, by looking if the start date was any of the 3 months of that season. Whether this is the best method, is open for discussion. 


### set general variables and functions

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import calendar
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
country_iso3 = parameters["iso3_code"]

public_data_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR
country_data_processed_dir = public_data_dir / config.PROCESSED_DIR / country_iso3

dry_spells_dir=country_data_processed_dir / "dry_spells"
chirps_country_dir = country_data_processed_dir / "chirps"
plots_dir=country_data_processed_dir / "plots" / "dry_spells"
plots_seasonal_dir=os.path.join(plots_dir,"seasonal")
dry_spells_mean_aggr_adm2_path=os.path.join(dry_spells_dir,"dry_spells_during_rainy_season_list_2000_2020_mean_back.csv")
belowavg_seas_adm2_path=os.path.join(chirps_country_dir,"chirps_seasonal_below_average_precipitation.csv")
belowavg_monthly_adm2_path=os.path.join(chirps_country_dir,"chirps_monthly_below_average_precipitation.csv")
```

#### Load dry spell data

```python
df_ds=pd.read_csv(dry_spells_mean_aggr_adm2_path) 
df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
df_ds["ds_fd_m"]=df_ds.dry_spell_first_date.dt.to_period("M")
```

```python
df_ds.head()
```

```python
#compute if start of dryspell per month-ADM2
#for now only want to know if a dry spell occured in a given month, so drop those that have several dry spells confirmed within a month
df_ds_drymonth=df_ds.drop_duplicates(["ADM2_EN","ds_fd_m"]).groupby(["ds_fd_m","ADM2_EN"],as_index=False).agg("count")[["ds_fd_m","ADM2_EN","dry_spell_first_date"]]
```

#### Load historical seasonal below average rainfall
And remove seasons outside the rainy season    

```python
df_belowavg_seas=pd.read_csv(belowavg_seas_adm2_path)
#remove day part of date (day doesnt indicate anything with this data and easier for merge)
df_belowavg_seas.date_month=pd.to_datetime(df_belowavg_seas.date_month).dt.to_period("M")
```

```python
#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean_back.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])
```

```python
#set the onset and cessation date for the seasons where these are missing 
#(meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))
```

```python
df_rain_filled["onset_month"]=df_rain_filled["onset_date"].dt.to_period("M")
df_rain_filled["cessation_month"]=df_rain_filled["cessation_date"].dt.to_period("M")
```

```python
#remove the adm2-date entries outside the rainy season for that specific adm2
#df_belowavg_seas only includes data from 2000, so the 1999 entries are not included
list_hist_rain_adm2=[]
for a in df_rain_filled.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
        df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
        seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
        dates_adm2=dates_adm2.union(seas_range)
    list_hist_rain_adm2.append(df_belowavg_seas[(df_belowavg_seas.ADM2_EN==a)&(df_belowavg_seas.date_month.isin(dates_adm2))])
df_belowavg_seas_rain=pd.concat(list_hist_rain_adm2)
```

### Merge dry spells with Seasonal below average rainfall
**NOTE: we currently only include the season (3-month period) if all months are within the rainy season. E.g. if the rainy season ends in April for an admin2, MAM will not be included for that admin2**

```python
df_ds_drymonth["month"]=df_ds_drymonth.ds_fd_m.dt.month
```

```python
df_ds_drymonth.month.unique()
```

```python
df_belowavg_seas_rain
```

```python
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_ds_drymonth_rain=df_ds_drymonth.merge(df_belowavg_seas_rain[["ADM2_EN","date_month"]],how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])
```

```python
#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_ds_drymonth_rain.dry_spell_first_date=df_ds_drymonth_rain.dry_spell_first_date.replace(np.nan,0)
```

```python
#fill the data frame to include all months, also outside the rainy season --> this enables us to take the rolling sum 
#(else e.g. the rolling sum for Nov might include May-June-Nov)
df_ds_drymonth_alldates=df_ds_drymonth_rain.sort_values("date_month").set_index("date_month").groupby('ADM2_EN').resample('M').sum().drop("ADM2_EN",axis=1).reset_index()
```

```python
len(df_ds)
```

```python
#number of entries with dry spell
#aggregated to month, so can be that in one month two dry spells started in same adm2
len(df_ds_drymonth_alldates[df_ds_drymonth_alldates.dry_spell_first_date==1])
```

```python
#compute the rolling sum of months having a dry spell per admin2
s_ds_dryseas=df_ds_drymonth_alldates.sort_values("date_month").set_index("date_month").groupby('ADM2_EN')['dry_spell_first_date'].rolling(3).sum()
#convert series to dataframe
df_ds_dryseas=pd.DataFrame(s_ds_dryseas).reset_index().sort_values(["ADM2_EN","date_month"])
df_ds_dryseas.rename(columns={"dry_spell_first_date":"num_dry_spell_seas"},inplace=True)
```

```python
#never occured that two or three consecutive months in same adm2 experienced a dry spell..
sns.histplot(df_ds_dryseas,x="num_dry_spell_seas")
```

```python
df_ds_dryseas
```

```python
#merge the dry spells with the info if a month had below average rainfall
#merge on right such that only the dates within the rainy season are included, df_ds_dryseas also includes all other months
df_comb_seas=df_ds_dryseas.merge(df_belowavg_seas_rain,how="right",on=["date_month","ADM2_EN"])
```

```python
#remove dates where dry_spell_confirmation is nan, i.e. where rolling sum could not be computed for (first dates)
df_comb_seas=df_comb_seas[df_comb_seas.num_dry_spell_seas.notna()]
```

```python
#set the occurence of a dry spell to true if in at least one of the months of the season (=3 months) a dry spell occured
df_comb_seas["dry_spell"]=np.where(df_comb_seas.num_dry_spell_seas>=1,1,0)
```

```python
#mapping of month to season. Computed by rolling sum, i.e. month indicates last month of season
seasons_rolling={3:"JFM",4:"FMA",5:"MAM",6:"AMJ",7:"MJJ",8:"JJA",9:"JAS",10:"ASO",11:"SON",12:"OND",1:"NDJ",2:"DJF"}
df_comb_seas["season"]=df_comb_seas.date_month.dt.month.map(seasons_rolling)
```

```python
#seasons-adm2s where during at least one of the months a dry spell occured
len(df_comb_seas[df_comb_seas.dry_spell==1])
```

```python
len(df_comb_seas[df_comb_seas.below_average==1])/len(df_comb_seas)
```

```python
len(df_comb_seas[df_comb_seas.dry_spell==1])/len(df_comb_seas)
```

```python
print(f"{len(df_comb_seas[df_comb_seas.below_average==1])/len(df_comb_seas[df_comb_seas.dry_spell==1]):.2f} times as many occurences of below average than dry spells")
```

```python
df_comb_seas
```

#### Analyse distributions below average rainfall and dry spells
There is barely a difference in the distribution of the percentage with below average rainfall when a dry spell occurs or not --> really not possible to separate the two..

```python
#perc_threshold indicates the percentage of area with below average rainfall per admin2-date combination
g=sns.histplot(df_comb_seas,x="perc_threshold",stat="density",common_norm=False,kde=True,hue="dry_spell")
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
g.axes.set_xlabel("Percentage of area of ADM2 with below average seasonal precipitation")
```

```python
#max_cell indicates the cell per adm2-date combination with the maximum value
#however, all cells with above average rainfall are set to -999 
#--> if max cell is positive, all cells in the adm2 have below average rainfall
g=sns.histplot(df_comb_seas,x="max_cell",stat="density",common_norm=False,kde=True,hue="dry_spell")
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
```

#### Correlations
Not really any correlation, but is to be expected from the distributions

```python
df_comb_seas.season.unique()
```

```python
#all seasons-adm2s where a dry spell occured, May was already outside the rainy season --> MAM not included
df_comb_seas[df_comb_seas.dry_spell==1].season.unique()
```

```python
y_target =    df_comb_seas["dry_spell"]
#below_average is defined as perc_threshold>=50 where perc_threshold indicates the percentage of area with below average rainfall
y_predicted = df_comb_seas["below_average"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
# print(cm)
tn,fp,fn,tp=cm.flatten()
print(f"hit rate: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
print(f"miss rate: {round(fp/(tp+fp)*100,1)}% ({fp}/{tp+fp})")
fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,class_names=["No","Yes"])
ax.set_ylabel("Below-average precipitation in ADMIN2 during 3-month period")
ax.set_xlabel("Below-average precipitation in ADMIN2 during 3-month period")
plt.show()
fig.tight_layout()
# fig.savefig(os.path.join(plots_seasonal_dir,"seasonal_below_average_confusionmatrix.png"))
```

```python
#check if difference per admin1
colp_num=3
num_plots=len(df_comb_seas.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_seas.ADM1_EN.unique()):
    y_target =    df_comb_seas.loc[df_comb_seas.ADM1_EN==m,"dry_spell"]
    y_predicted = df_comb_seas.loc[df_comb_seas.ADM1_EN==m,"below_average"]
    
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    print(f"hit rate {m}: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
    print(f"miss rate {m}: {round(fp/(tp+fp)*100,1)}% ({fp}/{tp+fp})")
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during season")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
    ax.set_title(m)
```

```python
#check if difference per season
colp_num=3
num_plots=len(df_comb_seas.date_month.dt.month.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_seas.sort_values(by="date_month").date_month.dt.month.unique()):
    y_target =    df_comb_seas.loc[df_comb_seas.date_month.dt.month==m,"dry_spell"]
    y_predicted = df_comb_seas.loc[df_comb_seas.date_month.dt.month==m,"below_average"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    tn,fp,fn,tp=cm.flatten()
    print(f"hit rate {seasons_rolling[m]}: {round(tp/(tp+fn)*100,1)}% ({tp}/{tp+fn})")
    print(f"miss rate {seasons_rolling[m]}: {round(fp/(tp+fp)*100,1)}% ({fp}/{tn+fp})")
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during season")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during season")
    ax.set_title(seasons_rolling[m])
```

### Correlation number of adm2's with dry spells and below avg precip
The data might be too noisy to detect specific adm2's but there might be a correlation when looking at the total of adm2s having a dry spell/below avg precip during season

Conclusion: there is not😥

```python
#number of adm2s experiencing a dry spell/below average
#these don't have to be the same with the current method of computation
df_numadm=df_comb_seas.groupby("date_month")["dry_spell","below_average"].sum()
```

```python
df_numadm
```

```python
sns.regplot(data = df_numadm, x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)
```

```python
#do the same but per adm1 instead of national
df_nummonthadm=df_comb_seas.groupby(["date_month","ADM1_EN"],as_index=False)[["dry_spell","below_average"]].sum()
```

```python
colp_num=3
num_plots=len(df_comb_seas.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,5))
for i,a in enumerate(df_nummonthadm.ADM1_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.regplot(data = df_nummonthadm[df_nummonthadm.ADM1_EN==a], x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3},ax=ax)
    ax.axes.set_title(a)
#     plt.show()
```

### Distributions below average and dry spells
Understand a little better why no correlation number of adm2's with below average and dry spell

```python
df_comb_seas["year"]=df_comb_seas.date_month.dt.year
```

```python
#for now focus on FMA since this is the season having most dry spels
df_fma=df_comb_seas[df_comb_seas.date_month.dt.month==4]
```

```python
df_fma_bavgy=df_fma[["ADM1_EN","year","dry_spell","below_average"]].groupby(["year","below_average"],as_index=False).sum()
```

```python
#number of dry spells that did and did not have below avg rainfall per year
df_fma_bavgy['below_average_label'] = df_fma_bavgy['below_average'].map({0: 'No', 1: 'Yes'})
g=sns.barplot(x="year", y="dry_spell", hue="below_average_label", data=df_fma_bavgy)#,legend_out=True)
ax=g.axes
plt.xticks(rotation=90)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Number of admin2's with a dry spell during the FMA season")
ax.legend(title="Below average rainfall",loc="upper left")
```

```python
#difference distribution number of dry spells per year, with below and not below avg rainfall
#--> barely difference in distribution
g=sns.histplot(
df_fma_bavgy,x="dry_spell", bins=np.arange(0,df_fma_bavgy.dry_spell.max()+2),stat="count",hue="below_average",common_norm=False,kde=True) #,hue="below_average"
ax=g.axes
# plt.title(v)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlabel(v)
ax.set_xticks(np.arange(0,df_fma_bavgy.dry_spell.max()+1))
plt.show()
```

```python
#stats summary
df_fma_bavgy["Below average rainfall"]=df_fma_bavgy["below_average_label"]
df_fma_bavgy["Number of dry spells"]=df_fma_bavgy["dry_spell"]
df_fma_bavgy[["Below average rainfall","Number of dry spells"]].groupby("Below average rainfall").agg(['sum','mean','min','max'])
```

```python
df_fma_bavgadm=df_fma[["ADM1_EN","year","dry_spell","below_average"]].groupby(["ADM1_EN","year","below_average"],as_index=False).sum()
```

```python
df_fma_bavgadm
```

```python
colp_num=3
num_plots=len(df_comb_seas.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,5))
for i,a in enumerate(df_nummonthadm.ADM1_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.regplot(data = df_nummonthadm[df_nummonthadm.ADM1_EN==a], x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3},ax=ax)
    ax.axes.set_title(a)
#     plt.show()
```

```python
colp_num=3
num_plots=len(df_fma_bavgadm.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,5))
for i,a in enumerate(df_fma_bavgadm.ADM1_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    if len(df_fma_bavgadm[(df_fma_bavgadm.ADM1_EN==a)&(df_fma_bavgadm.dry_spell==1)])>0:
        g=sns.histplot(
        df_fma_bavgadm[df_fma_bavgadm.ADM1_EN==a],x="dry_spell",ax=ax,bins=np.arange(0,df_fma_bavgadm[df_fma_bavgadm.ADM1_EN==a].dry_spell.max()+2),stat="count",hue="below_average",common_norm=False,kde=True) #,hue="below_average"
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(np.arange(0,df_fma_bavgadm[df_fma_bavgadm.ADM1_EN==a].dry_spell.max()+1))
        ax.set_title(a)
```

```python
##Attempt to check for differences per adm2
# boo=df_fma[["ADM1_EN","ADM2_EN","year","dry_spell","below_average"]].groupby(["ADM2_EN","below_average","year"],as_index=False).sum()
# boo[["ADM2_EN","below_average","dry_spell"]].groupby(["ADM2_EN","below_average"]).agg(['sum','mean','min','max'])
```

```python
# #not really informative, so leave out
# #select only entries with below average rainfall
# df_fma_bavg=df_fma[df_fma.below_average==1]
# df_fma_bavg_admy=df_fma_bavg[["ADM1_EN","year","dry_spell"]].groupby(["ADM1_EN","year"],as_index=False).sum()
# #very little adm2s where there was below avg rainfall when there was a dry spell
# g=sns.barplot(x="year", y="dry_spell", hue="ADM1_EN", data=df_fma_bavg_admy)
# ax=g.axes
# plt.xticks(rotation=90)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_title("Number of admin2's with below average rainfall that also had a dry spell")
```

### Observed dryspells and correlation with below average monthly rainfall
Repeat the analysis but now with below average monthly rainfall

```python
df_belowavg_month=pd.read_csv(belowavg_monthly_adm2_path)
#remove day part of date (day doesnt indicate anything with this data and easier for merge)
df_belowavg_month.date_month=pd.to_datetime(df_belowavg_month.date_month).dt.to_period("M")
```

```python
len(df_belowavg_month[df_belowavg_month.below_average==1])/len(df_belowavg_month)
```

```python
df_belowavg_month
```

```python
#path to data start and end rainy season
df_rain=pd.read_csv(os.path.join(country_data_processed_dir,"dry_spells","rainy_seasons_detail_2000_2020_mean.csv"))
df_rain["onset_date"]=pd.to_datetime(df_rain["onset_date"])
df_rain["cessation_date"]=pd.to_datetime(df_rain["cessation_date"])
```

```python
#set the onset and cessation date for the seasons where these are missing 
#(meaning there was no dry spell data from start/till end of the season)
df_rain_filled=df_rain.copy()
df_rain_filled=df_rain_filled[(df_rain_filled.onset_date.notnull())|(df_rain_filled.cessation_date.notnull())]
df_rain_filled[df_rain_filled.onset_date.isnull()]=df_rain_filled[df_rain_filled.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
df_rain_filled[df_rain_filled.cessation_date.isnull()]=df_rain_filled[df_rain_filled.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))
```

```python
df_rain_filled["onset_month"]=df_rain_filled["onset_date"].dt.to_period("M")
df_rain_filled["cessation_month"]=df_rain_filled["cessation_date"].dt.to_period("M")
```

```python
#remove the adm2-date entries outside the rainy season for that specific adm2
#df_belowavg_month only includes data from 2000, so the 1999 entries are not included
list_hist_rain_adm2=[]
for a in df_rain_filled.ADM2_EN.unique():
    dates_adm2=pd.Index([])
    for i in df_rain_filled[df_rain_filled.ADM2_EN==a].season_approx.unique():
        df_rain_adm2_seas=df_rain_filled[(df_rain_filled.ADM2_EN==a)&(df_rain_filled.season_approx==i)]
        seas_range=pd.period_range(df_rain_adm2_seas.onset_date.values[0],df_rain_adm2_seas.cessation_date.values[0],freq="M")
        dates_adm2=dates_adm2.union(seas_range)
    list_hist_rain_adm2.append(df_belowavg_month[(df_belowavg_month.ADM2_EN==a)&(df_belowavg_month.date_month.isin(dates_adm2))])
df_belowavg_month_rain=pd.concat(list_hist_rain_adm2)
```

### Merge dry spells with Monthly below average rainfall

```python
#dataframe that includes all month-adm2 combinations that experienced a dry spell 
#(if several dry spells started within a month in an adm2, these "duplicates" are dropped)
df_ds_drymonth.head()
```

```python
#include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
df_comb_month=df_ds_drymonth.merge(df_belowavg_month_rain,how="outer",left_on=['ADM2_EN','ds_fd_m'],right_on=["ADM2_EN","date_month"])
```

```python
#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_comb_month.dry_spell_first_date=df_comb_month.dry_spell_first_date.replace(np.nan,0)
#date becomes binary
df_comb_month.rename(columns={"dry_spell_first_date":"dry_spell"},inplace=True)
```

```python
df_comb_month["month"]=df_comb_month.date_month.dt.month
```

```python
df_comb_month.head()
```

```python
#seasons-adm2s where during at least one of the months a dry spell occured
len(df_comb_month[df_comb_month.dry_spell==1])
```

```python
len(df_comb_month[df_comb_month.below_average==1])/len(df_comb_month)
```

```python
len(df_comb_month[df_comb_month.dry_spell==1])/len(df_comb_month)
```

```python
print(f"{len(df_comb_month[df_comb_month.below_average==1])/len(df_comb_month[df_comb_month.dry_spell==1]):.2f} times as many occurences of below average than dry spells")
```

#### Analyse distributions below average rainfall and dry spells
There is barely a difference in the distribution of the percentage with below average rainfall when a dry spell occurs or not --> really not possible to separate the two..

```python
#perc_threshold indicates the percentage of area with below average rainfall per admin2-date combination
g=sns.histplot(df_comb_month,x="perc_threshold",stat="density",common_norm=False,kde=True,hue="dry_spell")
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
g.axes.set_xlabel("Percentage of area of ADM2 with below average monthly precipitation")
```

```python
#max_cell indicates the cell per adm2-date combination with the maximum value
#however, all cells with above average rainfall are set to -999 
#--> if max cell is positive, all cells in the adm2 have below average rainfall
g=sns.histplot(df_comb_month,x="max_cell",stat="density",common_norm=False,kde=True,hue="dry_spell")
g.axes.spines['right'].set_visible(False)
g.axes.spines['top'].set_visible(False)
```

#### Correlations
Not really any correlation, but is to be expected from the distributions

```python
y_target =    df_comb_month["dry_spell"]
#below_average is defined as perc_threshold>=50 where perc_threshold indicates the percentage of area with below average rainfall
y_predicted = df_comb_month["below_average"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
# print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during month")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
plt.show()
```

```python
#check if difference per admin1
colp_num=3
num_plots=len(df_comb_month.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_month.ADM1_EN.unique()):
    y_target =    df_comb_month.loc[df_comb_month.ADM1_EN==m,"dry_spell"]
    y_predicted = df_comb_month.loc[df_comb_month.ADM1_EN==m,"below_average"]
    
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during month")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
    ax.set_title(m)
```

```python
#check if difference per season
colp_num=3
num_plots=len(df_comb_month.date_month.dt.month.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(25,20))
for i,m in enumerate(df_comb_month.sort_values(by="date_month").date_month.dt.month.unique()):
    y_target =    df_comb_month.loc[df_comb_month.date_month.dt.month==m,"dry_spell"]
    y_predicted = df_comb_month.loc[df_comb_month.date_month.dt.month==m,"below_average"]

    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted)
    ax = fig.add_subplot(rows,colp_num,i+1)
    plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,axis=ax)
    ax.set_ylabel("Dry spell in ADMIN2 during month")
    ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
    ax.set_title(calendar.month_name[m])
```

### Correlation number of adm2's with dry spells and below avg precip
The data might be too noisy to detect specific adm2's but there might be a correlation when looking at the total of adm2s having a dry spell/below avg precip during season

Conclusion: there is not😥

```python
#number of adm2s experiencing a dry spell/below average
#these don't have to be the same with the current method of computation
df_numadm=df_comb_month.groupby("date_month")["dry_spell","below_average"].sum()
```

```python
df_numadm
```

```python
sns.regplot(data = df_numadm, x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3})#,x_jitter = 0.2, y_jitter = 0.2)
```

```python
#do the same but per adm1 instead of national
df_nummonthadm=df_comb_month.groupby(["date_month","ADM1_EN"],as_index=False)[["dry_spell","below_average"]].sum()
```

```python
colp_num=3
num_plots=len(df_comb_month.ADM1_EN.unique())
if num_plots==1:
    colp_num=1
rows = math.ceil(num_plots / colp_num)
position = range(1, num_plots + 1)
fig=plt.figure(figsize=(20,5))
for i,a in enumerate(df_nummonthadm.ADM1_EN.unique()):
    ax = fig.add_subplot(rows,colp_num,i+1)
    sns.regplot(data = df_nummonthadm[df_nummonthadm.ADM1_EN==a], x = 'dry_spell', y = 'below_average', fit_reg = False,
            scatter_kws = {'alpha' : 1/3},ax=ax)
    ax.axes.set_title(a)
```

### Distributions below average and dry spells
Understand a little better why no correlation number of adm2's with below average and dry spell

```python
df_comb_month["year"]=df_comb_month.date_month.dt.year
```

```python
#for now focus on April since this is the month having most dry spels
df_apr=df_comb_month.copy()#df_comb_month[df_comb_month.date_month.dt.month==3]
```

```python
df_apr_bavgy=df_apr[["ADM1_EN","year","dry_spell","below_average"]].groupby(["year","below_average"],as_index=False).sum()
```

```python
#number of dry spells that did and did not have below avg rainfall per year
df_apr_bavgy['below_average_label'] = df_apr_bavgy['below_average'].map({0: 'No', 1: 'Yes'})
g=sns.barplot(x="year", y="dry_spell", hue="below_average_label", data=df_apr_bavgy)#,legend_out=True)
ax=g.axes
plt.xticks(rotation=90)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title("Number of admin2's with a dry spell during April")
ax.legend(title="Below average rainfall",loc="upper left")
```

```python
#difference distribution number of dry spells per year, with below and not below avg rainfall
#--> barely difference in distribution
g=sns.histplot(
df_apr_bavgy,x="dry_spell", bins=np.arange(0,df_apr_bavgy.dry_spell.max()+2),stat="count",hue="below_average",common_norm=False,kde=True) #,hue="below_average"
ax=g.axes
# plt.title(v)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xlabel(v)
ax.set_xticks(np.arange(0,df_apr_bavgy.dry_spell.max()+1))
ax.set_xlabel("Number of ADM2's with a dry spell")
plt.show()
```

```python
#stats summary
df_apr_bavgy["Below average rainfall"]=df_apr_bavgy["below_average_label"]
df_apr_bavgy["Number of dry spells"]=df_apr_bavgy["dry_spell"]
df_apr_bavgy[["Below average rainfall","Number of dry spells"]].groupby("Below average rainfall").agg(['sum','mean','min','max'])
```

```python
df_apr_bavgadm=df_apr[["ADM1_EN","year","dry_spell","below_average"]].groupby(["ADM1_EN","year","below_average"],as_index=False).sum()
```
