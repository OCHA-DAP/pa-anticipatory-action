---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: antact
    language: python
    name: antact
---

### Understand false alarms
This notebook explores the occurrences of false alarms with the proposed trigger. The goal of this analysis is to better understand if those false alarms were events where the situation was detoriating, even though it didn't correspond with the set definition of a dry spell (<=2mm cumulative rainfall during 14 days). 

The trigger is met if the forecast projects at least 50% probability of the mean value in the Southern region being <=210 mm in the months January or February. 

This notebook compares these events to the occcurrences of a dry spell, a 10day period with <=10mm cumulative rainfall, MVAC's analysis of dry spells, and observed monthly precipitaiton of <=210mm. 

```python
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path
import os
import sys

import pandas as pd
import numpy as np
import calendar
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.statistics import get_return_periods_dataframe
```

#### Set config values

```python
country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",country_iso3)
private_country_data_raw_dir=os.path.join(config.DATA_DIR,config.PRIVATE_DIR,config.RAW_DIR,country_iso3)
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")
monthly_precip_exploration_dir=os.path.join(country_data_exploration_dir,"dryspells","monthly_precipitation")
dry_spell_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","chirps_monthly_total_precipitation_admin1.csv")
mvac_path=os.path.join(private_country_data_raw_dir,"mvac","wfp_dryspells.csv")
tenday_dryspell_path=os.path.join(dry_spell_processed_dir,"false_alarms_deep_dive.csv")
```

```python
sel_adm=["Southern"]
sel_months=[1,2]
sel_leadtime=[2,4]
seas_years=range(2000,2020)

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
lt_str="".join([str(l) for l in sel_leadtime])

#for now using mean cell as this requires one variable less to be set (else need to set percentage of cells)
aggr_meth="mean_cell"
```

```python
#these are determined before in the notebook mwi_ecmwf_monthly_skill_dryspells
#meaning that forecast is classified as predicting a dry spell if >=probability of the members predict <=threshold_perc mm
threshold_perc=210
probability=0.5
```

```python
for_data_path=os.path.join(monthly_precip_exploration_dir,f"mwi_list_dsobs_forblw_th{int(threshold_perc)}_perc_{int(probability*100)}_{adm_str}_{month_str}.csv")
```

#### load forecast data
The forecast is the monthly forecast by ECMWF, which is further explained in the `mwi_ecmwf_monthly_skill_dryspells` notebook. The data loaded here is already processed data. 

```python
df=pd.read_csv(for_data_path,parse_dates=["date_month"])
df.date_month=df.date_month.dt.to_period("M")
df_sel=df[df.leadtime.isin(sel_leadtime)]
```

```python
df_sel.head()
```

#### load observed monthly precipitation data

```python
df_total_month=pd.read_csv(monthly_precip_path,parse_dates=["date_month"])
#remove day part of date (day doesnt indicate anything with this data and easier for merge)
df_total_month.date_month=df_total_month.date_month.dt.to_period("M")
```

```python
df_total_month["obs_below_th"]=np.where(df_total_month[aggr_meth]<=threshold_perc,1,0)
```

```python
df_total_month.head()
```

```python
#merge forecasted and observed monthly precipitation
df_forobs=df_sel.merge(df_total_month[["date_month","ADM1_EN","obs_below_th"]],how="left",on=["date_month","ADM1_EN"])
```

#### load 10 day dry spell data
The definition used to classify a dry spell (14 consecutive days with <=2mm cumulative rainfall) is rather strict. We therefore analyse the occurrence of 10 days with <=10mm rainfall. This is a definition also commonly used and thus might already signify significant damage to crops.   

However, it is hard to determine what exactly the kind of phenomenon is that we would want to anticapate to. But by exploring a looser definition, we can get a better indication of the usefulness of the current trigger. 

```python
df_10dds=pd.read_csv(tenday_dryspell_path)
```

```python
df_10dds["date_month"]=pd.to_datetime([f'{y}-{m}-01' for y, m in zip(df_10dds.year, df_10dds.month)])
df_10dds["date_month"]=df_10dds["date_month"].dt.to_period("M")
```

```python
df_10dds.head()
```

```python
df_forobs=df_forobs.merge(df_10dds[["date_month","with_10d_10mm"]],how="left",on="date_month")
```

#### Load MVAC data
This data was shared privately, and therefore not published in our public data directory. By not showing intermediate outputs, but only showing final results, we keep the detailed privately, as agreed upon.

```python
def ab_ify(row):
    #this means no analysis was done that year --> set to nan
    if row.year < row.start_analysis:
        return np.nan
    #dry spell occured
    elif row.Hazard=="Dry Spell" or row.Hazard=="Floods + Dry Spell":
        return 1
    #analysis was done and no dry spell occured
    else:
        return 0

#load the raw MVAC data
df_mvac=pd.read_csv(mvac_path,delimiter=";",header=2)
#"year" + Hazards are the columns in the dataset which indicate the type of hazard (NaN means no hazard)
hazard_cols=[col for col in df_mvac.columns if "Hazards" in col]
#2011 and 2012 have two columns Hazards, where one of them is empty
hazard_cols.remove("2011 Hazards.1")
hazard_cols.remove("2012 Hazards.1")
df_haz=df_mvac[["District","TA Name 2018 Cencus","Year MVAC Started Analyzing TA"]+hazard_cols].dropna(how="all")
df_haz.rename(columns={"TA Name 2018 Cencus":"TA","Year MVAC Started Analyzing TA":"start_analysis"},inplace=True)
#switch year and hazard, in order to apply wide_to_long
for i in hazard_cols:
    year=i[:4]
    df_haz[f"Hazard {year}"]=df_haz[i]
#get one row per TA and year with the hazard
df_hazl=pd.wide_to_long(df_haz,"Hazard",i=["District","TA"],j="year",sep=" ")[["start_analysis","Hazard"]].reset_index()
#compute whether a dry spell occured
df_hazl['dry_spell'] = df_hazl.apply(lambda x: ab_ify(x),axis=1)
gdf_adm2=gpd.read_file(adm2_bound_path)
df_hazl=df_hazl.merge(gdf_adm2[["ADM2_EN","ADM1_EN"]],how="left",left_on="District",right_on="ADM2_EN")
df_mvac_year=df_hazl.groupby(["year","ADM1_EN"])["dry_spell"].agg(ds_sum="sum",ds_count="count").reset_index()
#compute percentage of analyzed TAs experiencinga dry spell per year
df_mvac_year["perc"]=df_mvac_year.ds_sum/df_mvac_year.ds_count*100
# MVAC's year is the year during which thre crops were harvested, i.e. the end of the rainy season
#While for the obs dry spells from CHIRPS we use season_approx as the year the rainy season started --> add that variable to the dataset
df_mvac_year.rename(columns={"year":"season_approx_mvac"},inplace=True)
df_mvac_year["season_approx"]=df_mvac_year.season_approx_mvac-1
```

```python
df_mvac_year[df_mvac_year.ADM1_EN=="Southern"].head()
```

```python
#plot the percentages across years (ugly just for inspection)
df_mvac_year[df_mvac_year.ADM1_EN=="Southern"].plot(x="season_approx",y="perc",kind="bar",figsize=(5,2),legend=False,title="percentage of area experiencing a dry spell")
```

```python
#start month of the rainy season
#the start month is based on FewsNet's calendar (https://fews.net/sites/default/files/styles/large/public/seasonal-calendar-malawi.png)
start_rainy_seas=10
#season approx indicates the year during which the rainy season started
#this is done because it can start during one year and continue the next calendar year
#we therefore prefer to group by rainy season instead of by calendar year
df_forobs["season_approx"]=np.where(df_forobs.date_month.dt.month>=start_rainy_seas,df_forobs.date_month.dt.year,df_forobs.date_month.dt.year-1)
```

```python
#group to rainy season, as mvac is yearly data
df_forobs_seas=df_forobs.groupby(["season_approx","leadtime","ADM1_EN"],as_index=False).sum()
```

```python
df_forobs_seas=df_forobs_seas.merge(df_mvac_year[["season_approx","ADM1_EN","perc"]],how="left",on=["season_approx","ADM1_EN"])
df_forobs_seas.rename(columns={"perc":"perc_ds_mvac"},inplace=True)
```

```python
df_rp_emp=get_return_periods_dataframe(df_forobs_seas.loc[df_forobs_seas.perc_ds_mvac.notnull()],"perc_ds_mvac",[1.5, 2, 3, 4, 5, 10],method="empirical",show_plots=True)
```

```python
df_rp_emp["rp_round"]=5*np.round(df_rp_emp["rp"]/5)
```

```python
df_rp_emp
```

```python
df_forobs_seas.loc[df_forobs_seas.perc_ds_mvac.notnull(),"mvac_rp3"]=np.where(df_forobs_seas[df_forobs_seas.perc_ds_mvac.notnull()].perc_ds_mvac>=df_rp_emp.loc[3,"rp_round"],1,0)
```

```python
#only select the years for which mvac data is available
df_forobs_mvac=df_forobs_seas[df_forobs_seas.mvac_rp3.notnull()]
```

### Analyse false alarms
Compare the false alarms to the occurrence of <=210 mm observed precipitation and 10 consecutive days with cumulative <=10mm on a monthly frequency. Compare the false alarms with the MVAC data on a yearly frequency. 
The goal of this analysis is to better understand if those false alarms were events where the situation was detoriating, even though it didn't correspond with the set definition of a dry spell (<=2mm cumulative rainfall during 14 days). 

```python
df_forobs_seas[df_forobs_seas.leadtime==2].tail()
```

```python
#compute the metrics per leadtime
df_metrics_lt=pd.DataFrame(sel_leadtime,columns=["leadtime"]).set_index(['leadtime'])
for l in sel_leadtime:
    #different dataframes for the metrics on monthly and yearly frequency
    df_forobs_lt=df_forobs[df_forobs.leadtime==l]
    df_forobs_seas_lt=df_forobs_seas[df_forobs_seas.leadtime==l]
    df_forobs_mvac_lt=df_forobs_mvac[df_forobs_mvac.leadtime==l]
    df_metrics_lt.loc[l,"trigger_month"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)])
    df_metrics_lt.loc[l,"hit_month"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)&(df_forobs_lt.dry_spell==1)])
    df_metrics_lt.loc[l,"false_alarms_month"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)&(df_forobs_lt.dry_spell==0)])
    for m in sel_months:
        df_metrics_lt.loc[l,f"false_alarms_{m}"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)&(df_forobs_lt.dry_spell==0)&(df_forobs_lt.date_month.dt.month==m)])
        df_metrics_lt.loc[l,f"false_alarms_{m}_obs"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)&(df_forobs_lt.dry_spell==0)&(df_forobs_lt.date_month.dt.month==m)&(df_forobs_lt.obs_below_th==1)])
        df_metrics_lt.loc[l,f"false_alarms_{m}_10d"]=len(df_forobs_lt[(df_forobs_lt.for_below_th>=1)&(df_forobs_lt.dry_spell==0)&(df_forobs_lt.date_month.dt.month==m)&(df_forobs_lt.with_10d_10mm==1)])
    df_metrics_lt.loc[l,"false_alarms_year"]=len(df_forobs_seas_lt[(df_forobs_seas_lt.for_below_th>=1)&(df_forobs_seas_lt.dry_spell==0)])
    #false alarms during years with mvac data
    df_metrics_lt.loc[l,"false_alarms_year_mvac"]=len(df_forobs_mvac_lt[(df_forobs_mvac_lt.for_below_th>=1)&(df_forobs_mvac_lt.dry_spell==0)])
    df_metrics_lt.loc[l,"false_alarms_mvac"]=len(df_forobs_mvac_lt[(df_forobs_mvac_lt.for_below_th>=1)&(df_forobs_mvac_lt.dry_spell==0)&(df_forobs_mvac_lt.mvac_rp3==1)])
    df_metrics_lt.loc[l,"fa_mvac_perc"]=df_metrics_lt.loc[l,"false_alarms_mvac"]/len(df_forobs_mvac_lt[(df_forobs_mvac_lt.for_below_th>=1)&(df_forobs_mvac_lt.dry_spell==0)])
```

```python
print(f"total number of months that were included in the analysis: {len(df_forobs_lt)}")
```

```python
#summary metrics
df_metrics_lt
```

### Written summary


Leadtime=4 months, forecasting for Jan and Feb (leadtime of 4 months means that the forecast released mid-Oct is projecting the situation for January)
- 22/40 (55%) of the months the trigger was met
- 19/22 (86%) of the times the trigger was met, it was a false alarm. 
- 14/19 (74%) false alarms occurred during february, 5 during January (26%)
- The forecast skill is lower during January than February. During 1/5 (20%) false alarms of January, the observed precipitation was below 210 mm, for February this was for 11/14 (80%)
- 2/5 (40%) false alarms in January co-occured with a 10d period with no more than 10 mm precipitaiton. In February this was 11/14 (80%). 
- The false alarms were spread across 13 years
- 6 of the years with a false alarm occurred during years MVAC data was available
- 3/6 (50%) of these years with a false alarm did occurr during years where more than 85% of the region experienced a dry spell according to MVAC


Leadtime=2 months, forecasting for Jan and Feb (leadtime of 2 months means that the forecast released mid-Dec is projecting the situation for January)
- 13/40 (33%) of the months the trigger was met
- 11/13 (85%) of the times the trigger was met, it was a false alarm. 
- 8/11 (73%) false alarms occurred during february, 3 during January (27%)
- The forecast skill is lower during January than February. During 1/3 (33%) false alarms of January, the observed precipitation was below 210 mm, for February this was for 6/8 (75%)
- 1/3 (33%) false alarms in January co-occured with a 10d period with no more than 10 mm precipitaiton. In February this was 5/8 (63%). 
- The false alarms were spread across 9 years
- 4 of the years with a false alarm occurred during years MVAC data was available
- 2/4 (50%) of these years with a false alarm did occurr during years where more than 85% of the region experienced a dry spell according to MVAC

```python

```
