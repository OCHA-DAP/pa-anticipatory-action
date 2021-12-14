### Distributions of values with different aggregation methods
This notebook is similair to `ecmwf_cds_compare_aggr.md`. However, here we focus on the data directly retrieved from ECMWF's API. This data has a higher resolution of 0.4 degrees compared to the 1 degree of CDS. 

We only recently got access to this data, so in this notebook we compare the values between this high resolution data and the CDS data when aggregating to the Southern region. 

A written summary of this analysis can be found [here](https://docs.google.com/presentation/d/1jBdZEnzWwOKoFA9-h4xR4x0FmTyMRDCNBkWGKptfZs8/edit)

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
import pandas as pd
import numpy as np

import calendar
import seaborn as sns
import math
import geopandas as gpd
import altair as alt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import processing
reload(processing)
```

#### Set config values

```python
country_iso3="mwi"
config=Config()
parameters = config.parameters(country_iso3)

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.RAW_DIR,country_iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,country_iso3)

dry_spells_processed_dir = os.path.join(country_data_processed_dir, "dry_spells", f"v{parameters['version']}")

adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
all_dry_spells_list_path=os.path.join(dry_spells_processed_dir,"full_list_dry_spells_2000_2021.csv")
monthly_precip_path=os.path.join(country_data_processed_dir,"chirps","chirps_monthly_total_precipitation_admin1.csv")
```

```python
#set plot colors
ds_color='#F2645A'
no_ds_color='#CCE5F9'
```

### Define variables

```python
start_year=2000
end_year=2020
#just locking the date to keep the analysis the same even though data is added
#might wanna delete again later
start_date="1-1-2000"
end_date="2-1-2020"
start_rainy_seas=10
adm_level=1
```

```python
#define areas, months, years, and leadtimes of interest for the trigger
#for this analysis we are only interested in the southern region during a few months that the dry spells have the biggest impact
sel_adm=["Southern"]
sel_months=[1,2]
seas_years=range(start_year,end_year)

adm_str="".join([a.lower() for a in sel_adm])
month_str="".join([calendar.month_abbr[m].lower() for m in sel_months])
```

### Load observational data

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
#load the monthly precipitation data
df_obs_month=pd.read_csv(monthly_precip_path,parse_dates=["date"])
df_obs_month["date_month"]=df_obs_month.date.dt.to_period("M")
df_obs_month["season_approx"]=np.where(df_obs_month.date.dt.month>=start_rainy_seas,df_obs_month.date.dt.year,df_obs_month.date.dt.year-1)

#select relevant admins and months
df_obs_month_sel=df_obs_month[(df_obs_month.ADM1_EN.isin(sel_adm))&(df_obs_month.date.dt.month.isin(sel_months))&(df_obs_month.season_approx.isin(seas_years))]
```

```python
df_ds=load_dryspell_data(all_dry_spells_list_path)
```

### Load the forecast data
And select the data of interest

```python
stat_col_for="mean_ADM1_PCODE"
```

```python
def load_forecast_data(iso3,config,use_unrounded_area_coords,resolution,all_touched,start_date,end_date,adm_level,stat_col_for,source_cds=True):
    #read the ecmwf forecast per adm1 per date and concat all dates
    date_list=pd.date_range(start=start_date, end=end_date, freq='MS')
    all_files=[processing.get_stats_filepath(country_iso3,config,date,resolution=resolution,adm_level=adm_level,use_unrounded_area_coords=use_unrounded_area_coords,source_cds=source_cds,all_touched=all_touched) for date in date_list]

    df_from_each_file = (pd.read_csv(f,parse_dates=["date"]) for f in all_files)
    df_for   = pd.concat(df_from_each_file, ignore_index=True)
    #for earlier dates, the model included less members --> values for those members are nan --> remove those rows
    df_for = df_for[df_for[stat_col_for].notna()]
    #season approx indicates the year during which the rainy season started
    #this is done because it can start during one year and continue the next calendar year
    #we therefore prefer to group by rainy season instead of by calendar year
    df_for["season_approx"]=np.where(df_for.date.dt.month>=start_rainy_seas,df_for.date.dt.year,df_for.date.dt.year-1)
   
    return df_for
```

```python
def sel_forecast_data(df_for,sel_adm,sel_months,seas_years):
    return df_for[(df_for.ADM1_EN.isin(sel_adm))&(df_for.date.dt.month.isin(sel_months))&(df_for.season_approx.isin(seas_years))]
```

```python
def compute_quantile(df,probability,adm_col="ADM1_EN"):
    df_quant=df.groupby(["date",adm_col,"leadtime"],as_index=False).quantile(probability)
    df_quant["date_month"]=df_quant.date.dt.to_period("M")
    return df_quant
```

```python
def combine_for_obs(df_for,df_obs):
    #include all dates present in the observed rainfall df but not in the dry spell list, i.e. where no dryspells were observed, by merging outer
    df_obs_for=df_obs.merge(df_for,how="right",on=["ADM1_EN","date_month"])
    df_obs_for.loc[:,"dry_spell"]=df_obs_for.dry_spell.replace(np.nan,0).astype(int)
    #extract month and names for plotting
    df_obs_for["month"]=df_obs_for.date_month.dt.month
    df_obs_for["month_name"]=df_obs_for.month.apply(lambda x: calendar.month_name[x])
    df_obs_for["month_abbr"]=df_obs_for.month.apply(lambda x: calendar.month_abbr[x])
#     df_obs_for_labels=df_obs_for.replace({"dry_spell":{0:"no",1:"yes"}}).sort_values("dry_spell",ascending=True)
    return df_obs_for
```

```python
#define the parameters for the different aggregation metods
meth_dict={"cds_unrounded_center":{"use_unrounded_area_coords":True,"resolution":None,"all_touched":False,"source_cds":True},
"api_center":{"use_unrounded_area_coords":False,"resolution":None,"all_touched":False,"source_cds":False},
"api_weightavg":{"use_unrounded_area_coords":False,"resolution":0.05,"all_touched":False,"source_cds":False},
"api_alltouched":{"use_unrounded_area_coords":False,"resolution":None,"all_touched":True,"source_cds":False},
}
```

Load the forecast data for different methods of aggregation

```python
for key, values in meth_dict.items():
    values["df"]=load_forecast_data(country_iso3,config,values["use_unrounded_area_coords"],
                                    resolution=values["resolution"],all_touched=values["all_touched"],source_cds=values["source_cds"],
                                    start_date=start_date,end_date=end_date,adm_level=adm_level,stat_col_for=stat_col_for)
    values["df_sel"]=sel_forecast_data(values["df"],sel_adm=sel_adm,sel_months=sel_months,seas_years=seas_years)
    values["df_sel_quant"]=compute_quantile(values["df_sel"],probability=0.5)
    values["df_obs_for"]=combine_for_obs(values["df_sel_quant"],df_ds)
```

```python
#check that same number of months in observed and forecasted
print("number of months in forecasted data",len(meth_dict["cds_unrounded_center"]["df_sel"].date.unique()))
print("number of months in observed data",len(df_obs_month_sel.date.unique()))
```

```python
#combine the different methods to one dataframe for plotting
df_list=[]
sel_cols=["ADM1_EN","date","month_name","leadtime",stat_col_for,"method","dry_spell"]
for key, values in meth_dict.items():
    values["df_rename"]=values["df_obs_for"].copy()
    values["df_rename"]["method"]=key
    df_list.append(values["df_rename"][sel_cols])
df_comb=pd.concat(df_list)
#compute medians, used for plotting
df_med=df_comb.groupby(["leadtime","month_name","method"],as_index=False).median()
df_med.mean_ADM1_PCODE=df_med.mean_ADM1_PCODE.round()
df_med.rename(columns={"mean_ADM1_PCODE":"median_for"},inplace=True)
df_comb_med=df_comb.merge(df_med[["leadtime","month_name","method","median_for"]],how="left",on=["leadtime","month_name","method"])
```

### Plotting


#### Diverging bar chart

```python
df_comb_pivot=df_comb.pivot(index=["ADM1_EN","date","month_name","leadtime"],columns="method",values="mean_ADM1_PCODE").reset_index()
df_ds["date"]=df_ds.date_month.dt.to_timestamp()
df_comb_pivot=df_ds[["date","ADM1_EN","dry_spell"]].merge(df_comb_pivot,how="right",on=["ADM1_EN","date"])
df_comb_pivot.loc[:,"dry_spell"]=df_comb_pivot.dry_spell.replace(np.nan,0).astype(int)
```

```python
def plt_div_bar(meth_comb,colp_num=2,figsize=(30,10)):
    sel_lt=3
    num_plots=len(meth_comb)
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)
    fig=plt.figure(figsize=figsize)
    df_comb_plot=df_comb_pivot.copy()
    df_comb_plot_sel=df_comb_plot[df_comb_plot.leadtime==sel_lt]

    for i, meth in enumerate(meth_comb):
        ax = fig.add_subplot(rows,colp_num,i+1)
        val_col=f"{meth[0]}-{meth[1]}"
        df_comb_plot_sel[val_col]=df_comb_plot_sel[meth[0]]-df_comb_plot_sel[meth[1]]
        df_comb_plot_sel.sort_values(val_col,inplace=True,ascending=False)
        df_comb_plot_sel["colors"]=np.where(df_comb_plot_sel["dry_spell"]==1,ds_color,no_ds_color)
        df_comb_plot_sel=df_comb_plot_sel.reset_index(drop=True)
        # Plotting the horizontal lines
        plt.hlines(y=df_comb_plot_sel.index
                , xmin=0, xmax=df_comb_plot_sel[val_col],
                   color=df_comb_plot_sel.colors,
                   linewidth=5)

        # Decorations
        # Setting the labels of x-axis and y-axis
        plt.gca().set(ylabel='Month', xlabel=f'Difference (mm),{val_col.replace("-"," minus ")}')

        # Setting Date to y-axis
        plt.yticks(df_comb_plot_sel.index, df_comb_plot_sel.date.dt.strftime("%Y-%m"), fontsize=12)
        ax.xaxis.label.set_size(16)

        plt.title(f'{val_col.replace("-"," minus ")}', fontdict={
                  'size': 20})
        plt.grid(linestyle='--', alpha=0.5)
        ds_patch = mpatches.Patch(color=ds_color, label="dry spell")
        nods_patch = mpatches.Patch(color=no_ds_color, label="no dry spell")

        ax.set_xlim(-20,55)
#         ax.set_xlim(-40,35)

        plt.legend(handles=[ds_patch, nods_patch])
    fig.suptitle("Difference in forecasted precipitation at leadtime=3 for different aggregation methods",fontsize=30)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
```

```python
meth_comb=[[m,"cds_unrounded_center"] for m in ["api_center","api_weightavg","api_alltouched"]]
plt_div_bar(meth_comb,colp_num=3)
```

#### Boxplot
Only 20 datapoints per boxplot, so boxplot might not be the best method

```python
plot=alt.Chart().mark_boxplot(size=50).encode(
    x=alt.X('method:N',sort=df_comb.method.unique()),
    y=alt.Y(stat_col_for,scale=alt.Scale(zero=False),title="Forecasted precipitation"),
#     facet=alt.Facet(column="month_name:N",sort=df_comb.month_name.unique())
).properties(width=400,height=400)
plot.facet(column=alt.Column("month_name:N",sort=df_comb.month_name.unique(),title="Month"),row=alt.Row("leadtime:N",sort=[2,3,4]),data=df_comb[df_comb.leadtime.isin([2,3,4])],
           title=["Distribution of forecasted precipitation for different methods, facetted by leadtime and month", "Each boxplot only contains 20 datapoints"])
```

#### Histogram
Stacked based on occurrence of dry spell    
Due to so little observations it is not really a distrubtion which makes it hard to compare


For some unexplainable reason the text is not readable at all. Almost seems several layers are printed on top of each other

```python
#is stacked!
df_comb_med["ds_str"]=np.where(df_comb_med.dry_spell==1,"yes","no")
histo = alt.Chart().mark_bar().encode(
    x=alt.X("mean_ADM1_PCODE",bin=alt.Bin(step=10),title="Forecasted precipitation"),
    y="count()",
    color=alt.Color('ds_str:N',scale=alt.Scale(range=[no_ds_color,ds_color]),legend=alt.Legend(title="Dry spell")),
)
med_line = alt.Chart().mark_rule(color="purple").encode(
    x='median(mean_ADM1_PCODE)')
#TODO: for some reason text not readable, no idea why..
med_text = alt.Chart().mark_text(align='left', dx=90, dy=-130,fontSize=15,color="purple").encode(
    text="label:N",
).transform_calculate(label="Median: " + alt.datum.median_for)

(histo+med_line+med_text).facet(column=alt.Column("month_name:N",sort=df_comb_med.month_name.unique(),title="Month"), row=alt.Row('method:N',sort=df_comb_med.method.unique()),
                                       data=df_comb_med[df_comb_med.leadtime==3],title="Stacked histogram of forecasted precipitation for leadtime=3, facetted by month and aggregation method")
```

```python
# #select methods for plotting of slides
# #is stacked!
# sel_meth=["api_center","api_weightavg","api_alltouched"]
# df_comb_med["ds_str"]=np.where(df_comb_med.dry_spell==1,"yes","no")
# histo = alt.Chart().mark_bar().encode(
#     x=alt.X("mean_ADM1_PCODE",bin=alt.Bin(step=10),title="Forecasted precipitation"),
#     y="count()",
#     color=alt.Color('ds_str:N',scale=alt.Scale(range=[no_ds_color,ds_color]),legend=alt.Legend(title="Dry spell")),
# )
# med_line = alt.Chart().mark_rule(color="purple").encode(
#     x='median(mean_ADM1_PCODE)')
# #TODO: for some reason text not readable, no idea why..
# med_text = alt.Chart().mark_text(align='left', dx=90, dy=-130,fontSize=15,color="purple").encode(
#     text="label:N",
# ).transform_calculate(label="Median: " + alt.datum.median_for)

# (histo+med_line+med_text).facet(column=alt.Column("month_name:N",sort=df_comb_med.month_name.unique(),title="Month"), row=alt.Row('method:N',sort=df_comb_med.method.unique()),
#                                        data=df_comb_med[(df_comb_med.leadtime==3)&(df_comb_med.method.isin(sel_meth))],title="Stacked histogram of forecasted precipitation for leadtime=3, facetted by month and aggregation method")
```

```python
# #select methods for plotting of slides
# #is stacked!
# sel_meth=["cds_unrounded_center","api_center"]
# df_comb_med["ds_str"]=np.where(df_comb_med.dry_spell==1,"yes","no")
# histo = alt.Chart().mark_bar().encode(
#     x=alt.X("mean_ADM1_PCODE",bin=alt.Bin(step=10),title="Forecasted precipitation"),
#     y="count()",
#     color=alt.Color('ds_str:N',scale=alt.Scale(range=[no_ds_color,ds_color]),legend=alt.Legend(title="Dry spell")),
# )
# med_line = alt.Chart().mark_rule(color="purple").encode(
#     x='median(mean_ADM1_PCODE)')
# #TODO: for some reason text not readable, no idea why..
# med_text = alt.Chart().mark_text(align='left', dx=90, dy=-130,fontSize=15,color="purple").encode(
#     text="label:N",
# ).transform_calculate(label="Median: " + alt.datum.median_for)

# (histo+med_line+med_text).facet(column=alt.Column("month_name:N",sort=df_comb_med.month_name.unique(),title="Month"), row=alt.Row('method:N',sort=df_comb_med.method.unique()),
#                                        data=df_comb_med[(df_comb_med.leadtime==3)&(df_comb_med.method.isin(sel_meth))],title="Stacked histogram of forecasted precipitation for leadtime=3, facetted by month and aggregation method")
```