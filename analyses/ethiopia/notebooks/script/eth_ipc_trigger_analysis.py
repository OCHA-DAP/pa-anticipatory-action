#!/usr/bin/env python
# coding: utf-8

# ### Test different thresholds of IPC levels for FewsNet and Global IPC
# For the anticipatory action framework, we want to define the trigger mechanism based on data. One of the possible data sources are IPC levels. Based on the historical analysis of FewsNet and Global IPC, and conversations with partners, different triggers were tested. This notebook provides a subset of tested triggers and the code to easily test any triggers of interest.    
# 
# IPC trigger design as of 08-10-2020:   
# EITHER: At least 20% population of one or more ADMIN1 regions projected at IPC4+ in 3 months   
# OR:    
# At least 30% of ADMIN1 population projected at IPC3+ AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ in 3 months compared to current state
# 
# Main experimenting was done with FewsNet due to more historical data. The most relevant triggers were also analysed for Global IPC

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
print(path_mod)
sys.path.append(path_mod)
# from indicators.drought.iri_rainfallforecast import get_iri_data,statistics_alldates
# from indicators.drought.icpac_rainfallforecast import get_icpac_data
# from indicators.drought.nmme_rainfallforecast import get_nmme_data
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.ipc_definemetrics import define_trigger_percentage, define_trigger_increase, define_trigger_increase_rel
from utils_general.plotting import plot_boundaries_binary


country="ethiopia"
#suffix of filenames
suffix=""
config=Config()
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
fewsnet_dir = os.path.join(country_folder, config.DATA_DIR, config.FEWSWORLDPOP_PROCESSED_DIR)


# ### FewsNet

# ### Explore thresholds on admin1 level

admin_level=1
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)
adm1_bound_path= os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])


df_fadm=pd.read_csv(os.path.join(fewsnet_dir,fewsnet_filename))
df_fadm["date"]=pd.to_datetime(df_fadm["date"])
df_fadm["year"]=df_fadm["date"].dt.year
df_fadm["month"]=df_fadm["date"].dt.month


df_fadm.tail()


# TODO: The updates of 2020-09-01 and 2020-08-01 don't include any CS data! For the analysis CS data of 2020-06 should be used. Previously just added that data to the raw fewsnet data files, but would prefer to add it here

#TODO: replace CS values for 2020-08 and 2020-09 with those of 2020-06
CS_cols=[c for c in df_fadm.columns if "CS" in c]
# date_replace="2020-06-01"
# blub=df_fadm.copy()
for d in ["2020-08-01","2020-09-01"]:
    for i in df_fadm.ADMIN1.unique():
        df_fadm.loc[((df_fadm.date==d)&(df_fadm.ADMIN1==i)),CS_cols]=np.nan
        
#         print(i)
# #         print(df_fadm.loc[(df_fadm.date==date_replace)&(df_fadm.ADMIN1==i),CS_cols])
# #         df_fadm.loc[((df_fadm.date==d)&(df_fadm.ADMIN1==i)),CS_cols]=df_fadm.loc[((df_fadm.date==date_replace)&(df_fadm.ADMIN1==i)),CS_cols]
#         print(blub.loc[((blub.date==date_replace)&(blub.ADMIN1==i)),CS_cols])
# #         print(len([blub.loc[((blub.date==date_replace)&(blub.ADMIN1==i)),CS_cols]]))
# #         print(len(blub.loc[((blub.date==d)&(blub.ADMIN1==i)),CS_cols]))
#         blub.loc[((blub.date==d)&(blub.ADMIN1==i)),CS_cols] = blub.loc[((blub.date==d)&(blub.ADMIN1==i)),CS_cols].update(blub.loc[((blub.date==date_replace)&(blub.ADMIN1==i)),CS_cols])
# #         blub.loc[((blub.date==d)&(blub.ADMIN1==i)),CS_cols]=blub.loc[((blub.date==date_replace)&(blub.ADMIN1==i)),CS_cols].values #[blub.loc[((blub.date==date_replace)&(blub.ADMIN1==i)),CS_cols]]
#         print(blub.loc[((blub.date==d)&(blub.ADMIN1==i)),CS_cols])
        


#never been or forecasted to be IPC 5
print("CS 5",df_fadm.CS_5.unique())
print("ML1 5", df_fadm.ML1_5.unique())


#most current numbers
df_fadm.loc[df_fadm.date==df_fadm.date.max(),["date",f"ADMIN{admin_level}","perc_CS_3p","perc_CS_4","perc_ML1_3p","perc_ML1_4","perc_ML2_3p","perc_ML2_4"]]


#get yes/no for different thresholds, i.e. column value for row will be 1 if threshold is met and 0 if it isnt
df_fadm["trigger_CS_3_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",3,20),axis=1)
df_fadm["trigger_CS_3_40"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",3,40),axis=1)
df_fadm["trigger_CS_4_2"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",4,2.5),axis=1)
df_fadm["trigger_CS_4_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",4,20),axis=1)
df_fadm["trigger_CS_4_10"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",4,10),axis=1)
df_fadm["trigger_CS_4_1"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"CS",4,0.1),axis=1)
df_fadm["trigger_ML1_3_5"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,5),axis=1)
df_fadm["trigger_ML1_4_2"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",4,2.5),axis=1)
df_fadm["trigger_ML1_4_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_fadm["trigger_ML1_3_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,20),axis=1)
df_fadm["trigger_ML1_3_30"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_fadm["trigger_ML1_3_5ir"]=df_fadm.apply(lambda x: define_trigger_increase_rel(x,3,5),axis=1)
df_fadm["trigger_ML1_3_40ir"]=df_fadm.apply(lambda x: define_trigger_increase_rel(x,3,40),axis=1)
df_fadm["trigger_ML1_3_70ir"]=df_fadm.apply(lambda x: define_trigger_increase_rel(x,3,70),axis=1)
df_fadm["trigger_ML1_3_5i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
df_fadm["trigger_ML1_3_10i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,10),axis=1)
df_fadm["trigger_ML1_3_20i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,20),axis=1)
df_fadm["trigger_ML1_3_30i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,30),axis=1)
df_fadm["trigger_ML1_3_40i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,40),axis=1)
df_fadm["trigger_ML1_3_50i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,50),axis=1)
df_fadm["trigger_ML1_3_70i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,70),axis=1)
df_fadm["trigger_ML2_4_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML2",4,20),axis=1)
df_fadm["trigger_ML2_3_30"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML2",3,30),axis=1)
df_fadm["trigger_ML2_3_5i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML2",3,5),axis=1)


#initialize dict with all the analyses
dict_fan={}


#currently (Oct 2020) selected trigger
df_an1=df_fadm.loc[(df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1))]
display(df_an1.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4p','perc_CS_3p','perc_ML1_3p','perc_inc_ML1_3p'].agg(lambda x: list(x)))
dict_fan["an1"]={"df":df_an1,"trig_cols":["ML1_3p","CS_3p","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1 OR (At least 30% of ADMIN1 population projected at IPC3+  AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ compared to current state)"}


#Analysis 2: At least 20% of ADMIN1 population at IPC4+ in ML1
df_an2 = df_fadm.loc[(df_fadm["trigger_ML1_4_20"]==1)]
display(df_an2.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4'].agg(lambda x: list(x)))
dict_fan["an2"]={"df":df_an2,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1"}


#Analysis 3: At least 30% of ADMIN1 population projected to be at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months
df_an3 = df_fadm.loc[(df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1)]
display(df_an3.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
dict_fan["an3"]={"df":df_an3,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"At least 30% of ADMIN1 population in ML1 at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months"}


# #Some previously tested triggers
# #More triggers were analysed, the ones below substitute a subset which shows the variety of investigated triggers
# #Analysis 4: 20% IPC3+ (current situation) + 2.5% IPC4+ (current situation)
# df_an4 = df_fadm.loc[(df_fadm['trigger_CS_3_20']==1)&(df_fadm['trigger_ML1_4_2']==1)]
# dict_fan["an4"]={"df":df_an4,"trig_cols":["CS_3","CS_4","ML1_4"],"desc":"20% IPC3+ (current situation) + 2.5% IPC4+ (current situation)"}

# #Analysis 5: 20% IPC3+ (current situation) + (2.5% IPC4+ (current situation) OR 5% RELATIVE increase in IPC3+ (ML1))
# df_an5 = df_fadm.loc[(df_fadm['trigger_CS_3_20']==1)&((df_fadm['trigger_ML1_4_2']==1)| (df_fadm['trigger_ML1_3_5ir'] == 1))]
# dict_fan["an5"]={"df":df_an5,"trig_cols":["CS_3","CS_4","ML1_4"],"desc":"20% IPC3+ (current situation) + (2.5% IPC4+ (current situation) OR 5% RELATIVE increase in IPC3+ (ML1))"}

# #Analysis 6: 20% IPC3+ (current situation) + 2.5% IPC4+ (current situation) + 5% RELATIVE increase in IPC3+ (ML1)
# df_an6 = df_fadm.loc[(df_fadm['trigger_CS_3_20']==1)&(df_fadm['trigger_CS_4_2']==1) & (df_fadm['trigger_ML1_3_5ir'] == 1)]
# dict_fan["an6"]={"df":df_an6,"trig_cols":["CS_3","CS_4","ML1_4"],"desc":"20% IPC3+ (current situation) + 2.5% IPC4+ (current situation) + 5% RELATIVE increase in IPC3+ (ML1)"}

# #Analysis 7: IPC4 at 20% (current situation)
# df_an7 = df_fadm.loc[df_fadm['trigger_CS_4_20']==1]
# dict_fan["an7"]={"df":df_an7,"trig_cols":["CS_4"],"desc":"IPC4 at 20% (current situation)"}

# #Analysis 8: 5% increase in IPC3+ (ML1)
# df_an8 = df_fadm.loc[(df_fadm['trigger_ML1_3_5i']==1)]
# dict_fan["an8"]={"df":df_an8,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"5% increase in number of people in IPC3+ (ML1)"}

# #Analysis 9: At least 20% of ADMIN1 population projected to be at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months
# df_an9 = df_fadm.loc[(df_fadm["trigger_ML1_3_20"]==1) & (df_fadm["trigger_ML1_3_5i"]==1)]
# dict_fan["an9"]={"df":df_an9,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"At least 20% of ADMIN1 population in ML1 at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months"}

# #Analysis 10: At least 20% of population projected in ML2 in IPC4+
# df_an10=df_fadm.loc[(df_fadm['trigger_ML2_4_20']==1)]
# dict_fan["an10"]={"df":df_an10,"trig_cols":["ML2_4"],"desc":"20% of population projected in ML2 in IPC4+"}

#Analysis 11: At least 20% of population projected in ML2 in IPC4+ OR (30% in ML2 in IPC3+ AND 5% increase in IPC3+ in ML2 compared to CS)
df_an11=df_fadm.loc[(df_fadm["trigger_ML2_4_20"]) | ((df_fadm["trigger_ML2_3_30"]==1)&(df_fadm["trigger_ML2_3_5i"]==1))]
display(df_an11.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_CS_3p','perc_ML2_3p','perc_ML2_4'].agg(lambda x: list(x)))
dict_fan["an11"]={"df":df_an11,"trig_cols":["ML2_3","ML2_4"],"desc":"20% in ML2 in IPC4 OR (30% in ML2 in IPC3+ AND 5% increase in IPC3+ in ML2 compared to CS)"}


#plot all analysis in nicer format
for k in dict_fan.keys():
    d=dict_fan[k]["desc"]
    num_k=k.replace("an","")
    print(f"Analysis {num_k}: FewsNet, {d}")
    df=dict_fan[k]["df"]
    df_grouped=df.groupby(['date','year', 'month'], as_index=False)[f"ADMIN{admin_level}"].agg(lambda x: list(x))
    dict_fan[k]["df_group"]=df_grouped
    df_grouped[f"Regions triggered"]=[', '.join(map(str, l)) for l in df_grouped[f"ADMIN{admin_level}"]]
    df_grouped["Trigger description"]=d
    df_grouped_clean=df_grouped[["year","month","Regions triggered"]].set_index(['year', 'month'])
    display(df_grouped[["year","month","Regions triggered"]].set_index(['year', 'month']))


# ### FewsNet admin1, plotting characteristics of the trigger
# Chosen trigger: EITHER: At least 20% population of one or more ADMIN1 regions projected at IPC4+ in 3 months   
# OR:    
# At least 30% of ADMIN1 population projected at IPC3+ AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ in 3 months compared to current state
# 
# i.e. "an1"

df_fadm["threshold_reached"]= np.where((df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1)),1,0)
gdf=gpd.read_file(adm1_bound_path).rename(columns={parameters["shp_adm1c"]:config.ADMIN1_COL})
df_fadm=gdf[["ADMIN1","geometry"]].merge(df_fadm,how="right")


fig_boundbin=plot_boundaries_binary(df_fadm,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=4,only_show_reached=False,title_str="Regions triggered")


def plot_geo_var(df, sub_col, var_col, title=None, predef_bins=None, colp_num=2, cmap='YlOrRd'):
    """
    Plot the values of "col" for the dates present in df_trig
    If giving predef_bins then the data will be colored according to the bins, else a different colour will be assigned to each unique value in the data for each date
    df: DataFrame containing all the data of all regions
    df_trig: DataFrame containing the dates for which plots should be shown (generally those dates that the trigger is met)
    col: string with column to plot
    shape_path: relative path to the admin1 shapefile
    title: string with title of whole figure (so not the subplots)
    predef_bins: list with bin values
    """

    num_plots = len(df[sub_col].unique())
    rows = num_plots // colp_num
    rows += num_plots % colp_num
    position = range(1, num_plots + 1)

    if predef_bins is not None:
        scheme = None
        norm2 = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
        legend_kwds=None
    else:
        scheme = "natural_breaks"
        norm2 = None
        legend_kwds = {'bbox_to_anchor': (1.6, 1)}

    figsize = (16, 10)
    fig = plt.figure(1, figsize=(16, 6 * rows))

    for i, c in enumerate(df[sub_col].unique()):
        ax = fig.add_subplot(rows, colp_num, position[i])

        df_sub = df[df[sub_col] == c]
        #if no predef bins, set unique color for each unique value
        if predef_bins is None:
            colors = len(df_sub[var_col].dropna().unique())
        #else colors will be determined by norm and cmap
        else:
            colors = None

        if df_sub[var_col].isnull().values.all():
            print(f"No not-NaN values for {c}")
        elif df_sub[var_col].isnull().values.any():
            df_sub.plot(var_col, ax=ax, cmap=cmap, figsize=figsize, k=colors, norm=norm2, legend=True, scheme=scheme,
                        missing_kwds={"color": "lightgrey", "edgecolor": "red",
                                      "hatch": "///",
                                      "label": "Missing values"})
        else:
            df_sub.plot(var_col, ax=ax, cmap=cmap, figsize=figsize, k=colors, norm=norm2, legend=True, scheme=scheme)
        df.boundary.plot(linewidth=0.2, ax=ax)

        ax.axis("off")

        #prettify legend if using individual color for each value
        if predef_bins is None and not df_sub[var_col].isnull().values.all():
            leg = ax.get_legend()

            for lbl in leg.get_texts():
                label_text = lbl.get_text()
                upper = label_text.split(",")[-1].rstrip(']')

                try:
                    new_text = f'{float(upper):,.2f}'
                except:
                    new_text = upper
                lbl.set_text(new_text)

        plt.title(pd.DatetimeIndex([c])[0].to_period('M'))
    if title:
        fig.suptitle(title, fontsize=14, y=0.92)
#     fig.tight_layout()

    return fig


#all dates that trigger is met in at least one region (but include all regions on that date)
df_fadm_trig=df_fadm[df_fadm.date.isin(dict_fan["an1"]["df"].date.unique())]


#end value is not included, so set one higher than max value of last bin
bins=np.arange(0,101,10)


fig_trig_pml13p=plot_geo_var(df_fadm_trig, 
                "date",
               "perc_ML1_3p",
               title="Percentage of population projected in IPC3+ in ML1 for the dates the trigger is met",
               predef_bins=bins)


fig_trig_pml13p=plot_geo_var(df_fadm_trig, 
                "date",
               "perc_ML1_4p",
               title="Percentage of population projected in IPC3+ in ML1 for the dates the trigger is met",
                predef_bins=bins
               )


# #### Trigger analysis Global IPC data
# One of the goals was to compare the two sources of IPC data. Below are the results on the Global IPC data with the final chosen trigger

globalipc_dir=os.path.join(country_folder,config.DATA_DIR, config.GLOBALIPC_PROCESSED_DIR)
globalipc_path=os.path.join(globalipc_dir,f"{country}_globalipc_admin{admin_level}{suffix}.csv")


df_gadm=pd.read_csv(globalipc_path)


df_gadm["date"]=pd.to_datetime(df_fadmt["date"])
df_gadm["year"]=df_gadm["date"].dt.year
df_gadm["month"]=df_gadm["date"].dt.month


glob_adm1c="ADMIN1"


df_gadm.head(n=3)


df_gadm=df_gadm.replace(0,np.nan)


#get yes/no for different thresholds, i.e. column value for row will be 1 if threshold is met and 0 if it isnt
df_gadm["trigger_ML1_4_20"]=df_gadm.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_gadm["trigger_ML1_3_30"]=df_gadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_gadm["trigger_ML1_3_5i"]=df_gadm.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
df_gadm["trigger_ML2_4_20"]=df_gadm.apply(lambda x: define_trigger_percentage(x,"ML2",4,20),axis=1)
df_gadm["trigger_ML2_3_30"]=df_gadm.apply(lambda x: define_trigger_percentage(x,"ML2",3,30),axis=1)
df_gadm["trigger_ML2_3_5i"]=df_gadm.apply(lambda x: define_trigger_increase(x,"ML2",3,5),axis=1)


#initialize dict with all the analyses
dict_gan={}


#currently (Oct 2020) selected trigger
df_gan1=df_gadm.loc[(df_gadm["trigger_ML1_4_20"]==1) | ((df_gadm["trigger_ML1_3_30"]==1) & (df_gadm["trigger_ML1_3_5i"]==1))]
display(df_gan1.groupby(['year', 'month'], as_index=False)[glob_adm1c,'perc_ML1_4','perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
dict_gan["an1"]={"df":df_gan1,"trig_cols":["ML1_3p","CS_3p","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1 OR (At least 30% of ADMIN1 population projected at IPC3+  AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ compared to current state)"}


#Analysis 2: At least 20% of ADMIN1 population at IPC4+ in ML1
df_gan2 = df_gadm.loc[(df_gadm["trigger_ML1_4_20"]==1)]
display(df_gan2.groupby(['year', 'month'], as_index=False)[glob_adm1c,'perc_ML1_4'].agg(lambda x: list(x)))
dict_gan["an2"]={"df":df_gan2,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1"}


#Analysis 3: At least 30% of ADMIN1 population projected to be at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months
df_gan3 = df_gadm.loc[(df_gadm["trigger_ML1_3_30"]==1) & (df_gadm["trigger_ML1_3_5i"]==1)]
display(df_gan3.groupby(['year', 'month'], as_index=False)[glob_adm1c,'perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
dict_gan["an3"]={"df":df_gan3,"trig_cols":["ML1_3","CS_3","CS_4","ML1_4"],"desc":"At least 30% of ADMIN1 population in ML1 at IPC3+  AND5% increase in ADMIN1 pop. projected in IPC3+ in 3 months"}


#Analysis 11: At least 20% of population projected in ML2 in IPC4+ OR (30% in ML2 in IPC3+ AND 5% increase in IPC3+ in ML2 compared to CS)
df_gan11=df_gadm.loc[(df_gadm["trigger_ML2_4_20"]==1) | ((df_gadm["trigger_ML2_3_30"]==1)&(df_gadm["trigger_ML2_3_5i"]==1))]
display(df_gan11.groupby(['year', 'month'], as_index=False)[glob_adm1c,'perc_ML2_4','perc_CS_3p','perc_ML2_3p','perc_inc_ML2_3p','pop_CS','pop_ML2'].agg(lambda x: list(x)))
dict_gan["an11"]={"df":df_gan11,"trig_cols":["ML2_3","ML2_4"],"desc":"20% in ML2 in IPC4 OR (30% in ML2 in IPC3+ AND 5% increase in IPC3+ in ML2 compared to CS)"}


for k in dict_gan.keys():
    d=dict_gan[k]["desc"]
    num_k=k.replace("an","")
    print(f"Analysis {num_k}: GlobalIPC, {d}")
    df=dict_gan[k]["df"]
    df_grouped=df.groupby(['year', 'month'], as_index=False)[glob_adm1c].agg(lambda x: list(x))
    if df_grouped.empty:
        display(df_grouped)
    else:
        df_grouped[glob_adm1c]=[', '.join(map(str, l)) for l in df_grouped[glob_adm1c]]
        df_grouped["Trigger description"]=d
        df_grouped=df_grouped.rename(columns={glob_adm1c:"Regions triggered"})
        df_grouped_clean=df_grouped[["year","month","Regions triggered"]].set_index(['year', 'month'])
        display(df_grouped[["year","month","Regions triggered"]].set_index(['year', 'month']))
        b=df_grouped[["year","month","Regions triggered","Trigger description"]].set_index(['Trigger description','year', 'month'])


# ### FewsNet analysis Admin2
# While the previous analysis focused on admin1 level, it is also possible to design a trigger on admin2 level. 
# A small exploration was done on the FewsNet data. Finally, it was decided to focus on admin1 but this is an area that could be explored further in the future. 
# For now, we explored the trigger of having 1 or more, or 2 or more, admin2 regions/admin1 region projected to be in IPC4 in ML1

admin_level=2
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)
df_fadmt=pd.read_csv(os.path.join(fewsnet_dir,fewsnet_filename))
df_fadmt["date"]=pd.to_datetime(df_fadmt["date"])
df_fadmt["year"]=df_fadmt["date"].dt.year
df_fadmt["month"]=df_fadmt["date"].dt.month


adm2_bound_path= os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


df_fadmt


#ML1 values of all adm2 regions in all data
#not ever been or forecasted to be IPC 5
df_fadmt.value_counts("ML1_5")


#select admin 2 regions with projected IPC level 4 in ML1
df_fadmtp=df_fadmt[df_fadmt.perc_ML1_4p>=20]


df_g=df_fadmtp.groupby(["year","month","ADMIN1"], as_index=False).agg(lambda x: list(x))
df_g=df_g[["year","month","ADMIN1","ADMIN2"]]
df_g["# ADM2 regions ML1 IPC4"]=df_g["ADMIN2"].str.len()


print("Analysis a): 1 or more ADMIN2 regions have IPC4 >= 20% in ML1")
df_g.drop("ADMIN2",axis=1).set_index(['year', 'month',"ADMIN1"])


print("Analysis b): 2 or more ADMIN2 regions have IPC4 >= 20% in ML1")
df_g[df_g["# ADM2 regions ML1 IPC4"]>1].drop("ADMIN2",axis=1).set_index(['year', 'month',"ADMIN1"])


df_fadmt["threshold_reached"]= np.where(df_fadmt.perc_ML1_4p>=20,1,0)
gdft=gpd.read_file(adm2_bound_path).rename(columns={parameters["shp_adm2c"]:config.ADMIN2_COL})
df_fadmt=gdft[["ADMIN2","geometry"]].merge(df_fadmt,how="right")


fig_boundbint=plot_boundaries_binary(df_fadmt,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN2",colp_num=4,only_show_reached=False,title_str="ADMIN2's % IPC4+ in ML1 >=20",print_reg=False)


df_fadmt["date_str"]=df_fadmt.date.dt.strftime("%Y-%m")


fig_boundbint=plot_boundaries_binary(df_fadmt,"threshold_reached",subplot_col="date",subplot_str_col="date_str",region_col="ADMIN2",colp_num=4,only_show_reached=False,title_str="ADMIN2's % IPC4+ in ML1 >=20",print_reg=False)




