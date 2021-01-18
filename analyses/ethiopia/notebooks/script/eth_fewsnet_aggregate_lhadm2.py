#!/usr/bin/env python
# coding: utf-8

# ### Different methods of aggregation from livelihood to admin2

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import seaborn as sns
import numpy as np
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
from indicators.food_insecurity.utils import compute_percentage_columns
from utils_general.plotting import plot_boundaries_binary


country="ethiopia"
#suffix of filenames
suffix=""
config=Config()
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
adm1_bound_path= os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
fnwp_dir = os.path.join(country_folder, config.DATA_DIR, config.FEWSWORLDPOP_PROCESSED_DIR)
fnwp_path = os.path.join(fnwp_dir,config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix))
fnocha_dir = os.path.join(country_folder, config.DATA_DIR, config.FEWSADMPOP_PROCESSED_DIR)
fnocha_path = os.path.join(fnocha_dir,f"{country}_fewsnet_admin{admin_level}.csv")

admin_level=1


gdf=gpd.read_file(adm1_bound_path).rename(columns={parameters["shp_adm1c"]:config.ADMIN1_COL})


# ### WorldPop

df_fadm=pd.read_csv(fnwp_path)
df_fadm["date"]=pd.to_datetime(df_fadm["date"])
df_fadm["year"]=df_fadm["date"].dt.year
df_fadm["month"]=df_fadm["date"].dt.month


df_fadm.tail()


df_fadm["trigger_ML1_4_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_fadm["trigger_ML1_3_30"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_fadm["trigger_ML1_3_5i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
#currently (Oct 2020) selected trigger
df_fadm_trig=df_fadm.loc[(df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1))]
display(df_fadm_trig.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4','perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
# df_fadm_trig["an1"]={"df":df_an1,"trig_cols":["ML1_3p","CS_3p","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1 OR (At least 30% of ADMIN1 population projected at IPC3+  AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ compared to current state)"}


df_focha=pd.read_csv(fnocha_path)
df_focha=df_focha.rename(columns={parameters["shp_adm1c"]:"ADMIN1"})
df_focha["date"]=pd.to_datetime(df_focha["date"])
df_focha["year"]=df_focha["date"].dt.year
df_focha["month"]=df_focha["date"].dt.month
df_focha=compute_percentage_columns(df_focha,config)


df_focha["trigger_ML1_4_20"]=df_focha.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_focha["trigger_ML1_3_30"]=df_focha.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_focha["trigger_ML1_3_5i"]=df_focha.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
#currently (Oct 2020) selected trigger
df_focha_trig=df_focha.loc[(df_focha["trigger_ML1_4_20"]==1) | ((df_focha["trigger_ML1_3_30"]==1) & (df_focha["trigger_ML1_3_5i"]==1))]
display(df_focha_trig.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4','perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
# df_focha_trig["an1"]={"df":df_an1,"trig_cols":["ML1_3p","CS_3p","ML1_4"],"desc":"At least 20% of ADMIN1 population in IPC4+ at ML1 OR (At least 30% of ADMIN1 population projected at IPC3+  AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ compared to current state)"}


df_fadm["threshold_reached"]= np.where((df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1)),1,0)
df_fadm=gdf[["ADMIN1","geometry"]].merge(df_fadm,how="right")

df_focha["threshold_reached"]= np.where((df_focha["trigger_ML1_4_20"]==1) | ((df_focha["trigger_ML1_3_30"]==1) & (df_focha["trigger_ML1_3_5i"]==1)),1,0)
df_focha=gdf[["ADMIN1","geometry"]].merge(df_focha,how="right")


len(df_focha_trig)


len(df_fadm_trig)


fig_boundbin=plot_boundaries_binary(df_focha,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=4,only_show_reached=False)


fig_boundbin=plot_boundaries_binary(df_fadm,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=4,only_show_reached=False)


# #### Differences FN OCHA adm pop and Worldpop pop
# 2010-01 Oromia is triggered with ocha but not with worldpop. This is because the 30% 3+ ML1 is not reached. For worldpop the percentage is 28 whereas for ocha it is 34. Hard to say from images which makes more sense, is a border case.   
# 2016-02 Oromia is triggered with ocha and not with worldpop. This is because the 5% increase threshold is not reached. For worldpop the percentage is 1.6 and for ocha 9.6. Looking at the images, a relatively small areas changes from ipc3+ to ipc2. However, this area does change one large adm2 from ipc2 to ipc3+. I would argue the worldpop approach is more correct   
# 2019-10 Afar is triggered with worldpop and not with ocha. This is because the 5% increase threshold is not reached. Looking at the images, there is an increase in area that is projected to be IPC3+ but this doesn't change the major ipc phase coverage of an admin2. Thus would argue the worldpop approach is more correct

# #### Digging deeper into differences

df_focha[(df_focha.year==2016)&(df_focha.threshold_reached==1)]


df_fadm[(df_fadm.year==2016)&(df_fadm.threshold_reached==1)]


df_focha[(df_focha.date=="2016-02-01")][["ADMIN1","trigger_ML1_4_20","trigger_ML1_3_30","trigger_ML1_3_5i","perc_CS_3p","perc_ML1_3p","perc_inc_ML1_3p"]]


df_fadm[(df_fadm.date=="2016-02-01")][["ADMIN1","trigger_ML1_4_20","trigger_ML1_3_30","trigger_ML1_3_5i","perc_CS_3p","perc_ML1_3p","perc_inc_ML1_3p"]]

