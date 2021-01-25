#Compute the trigger information needed for the dashboard

# IPC trigger design as of 08-10-2020:
# EITHER: At least 20% population of one or more ADMIN1 regions projected at IPC4+ in 3 months
# OR:
# At least 30% of ADMIN1 population projected at IPC3+ AND increase by 5 percentage points in ADMIN1 pop.  projected in IPC3+ in 3 months compared to current state
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.ipc_definemetrics import define_trigger_percentage, define_trigger_increase

admin_level=1
country="ethiopia"
#suffix of filenames
suffix=""
config=Config()
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
fewsnet_dir = os.path.join(country_folder, config.DATA_DIR, config.FEWSWORLDPOP_PROCESSED_DIR)
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)
globalipc_dir=os.path.join(country_folder,config.DATA_DIR, config.GLOBALIPC_PROCESSED_DIR)
globalipc_path=os.path.join(globalipc_dir,f"{country}_globalipc_admin{admin_level}{suffix}.csv")

adm_bound_path= os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters[f"path_admin{admin_level}_shp"])

df_fn=pd.read_csv(os.path.join(fewsnet_dir,fewsnet_filename))
df_fn["source"]="FewsNet"
df_gipc=pd.read_csv(globalipc_path)
df_gipc["source"]="GlobalIPC"

df=pd.concat([df_fn,df_gipc])
df=df.replace(0,np.nan)
df["date"]=pd.to_datetime(df["date"])
df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["trigger_ML1_4_20"]=df.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df["trigger_ML1_3_30"]=df.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df["trigger_ML1_3_5i"]=df.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
df[f"threshold_reached"]=np.where((df[f"trigger_ML1_4_20"]==1) | ((df[f"trigger_ML1_3_30"]==1) & (df[f"trigger_ML1_3_5i"]==1)),1,0)

# gdf=gpd.read_file(adm_bound_path).rename(columns={parameters[f"shp_adm{admin_level}c"]:f"ADMIN{admin_level}"})
# df=gdf[[f"ADMIN{admin_level}","geometry"]].merge(df,how="right")
df=df[(df["source"]=="FewsNet") & (df["date"]==df["date"].max())]
df.to_csv(os.path.join(config.DIR_PATH,"dashboard","data","foodinsecurity","ethiopia_foodinsec_trigger.csv"))

# fig_boundbin=plot_spatial_binary_column(df,"trigger_ML1",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=4,only_show_reached=False,title_str="Regions triggered")
# plt.show()
