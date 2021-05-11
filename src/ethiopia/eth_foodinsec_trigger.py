#Compute the trigger information needed for the dashboard. Ugly now for first mock-up, should be improved later on

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
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import Config
from src.indicators.food_insecurity.ipc_definemetrics import define_trigger_percentage, define_trigger_increase
from src.indicators.food_insecurity.utils import compute_percentage_columns

admin_level=1
country="ethiopia"
fn_process="admpop" #worldpop
#suffix of filenames
suffix=""
config=Config()
parameters = config.parameters(country)
country_data_raw_dir = os.path.join(config.DATA_DIR, 'public', 'raw', parameters['iso3_code'])
country_data_processed_dir = os.path.join(config.DATA_DIR, 'public', 'processed', parameters['iso3_code'])

if fn_process=="worldpop":
    fewsnet_dir = os.path.join(country_data_processed_dir, config.FEWSWORLDPOP_PROCESSED_DIR)
    fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)
elif fn_process=="admpop":
    fewsnet_dir = os.path.join(country_data_processed_dir, config.FEWSADMPOP_PROCESSED_DIR)
    fewsnet_filename = config.FEWSADMPOP_PROCESSED_FILENAME.format(country=country,admin_level=admin_level,suffix=suffix)
globalipc_dir=os.path.join(country_data_processed_dir, config.GLOBALIPC_PROCESSED_DIR)
globalipc_path=os.path.join(globalipc_dir,f"{country}_globalipc_admin{admin_level}{suffix}.csv")

adm_bound_path = os.path.join(country_data_raw_dir, config.SHAPEFILE_DIR, parameters[f'path_admin{admin_level}_shp'])

#TODO: remove index in process_fewsnet_admpop script
df_fn=pd.read_csv(os.path.join(fewsnet_dir,fewsnet_filename),index_col=False)
df_fn=df_fn.drop("Unnamed: 0",axis=1)
print(df_fn)
#TODO: rename these in process_fewsnet_admpop script
df_fn=df_fn.rename(columns={"ADM1_EN":"ADMIN1","ADM2_EN":"ADMIN2"})
#TODO: add percentages in process_fewsnet_admpop script
df_fn=compute_percentage_columns(df_fn,config)
#TODO: figure out a way to automatically add data to the updates in the FN processing script
#the data of 2021-01 is an update and thus doesn't include CS data or projected periods
#add them here manually, where the CS data is set to that of 2020-10
CS_cols=[c for c in df_fn.columns if 'CS' in c]
for c in CS_cols:
    for a in df_fn[f"ADMIN{admin_level}"].unique():
        df_fn.loc[(df_fn.date=="2021-01-01")&(df_fn[f"ADMIN{admin_level}"]==a),c]=df_fn.loc[(df_fn.date=="2020-10-01")&(df_fn[f"ADMIN{admin_level}"]==a),c].values
df_fn[df_fn.date=="2021-01-01"]=compute_percentage_columns(df_fn[df_fn.date=="2021-01-01"],config)
df_fn.loc[df_fn.date=="2021-01-01","period_ML1"]="Jan 2021"
df_fn.loc[df_fn.date=="2021-01-01","period_ML2"]="Feb - May 2021"
df_fn["source"]="FewsNet"
df_gipc=pd.read_csv(globalipc_path)
df_gipc["source"]="GlobalIPC"

df=pd.concat([df_fn,df_gipc])
# print(df)
df["country"]="eth"
df["date"]=pd.to_datetime(df["date"])
df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["trigger_ML1_4_20"]=df.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df["trigger_ML1_3_30"]=df.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df["trigger_ML1_3_5i"]=df.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
df[f"threshold_reached_ML1"]=np.where((df[f"trigger_ML1_4_20"]==1) | ((df[f"trigger_ML1_3_30"]==1) & (df[f"trigger_ML1_3_5i"]==1)),True,False)

df["trigger_ML2_4_20"]=df.apply(lambda x: define_trigger_percentage(x,"ML2",4,20),axis=1)
df["trigger_ML2_3_30"]=df.apply(lambda x: define_trigger_percentage(x,"ML2",3,30),axis=1)
df["trigger_ML2_3_5i"]=df.apply(lambda x: define_trigger_increase(x,"ML2",3,5),axis=1)
df[f"threshold_reached_ML2"]=np.where((df[f"trigger_ML2_4_20"]==1) | ((df[f"trigger_ML2_3_30"]==1) & (df[f"trigger_ML2_3_5i"]==1)),True,False)
print(df.columns)
df.to_csv(os.path.join("dashboard","data","foodinsecurity","ethiopia_foodinsec_trigger.csv"),index=False)

# fig_boundbin=plot_spatial_binary_column(df,"trigger_ML1",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=4,only_show_reached=False,title_str="Regions triggered")
# plt.show()
