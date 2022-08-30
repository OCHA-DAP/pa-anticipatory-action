### Different methods of aggregation from livelihood to admin2
We implemented two methodologies for aggregating FewsNet's livelihood data to admin levels. 
One methodology was to assign one phase per admin2, where this phase equals the one that covers the maximum area in km2. Thereafter this phase is combined with the subnational adm2 population data from HDX's Common Operational Dataset (COD)   
The other methodology is to use Worldpop's raster data, which enables us to assign a percentage of population per IPC phase to each admin2 (instead of having the whole population in one phase per admin2).   
This notebook explores the differences between the two methodologies. Based on this analysis it was chosen to use the Worldpop data as base source, since it captures changes in IPC phases more precisely. Nevertheless, a disadvantage can be that the distribution of population per admin2 by WorldPop can be different than COD, and that the total national population often doesn't correspond. Therefore, we will implement an option to rescale the numbers to COD numbers, whether to use that option will be discussed with the country team. 

```python
%load_ext autoreload
%autoreload 2
```

```python
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
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import ConfigFI
from src.indicators.food_insecurity.ipc_definemetrics import define_trigger_percentage, define_trigger_increase, define_trigger_increase_rel
from src.indicators.food_insecurity.utils import compute_percentage_columns
from src.utils_general.plotting import plot_spatial_binary_column
```

```python
iso3="eth"
admin_level=1
#suffix of filenames
suffix=""
config=ConfigFI()
parameters = config.parameters(iso3)

public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,iso3)
country_data_processed_dir = os.path.join(public_data_dir,config.PROCESSED_DIR,iso3)
adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
fnwp_dir = os.path.join(country_data_processed_dir, config.FEWSWORLDPOP_PROCESSED_DIR)
fnocha_dir = os.path.join(country_data_processed_dir, config.FEWSADMPOP_PROCESSED_DIR)
fnwp_path = os.path.join(fnwp_dir,config.FEWSWORLDPOP_PROCESSED_FILENAME.format(iso3=iso3,admin_level=admin_level,suffix=suffix))
fnocha_path = os.path.join(fnocha_dir,f"{iso3}_fewsnet_admin{admin_level}.csv")
pop_ocha_adm2_path=os.path.join(country_data_raw_dir,"cod_ps","Population_OCHA_2020","eth_admpop_adm2_20201028.csv")
```

```python
gdf=gpd.read_file(adm1_bound_path).rename(columns={parameters["shp_adm1c"]:config.ADMIN1_COL})
```

### WorldPop

```python
df_fadm=pd.read_csv(fnwp_path,parse_dates=["date"])
df_fadm["year"]=df_fadm["date"].dt.year
df_fadm["month"]=df_fadm["date"].dt.month
```

```python
df_fadm.tail()
```

```python
#display historical moments trigger was met with worldpop data
#compute if threshold is met for phase X, level Y, with Z% of population per admin1
df_fadm["trigger_ML1_4_20"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_fadm["trigger_ML1_3_30"]=df_fadm.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_fadm["trigger_ML1_3_5i"]=df_fadm.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
#currently (Oct 2020) selected trigger
df_fadm_trig=df_fadm.loc[(df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1))]
display(df_fadm_trig.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4','perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
```

```python
df_focha=pd.read_csv(fnocha_path)
df_focha=df_focha.rename(columns={parameters["shp_adm1c"]:"ADMIN1"})
df_focha["date"]=pd.to_datetime(df_focha["date"])
df_focha["year"]=df_focha["date"].dt.year
df_focha["month"]=df_focha["date"].dt.month
df_focha=compute_percentage_columns(df_focha,config)
```

```python
#display historical moments trigger was met with COD PS data
df_focha["trigger_ML1_4_20"]=df_focha.apply(lambda x: define_trigger_percentage(x,"ML1",4,20),axis=1)
df_focha["trigger_ML1_3_30"]=df_focha.apply(lambda x: define_trigger_percentage(x,"ML1",3,30),axis=1)
df_focha["trigger_ML1_3_5i"]=df_focha.apply(lambda x: define_trigger_increase(x,"ML1",3,5),axis=1)
#currently (Oct 2020) selected trigger
df_focha_trig=df_focha.loc[(df_focha["trigger_ML1_4_20"]==1) | ((df_focha["trigger_ML1_3_30"]==1) & (df_focha["trigger_ML1_3_5i"]==1))]
display(df_focha_trig.groupby(['year', 'month'], as_index=False)[f"ADMIN{admin_level}",'perc_ML1_4','perc_CS_3p','perc_ML1_3p'].agg(lambda x: list(x)))
```

```python
df_fadm["threshold_reached"]= np.where((df_fadm["trigger_ML1_4_20"]==1) | ((df_fadm["trigger_ML1_3_30"]==1) & (df_fadm["trigger_ML1_3_5i"]==1)),1,0)
df_fadm=gdf[["ADMIN1","geometry"]].merge(df_fadm,how="right")

df_focha["threshold_reached"]= np.where((df_focha["trigger_ML1_4_20"]==1) | ((df_focha["trigger_ML1_3_30"]==1) & (df_focha["trigger_ML1_3_5i"]==1)),1,0)
df_focha=gdf[["ADMIN1","geometry"]].merge(df_focha,how="right")
```

```python
print(f"Number of times trigger met with OCHA COD PS data: {len(df_focha_trig)}")
print(f"Number of times trigger met with WorldPop data: {len(df_fadm_trig)}")
```

```python
#plot regions and years trigger was met spatially with OCHA COD PS
fig_boundbin=plot_spatial_binary_column(df_focha,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=5,only_show_reached=False)
```

```python
#plot regions and years trigger was met spatially with WorldPop
fig_boundbin=plot_spatial_binary_column(df_fadm,"threshold_reached",subplot_col="year",subplot_str_col="year",region_col="ADMIN1",colp_num=5,only_show_reached=False)
```

#### Differences FN OCHA adm pop and Worldpop pop
2010-01 Oromia is triggered with ocha but not with worldpop. This is because the 30% 3+ ML1 is not reached. For worldpop the percentage is 28 whereas for ocha it is 34. Hard to say from fewsnet images which makes more sense, is a border case.   
2016-02 Oromia is triggered with ocha and not with worldpop. This is because the 5% increase threshold is not reached. For worldpop the percentage is 1.6 and for ocha 9.6. Looking at the images, a relatively small areas changes from ipc3+ to ipc2. However, this area does change one large adm2 from ipc2 to ipc3+. I would argue the worldpop approach is more correct   
2019-10 Afar is triggered with worldpop and not with ocha. This is because the 5% increase threshold is not reached. Looking at the images, there is an increase in area that is projected to be IPC3+ but this doesn't change the major ipc phase coverage of an admin2. Thus would argue the worldpop approach is more correct


#### Digging deeper into differences

```python
df_focha[(df_focha.year==2016)&(df_focha.threshold_reached==1)]
```

```python
df_fadm[(df_fadm.year==2016)&(df_fadm.threshold_reached==1)]
```

```python
df_focha[(df_focha.date=="2016-02-01")][["ADMIN1","trigger_ML1_4_20","trigger_ML1_3_30","trigger_ML1_3_5i","perc_CS_3p","perc_ML1_3p","perc_inc_ML1_3p"]]
```

```python
df_fadm[(df_fadm.date=="2016-02-01")][["ADMIN1","trigger_ML1_4_20","trigger_ML1_3_30","trigger_ML1_3_5i","perc_CS_3p","perc_ML1_3p","perc_inc_ML1_3p"]]
```

## Differences in population totals and distrubtion of COD and WorldPop and Worldbank

```python
#total numbers. WB is currently used in the df_focha
df_adm2_popocha=pd.read_csv(pop_ocha_adm2_path)
pop_COD=df_adm2_popocha.sum()["Total"]
pop_WP=df_fadm[df_fadm.date=="2020-10-01"].sum()["pop_CS"]
pop_WB=df_focha[df_focha.date=="2020-10-01"].sum()["pop_CS"]
print(f"population assigned of WorldPop: {pop_WP:.0f}")
print(f"population assigned of WorldBank: {pop_WB:.0f} ({(pop_WP-pop_WB):.0f} less than WP)")
print(f"population of COD: {pop_COD:.0f} ({(pop_WP-pop_COD):.0f} less than WP)")
```

#### Differences in distribution COD and WorldPop

```python
#compute percentage of pop adm1 per adm2 COD
df_adm2_popocha=df_adm2_popocha[["ADM1_EN","ADM2_EN","Total"]]
df_adm1_popocha=df_adm2_popocha.groupby("ADM1_EN",as_index=False).sum()
#add population of adm1 to each adm2
df_adm2_popocha=df_adm2_popocha.merge(df_adm1_popocha.rename(columns={"Total":"ADM1_Total"}),on="ADM1_EN",how="left")
#compute percentage of pop adm1 per adm2
df_adm2_popocha["perc_ADM1"]=df_adm2_popocha["Total"]/df_adm2_popocha["ADM1_Total"]*100
```

```python
#compute percentage of pop adm1 per adm2 WorldPop
adm2_popwp_path=os.path.join(fnwp_dir,config.FEWSWORLDPOP_PROCESSED_FILENAME.format(suffix=suffix,iso3=iso3,admin_level=2))
df_adm2_popwp=pd.read_csv(adm2_popwp_path)
df_adm2_popwp.date=pd.to_datetime(df_adm2_popwp.date)
# df_adm1_popwp=df_fadm.groupby(["date","ADMIN1"],as_index=False).sum()
df_adm2_popwp=df_adm2_popwp.merge(df_fadm[["date","ADMIN1","pop_CS"]],suffixes=("","_ADM1"),on=["date","ADMIN1"],how="left")
df_adm2_popwp["perc_ADM1"]=df_adm2_popwp["pop_CS"]/df_adm2_popwp["pop_CS_ADM1"]*100
```

```python
#compute difference two population sources
df_adm2_merge=df_adm2_popocha[["ADM2_EN","perc_ADM1"]].merge(df_adm2_popwp[df_adm2_popwp.date=="2020-10-01"][["ADMIN1","ADMIN2","perc_ADM1"]],left_on="ADM2_EN",right_on="ADMIN2",suffixes=("_COD","_WorldPop"))
df_adm2_merge["perc_diff"]=df_adm2_merge["perc_ADM1_COD"]-df_adm2_merge["perc_ADM1_WorldPop"]
```

```python
df_adm2_merge.sort_values(by="perc_diff")
```

```python
#show adm2 with largest diff
df_adm2_merge[df_adm2_merge["ADMIN1"]=="Dire Dawa"][["ADMIN1","ADMIN2","perc_ADM1_COD","perc_ADM1_WorldPop","perc_diff"]]
```

```python
#show perc diff for admin2s in Afar
df_adm2_merge[df_adm2_merge["ADMIN1"]=="Afar"][["ADMIN1","ADMIN2","perc_ADM1_COD","perc_ADM1_WorldPop","perc_diff"]]
```