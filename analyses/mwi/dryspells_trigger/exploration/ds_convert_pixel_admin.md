### Compute dry spells based on pixels
This notebook computes the occurrences of dry spells per admin. As input a list of dry spells at pixel level is taken, where a dry spell is defined as 14 days with <=2mm cumulative rainfall. This notebook tests different threshold for the percentage of pixels, i.e. cells, that should be in a dry spell in order to classify the admin as experiencing a dry spell. These results are thereafter saved to use in a R script to create a heatmap of dry spells across time

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import geopandas as gpd

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

#### Set config values

```python
country="mwi"
config=Config()
parameters = config.parameters(country)
iso3=parameters["iso3_code"]

country_data_raw_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.RAW_DIR,iso3)
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,config.PROCESSED_DIR,iso3)
country_data_exploration_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR,"exploration",iso3)
dry_spells_processed_dir=os.path.join(country_data_processed_dir,"dry_spells")

adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
```

#### Define admin level

```python
adm_level=2
```

```python
dry_spells_pixel_path=os.path.join(dry_spells_processed_dir,f"ds_counts_per_pixel_adm{adm_level}.csv")
adm_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters[f"path_admin{adm_level}_shp"])
adm_name_col=f"ADM{adm_level}_EN"
adm_pcode_col=f"ADM{adm_level}_PCODE"
```

### Compute dry spells for different thresholds of pixels at admin
Have a dataset of the number of pixels, i.e. cells, experiencing a dry spell. We can set a threshold for the number of pixels to experience a dry spell, in order to classify the admin as experiencing a dry spell. Here different thresholds are set, and thereafter saved to a file to be used for a heatmap visualization in R. 

```python
df_ds_pixel=pd.read_csv(dry_spells_pixel_path,parse_dates=["date"])
```

```python
df_adm_bound=gpd.read_file(adm_bound_path)
```

```python
df_ds_pixel=df_ds_pixel.merge(df_adm_bound[[adm_name_col,adm_pcode_col]],how="left",on=adm_name_col)
df_ds_pixel.rename(columns={adm_pcode_col:"pcode"},inplace=True)
```

```python
def label_ds(row,threshold):
    if row["perc_ds_cells"]>=threshold:
        return 1
    else:
        return 0
```

```python
threshold_list=[10,30,50,70,90]
```

```python
for threshold in threshold_list:
    df_ds_pixel[f"ds_t{threshold}"]=np.where(df_ds_pixel.perc_ds_cells>=threshold,1,0)
    df_ds_pixel["dryspell_match"]=df_ds_pixel.apply(lambda row:label_ds(row,threshold),axis=1)
    #add dates that are not present in df_ds_both, i.e. outside rainy season
    df_ds_pixel_filled=df_ds_pixel.sort_values('date').set_index(['date']).groupby(adm_name_col).apply(lambda x: x.reindex(pd.date_range(pd.to_datetime('01-01-2000'), pd.to_datetime('31-12-2020'), name='date'),fill_value=0).drop(adm_name_col,axis=1).reset_index()).reset_index().drop("level_1",axis=1)
#     df_ds_pixel_filled.to_csv(os.path.join(country_data_exploration_dir,"dryspells",f"dryspells_pixel_adm{adm_level}_th{threshold}_viz.csv"))
```

```python
df_ds_pixel_filled.head()
```

```python
# #compute the number of dry spells
# df_ds_th=df_ds_pixel[df_ds_pixel[f"ds_t{threshold}"]==1]
# df_ds_th["ds_id"]=df_ds_th.sort_values([adm_name_col,"date"]).groupby(adm_name_col).date.diff().dt.days.ne(1).cumsum()
```

### Compare with mean method at admin2
Compare the dry spell events based on the definition of X% of the pixels experiencing a dry spell, to the method of the mean of all pixels having a dry spell
This comparison is saved to a file to be used to create a heatmap in R.    

Only works at admin2 since no list of mean-based dry spells at admin1 is available

```python
threshold=50
df_ds_pixel[f"ds_t{threshold}"]=np.where(df_ds_pixel.perc_ds_cells>=threshold,1,0)
```

```python
#load mean dry spells, since that is what we want to compare to
dry_spells_list_path=os.path.join(dry_spells_processed_dir,f"dry_spells_during_rainy_season_list_2000_2020_mean_back.csv")
df_ds_chirps=pd.read_csv(dry_spells_list_path)
df_ds_chirps["dry_spell_first_date"]=pd.to_datetime(df_ds_chirps["dry_spell_first_date"])
df_ds_chirps["dry_spell_last_date"]=pd.to_datetime(df_ds_chirps["dry_spell_last_date"])
df_ds_chirps["year"]=df_ds_chirps.dry_spell_first_date.dt.year

#create df with all dates that were part of a dry spell per adm2
#important to reset the index, since that is what is being joined on
df_ds_chirps_res=df_ds_chirps.reset_index(drop=True)
#create datetimeindex per row
a = [pd.date_range(*r, freq='D') for r in df_ds_chirps_res[['dry_spell_first_date', 'dry_spell_last_date']].values]
#join the daterange with the adm2, which create a column per date, then stack to have each adm2-date combination
df_ds_chirps_daterange=df_ds_chirps_res[["pcode"]].join(pd.DataFrame(a)).set_index(["pcode"]).stack().droplevel(-1).reset_index()
df_ds_chirps_daterange.rename(columns={0:"date"},inplace=True)
#all dates in this dataframe had an observed dry spell, so add that information
df_ds_chirps_daterange["dry_spell_chirps"]=1

#For Nsanje (MW311), there is an overlapping dry spell that is identified as two separate dry spells --> causes duplicate dates
df_ds_chirps_daterange=df_ds_chirps_daterange.drop_duplicates(["pcode","date"])
```

```python
def label_ds(row):
    if row["dry_spell_mean"]==1 and row["dry_spell_pixel"]==1:
        return 3
    elif row["dry_spell_mean"]==1:
        return 2
    elif row["dry_spell_pixel"]==1:
        return 1
    else:
        return 0
```

```python
#merge pixel and mean data
df_ds_both=df_ds_pixel.merge(df_ds_chirps_daterange,how="outer",on=["date","pcode"])
df_ds_both["dry_spell_mean"]=df_ds_both.dry_spell_chirps.replace(np.nan,0)
df_ds_both["dry_spell_pixel"]=df_ds_both[f"ds_t{threshold}"].replace(np.nan,0)
df_ds_both["season_approx"]=np.where(df_ds_both.date.dt.month>=10,df_ds_both.date.dt.year,df_ds_both.date.dt.year-1)

#encode dry spells and whether both sources observed a dry spell, or only one of the two
df_ds_both["dryspell_match"]=df_ds_both.apply(lambda row:label_ds(row),axis=1)
#add dates that are not present in df_ds_both, i.e. outside rainy season
df_ds_both_filled=df_ds_both.sort_values('date').set_index(['date']).groupby('pcode').apply(lambda x: x.reindex(pd.date_range(pd.to_datetime('01-01-2000'), pd.to_datetime('31-12-2020'), name='date'),fill_value=0).drop('pcode',axis=1).reset_index())
```

```python
#cause for now we only wanna show till end of 2020 cause no processed chirps data after that
df_ds_both_filled=df_ds_both_filled[df_ds_both_filled.date.dt.year<=2020]
```

```python
# df_ds_both_filled.to_csv(os.path.join(country_data_exploration_dir,"dryspells",f"dryspells_mean_pixel_th{threshold}_viz.csv"))
```
