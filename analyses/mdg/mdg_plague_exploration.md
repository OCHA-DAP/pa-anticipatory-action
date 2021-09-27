<!-- #region -->
### Analysis of plague data in Madagascar
This notebook explores plague data in Madagascar.


Note that this data is not publicly available. 
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import geopandas as gpd
import plotly.express as px
from datetime import date
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
```

```python
iso3="mdg"
config=Config()
parameters = config.parameters(iso3)
```

```python
public_data_dir = os.path.join(config.DATA_DIR, config.PUBLIC_DIR)
country_data_raw_dir = os.path.join(public_data_dir,config.RAW_DIR,iso3)
adm3_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin3_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])
```

```python
plague_data_filename = "Madagascar_IPM_Plague_cases_Aggregated_2021-09-24.csv"
plague_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / config.RAW_DIR / iso3 / "institut_pasteur"
plague_path = plague_dir / plague_data_filename
```

```python
df=pd.read_csv(plague_path, delimiter = ";")
```

```python
df.columns=df.columns.str.lower()
df.rename(columns={"mdg_com_code":"ADM3_PCODE"},inplace=True)
#to make pcodes correspond with shp file
df.ADM3_PCODE=df.ADM3_PCODE.str.replace("MDG","MG")
```

```python
df=df.sort_values(["year","week"])
df
```

There is an entry of week 53 in 2021, this cannot be correct so drop it (maybe should understand where it comes from)

```python
df=df[~((df.year==2021)&(df.week==53))]
```

### Understand the different columns
We inspect the unique values and describe what they mean

```python
df.clinical_form.unique()
```

```python
df.cases_class.unique()
```

```python
df.status.unique()
```

##### clinical_form
- PP = Plague Pneumonic
- NP = Not Specified
- PB = Plague Bubonic
- PS = Plague Septicemic

##### cases_class
- SUSP = Suspected  
- PROB = Probable  
- CONF = Confirmed

##### status
- NSP = Ne Sais Pas = Unknown
- DECEDE = deceased
- VIVANT = alive

##### mdg_com_code
This column contains the Commune Pcode, which is admin3 level. However, it seems some rows contain the admin2 instead of admin3 code


### Cases over time
Very basic plot with cases over time

```python
#create a datetime from the year and week as this is easier with plotting
df["date"]=df.apply(lambda x: date.fromisocalendar(x.year,x.week,1),axis=1)
df["date"]=pd.to_datetime(df["date"])
```

```python
df_date=df.groupby("date",as_index=False).sum()
```

```python
px.line(df_date,x="date",y="cases_number")
```

### Geographical coverage
Check correspondence data and shapefile, and plot where cases were reported

```python
gdf_adm3=gpd.read_file(adm3_bound_path)
```

```python
# Check that admin level names in the plague data and shapefile match
missing_df_gdf = np.setdiff1d(
    list(df["ADM3_PCODE"].dropna()),
    list(gdf["ADM3_PCODE"].dropna()),
)
print(f"Pcodes in plague data but not in adm3 shapefile: {missing_df_gdf}")

missing_gdf_df = np.setdiff1d(
    list(gdf["ADM3_PCODE"].dropna()),
    list(df["ADM3_PCODE"].dropna()),
)
print(f"Pcodes in adm3 shapefile and not in plague data: {missing_gdf_df}")
```

It seems that all the pcodes that are not foundin the adm3 shapefile are actually adm2 pcodes. We have to understand why and what to do with them. 

Many adm3 pcodes are not in the plague data but that can just indicate there were no cases ever reported in those adm3's

```python
df_adm = df.groupby("ADM3_PCODE",as_index=False).sum()[["ADM3_PCODE","cases_number"]]
```

```python
gdf_adm3_merge=gdf_adm3.merge(df_adm,on="ADM3_PCODE",how="outer")
```

```python
gdf_adm3_merge.plot(column="cases_number",
               legend=True,
               scheme="quantiles",
               missing_kwds={'color': 'lightgrey',"label":"no data"},
               figsize=(15,10),)
```

Check what the areas are where the adm2 pcode is reported in the plague data. 
It is not that many cases, so might be better to just remove these rows? 

```python
df_adm[df_adm.ADM3_PCODE.isin(missing_df_gdf)]
# df[df.ADM3_PCODE.isin(missing_df_gdf)]
```

```python
gdf_adm2=gpd.read_file(adm2_bound_path)
```

```python
gdf_adm2_merge=gdf_adm2.merge(df_adm,left_on="ADM2_PCODE",right_on="ADM3_PCODE",how="outer")
```

```python
gdf_adm2_merge.plot(column="cases_number",
               legend=True,
               scheme="quantiles",
               missing_kwds={'color': 'lightgrey',"label":"no data"},
               figsize=(15,10),)
```

### Cases 01-08-2021 to 20-09-2021


Questions:
- Bulletin shows aggregation by sex and age, we don't have that data? 

```python
df_sel=df[(df.date>="2021-08-01")&(df.date<="2021-09-20")]
```

They state that 37 cases were reported.. 

```python
df_sel
```

```python
df_sel.groupby("cases_class").sum()["cases_number"]
```

```python
df_sel.groupby("clinical_form").sum()["cases_number"]
```

In report this is 37

```python
df_sel.cases_number.sum()
```

```python
px.bar(df_sel.groupby(["date","cases_class"],as_index=False).sum(),x="date",y="cases_number",color="cases_class")
```

```python

```
