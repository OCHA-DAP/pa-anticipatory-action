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
import numpy as np
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
    list(gdf_adm3["ADM3_PCODE"].dropna()),
)
print(f"Pcodes in plague data but not in adm3 shapefile: {missing_df_gdf}")

missing_gdf_df = np.setdiff1d(
    list(gdf_adm3["ADM3_PCODE"].dropna()),
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
- We also don't have 2014-2016 data to compute historical average..

```python
df_sel=df[(df.date>="2021-08-01")&(df.date<="2021-09-20")]
```

They state that 37 cases were reported.. 

```python
df_sel.cases_number.sum()
```

```python
df_sel
```

```python
df_sel.drop(["year","week"],axis=1).groupby("cases_class").sum()
```

```python
df_clin=df_sel.drop(["year","week"],axis=1).groupby("clinical_form",as_index=False).sum()
df_clin["cases_perc"]=round(df_clin["cases_number"]/sum(df_clin["cases_number"])*100,2)
df_clin
```

```python
df_status=df_sel.groupby(['clinical_form', 'cases_class','status'])['cases_number'].sum().unstack()
df_status.fillna(0,inplace=True)
df_status["total"]=df_status.DECEDE+df_status.VIVANT
df_status["perc_decede"]=round(df_status.DECEDE/df_status.total*100,2)
df_status
```

In report no NP cases, PP cases align, PB had 2 less CONF cases (1 decede and 1 vivant)


![afbeelding.png](attachment:adef0948-ff9c-42d6-ab9d-9272cdd9dae9.png)


Todo: fix dates and colors (or use other tool than plotly)

```python
px.bar(df_sel.groupby(["date","cases_class"],as_index=False).sum(),x="date",y="cases_number",color="cases_class")
```

```python
px.bar(df_sel[df_sel.clinical_form=="PP"].
       groupby(["date","cases_class"],as_index=False).sum(),
       x="date",
       y="cases_number",
       color="cases_class",
      title="Cases of Pneumonic Plague by class")
```

```python
px.bar(df_sel[df_sel.clinical_form=="PB"].
       groupby(["date","cases_class"],as_index=False).sum(),
       x="date",
       y="cases_number",
       color="cases_class",
      title="Cases of Bubonic Plague by class")
```

### Historical average


According to the bulletin the average is computed by using the data from 2014,2015,2016,2018,and 2019 is used. 
For each week the numbers of that week, the two weeks before, and the two weeks after are taken. I.e. it is a rolling sum with a window length of 5 and centred in the middle. 


We only have data starting from 2017.. We will therefore for now use the data we have from 2018 till 2020 which is 3 years.
2017 is excluded since this year saw a large outbreak, so it will influence the historical average in a non-representable way. 


Questions:
- Is the std computed over the years or over all values included, so before the rolling sum? The bulletin: For each week, the threshold is calculated by adding 1.64 standard deviations to the historical average


1.64std represents a 90% confidence interval. I.e. 5/100 events are above the 1.64 threshold

```python
#group by date
df_date=df.groupby(["date","year","week"],as_index=False).sum()
df_date.set_index("date",inplace=True)
```

```python
#fill the weeks that are not included with 0, else they will be ignored when computing the historical average
df_date=df_date.asfreq('W-Mon').fillna(0)
#compute the year and week numbers from the dates
df_date[["year","week"]]=df_date.index.isocalendar()[["year","week"]]
```

```python
df_date
```

```python
#compute the historical average
df_hist_years=df_date[df_date.year.isin([2018,2019,2020])]
df_hist_years["rolling_sum"]=df_hist_years.cases_number.rolling(window=5,center=True).mean()
df_hist_weeks=df_hist_years.groupby("week",as_index=False).agg(["mean","std"]).drop("year",axis=1) #agg(["mean","count"])
```

```python
df_hist_weeks.head()
```

```python
df_2021 = df[df.year==2021]
df_2017 = df[df.year==2017]
```

How to add legend to this graph? 


Altairs definition of "ci" is a 95% confidence interval

```python
import altair as alt

line = alt.Chart(df_hist_years).mark_line().encode(
    x='week',
    y='mean(rolling_sum)'
)

band = alt.Chart(df_hist_years).mark_errorband(extent='ci').encode(
    x='week',
    y=alt.Y('rolling_sum', title='cases/week'),
)

# line_std = alt.Chart(df_hist_weeks).mark_line().encode(
#     x='week',
#     y='rolling_sum["std"]'
# )

line_2021 = alt.Chart(df_2021).mark_line(color="red").encode(
    x="week",
    y="sum(cases_number)",
)

line_2017 = alt.Chart(df_2017).mark_line(color="green").encode(
    x="week",
    y="sum(cases_number)",
)

band + line + line_2021 + line_2017
```

```python

```

#### Next steps
- Urban vs rural
- Do we agree with the historical average method? 
- Inspect situation in 2017

```python

```
