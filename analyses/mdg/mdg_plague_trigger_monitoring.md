### Trigger monitoring of plague data in Madagascar
This notebook monitors the data and whether it met the trigger. 
`mdg_plague_exploration.md` provides explanation for the trigger

```python
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
```

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[0]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.plague import plague
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
```

```python
country_data_private_processed_dir = os.path.join(config.DATA_DIR, config.PRIVATE_DIR,config.PROCESSED_DIR,iso3)
plot_dir = os.path.join(country_data_private_processed_dir,"plots","plague")
Path(plot_dir).mkdir(parents=True, exist_ok=True)
```

```python
urban_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3 / "urban_classification"
urban_filename = "mdg_adm3_urban_classification.csv"
urban_path = urban_dir / urban_filename
```

```python
#define period of monitoring
mon_start_date = pd.Timestamp(date.fromisoformat("2022-01-01"))
mon_end_date = pd.Timestamp(date.fromisoformat("2022-12-31"))
```

```python

```

```python
#was suggested to only look at the probable and confirmed cases, not the suspected
incl_cases_class=["PROB","CONF"]
```

```python
plague_filename,date_max=plague.latest_plague_path()
```

```python
plague_path = Path(plague.plague_raw_dir) / plague_filename
```

```python
plague_filename
```

```python
df=plague.load_plague_data(plague_path,keep_cases=incl_cases_class)
# new end dates for creating 2023 graph
mon2_start_date = pd.Timestamp(date.fromisoformat("2023-01-01"))
mon2_end_date = df.date.max()
```

```python
# a separate 2012 to 2016 data for analysis
# this is the disaggregated 2012-2016 data
plague_path_2012_2016_v2 = Path(plague.plague_raw_dir) / "Madagascar_IPM_Plague_cases_Aggregated_historic_2021-10-18.csv"
df_2012_2016_v2 = plague.load_plague_data(plague_path_2012_2016_v2,
                                          keep_cases=incl_cases_class,
                                          delimiter=";")


df = df.append(df_2012_2016_v2)
```

```python
df_sel=df[(df.date>=mon_start_date)&(df.date<=mon_end_date)]
```

##### Validity adm codes
Check correspondence data and shapefile. 

It happens that cases are reported at adm2 instead of adm3. Check if that is the case during the monitoring period

```python
gdf_adm3=gpd.read_file(adm3_bound_path)
```

```python
# Check that admin level names in the plague data and shapefile match
missing_df_gdf = np.setdiff1d(
    list(df_sel["ADM3_PCODE"].dropna()),
    list(gdf_adm3["ADM3_PCODE"].dropna()),
)
print(f"Pcodes in plague data but not in adm3 shapefile: {missing_df_gdf}")
df_sel[df_sel.ADM3_PCODE.isin(missing_df_gdf)]
```

```python
adm3_urban = pd.read_csv(urban_path, index_col=0)
```

```python
df = pd.merge(df, adm3_urban[["ADM3_PCODE", "urban_area_weighted_13"]], on="ADM3_PCODE", how="left")
# then filter to only urban areas with pneumonic plague
# df_urb = df_urb.loc[df_urb.urban_area_weighted_13 & (df_urb.clinical_form == "PP")]
df['urb_pp_cases']=np.where((df.urban_area_weighted_13) & (df.clinical_form == "PP"),df.cases_number,0)
df['urb_cases']=np.where(df.urban_area_weighted_13,df.cases_number,0)
#group by date
df['pp_cases']=np.where(df.clinical_form == "PP",df.cases_number,0)
df_date=plague.aggregate_to_date(df,cases_cols=["cases_number","urb_pp_cases","pp_cases"])
# df_date_urb=plague_group_by_date(df_urb, mon_start_date="2012-01-01", mon_end_date=mon_end_date)
```

```python
df_date=plague.aggregate_to_date(df)
```

```python
df_date_sel=df_date[(df_date.date>=pd.to_datetime(mon_start_date, format="%Y-%m-%d"))&(df_date.date<=pd.to_datetime(mon_end_date, format="%Y-%m-%d"))]
df_date_sel2=df_date[(df_date.date>=pd.to_datetime(mon2_start_date, format="%Y-%m-%d"))&(df_date.date<=pd.to_datetime(mon2_end_date, format="%Y-%m-%d"))]
df_date_sel2
```

```python
df_date_sel
```

### Cases over time
Very basic plot with cases over time

```python
df_sum=df.groupby("date").sum()
df_roll=df_sum.cases_number.rolling(window=3).sum()
df_roll=df_roll.to_frame().reset_index()
plt_roll = (
    alt.Chart()
    .mark_line()
    .encode(
        x=alt.X("date:T",axis=alt.Axis(format='%b %Y')),
        y=alt.Y("cases_number:Q", title="3-week sum of cases"),
    )
    .properties(
        width=1000,
        height=500,
    )
)

line = (
    alt.Chart()
    .mark_rule(color="red")
    .encode(
        y="a:Q",
    )
)

alt.layer(plt_roll, line, data=df_roll,).transform_calculate(a="50").properties(title=["3-week rolling sum of cases","the red line indicates the threshold"])
```

```python
px.line(
    df_date,
    x="date",
    y="urb_pp_cases",
    title="Pneumonic plague cases reported in urban areas, 2012 - 2022"
)
```

### Compute trigger

```python
def comp_thresh_rolling_cases(df,cap=50,cases_col="cases_number",window=3):
    df[f"rolling_sum_{cases_col}"]=df[cases_col].rolling(window=window).sum()
    df[f"thresh_reached_{cases_col}"]=np.where(df[f"rolling_sum_{cases_col}"]>=cap,True,False)
    return df.sort_values("date")
```

```python
df=comp_thresh_rolling_cases(df_date_sel,cases_col="cases_number")
```

```python
df_date_sel=comp_thresh_rolling_cases(df_date_sel,cases_col="cases_number")
```

```python
df_date_sel=comp_thresh_rolling_cases(df_date_sel,cap=10,cases_col="urb_pp_cases")
```

```python
print(f"The trigger has been reached {len(df_date_sel[(df_date_sel.thresh_reached_cases_number)|(df_date_sel.thresh_reached_urb_pp_cases)])}" 
      " times during the monitoring period")
```

```python
print(f"The trigger on all cases has been reached "
      f"{len(df_date_sel[df_date_sel.thresh_reached_cases_number])} " 
      "times during the monitoring period")
```

```python
print(f"The condition of urban pneunomic cases has been reached {len(df_date_sel[df_date_sel.thresh_reached_urb_pp_cases])}" 
      " times during the monitoring period")
```

```python
print(f"Total number of cases during monitoring period: {int(df_date_sel.cases_number.sum())}")
```

```python
bar_weekly = alt.Chart(df_date_sel).mark_bar().encode(
    x='week:N',
    y=alt.Y('sum(cases_number)',title="weekly cases")
    ).properties(width=300,height=100).facet(
    facet='year:N',
    columns=2
    ).properties(
    title='Weekly cases during monitoring period'
).resolve_scale(x="independent")
# bar_weekly.save(os.path.join(plot_dir,f"{iso3}_cases_weekly_{mon_end_date.strftime('%Y%m%d')}.svg"))
bar_weekly
```

```python
bar_3week = alt.Chart(df_date_sel).mark_bar().encode(
    x='week:N',
    y=alt.Y('rolling_sum_cases_number',title="3-week sum of cases")
    ).properties(width=300,height=100).facet(
    facet='year:N',
    columns=2
    ).properties(
    title='3-week rolling sum of cases during monitoring period'
).resolve_scale(x="independent")
# bar_3week.save(os.path.join(plot_dir,f"{iso3}_cases_3weeksum_{mon_end_date.strftime('%Y%m%d')}.svg"))
bar_3week
```

```python
alt.Chart(df_date_sel[~df_date_sel.rolling_sum_cases_number.isnull()]).mark_line(point=True).encode(
    x="date:T",
    y="rolling_sum_cases_number"
).facet('year:N'
       ).properties(title="3-week rolling sum during monitoring period"
                   ).resolve_scale(x="independent")
```

```python
alt.Chart(df_date_sel).mark_bar().encode(
    x='week:N',
    y=alt.Y('rolling_sum_urb_pp_cases',title="cases")
    ).properties(width=300,height=100).facet(
    facet='year:N',
    columns=2
    ).properties(
    title='3-week rolling sum of urban pneunomic cases during monitoring period'
).resolve_scale(x="independent")
```

```python
def groupby_status(df):    
    df_status=df.groupby(['clinical_form', 'cases_class','status'])['cases_number'].sum().unstack()
    df_status.fillna(0,inplace=True)
    df_status["total"]=df_status.sum(axis=1)
    df_status["perc_decede"]=round(df_status.DECEDE/df_status.total*100,2)
    display(df_status)
    
    df_clin=df.drop(["year","week"],axis=1).groupby("clinical_form").sum()
    df_clin["cases_perc"]=round(df_clin["cases_number"]/sum(df_clin["cases_number"])*100,2)
    display(df_clin)
```

```python
groupby_status(df_sel)
```

#TODO: this needs some cleaning up still


### Comparing to historical average


According to the bulletin the average is computed by using the data from 2014,2015,2016,2018,and 2019 is used. 
For each week the numbers of that week, the two weeks before, and the two weeks after are taken. I.e. it is a rolling sum with a window length of 5 and centred in the middle. 


We have data starting from 2012.. We will use the full data we have from 2012 till 2020 which is 8 years.
2017 is excluded since this year saw a large outbreak, so it will influence the historical average in a non-representable way. 


1.64std represents a 90% confidence interval. I.e. 5/100 events are above the 1.64 threshold

```python
hist_avg_years=[2012,2013,2014,2015,2016,2018,2019,2020]
```

```python
#compute the historical average
df_hist_years=df_date[df_date.year.isin(hist_avg_years)]
df_hist_years["rolling_mean"]=df_hist_years.cases_number.rolling(window=5,center=True).mean()
```

```python
df_hist_weeks=df_hist_years.groupby("week")["rolling_mean"].agg(rs_mean="mean",rs_std="std",rs_max="max").reset_index()
```

```python
df_hist_weeks["164std"]=df_hist_weeks.rs_std*1.64
df_hist_weeks["plus_164std"]=df_hist_weeks.rs_mean+df_hist_weeks["164std"]
df_hist_weeks["min_164std"]=df_hist_weeks.rs_mean-df_hist_weeks["164std"]
```

```python
base = alt.Chart(df_hist_weeks).transform_calculate(
    line="'historical average'",
    shade1="'historical +1.64std'",
)
scale = alt.Scale(domain=["historical average", "historical +1.64std"], range=['red', 'yellow'])
```

```python
line_avg = base.mark_line(color="red").encode(
    x='week:N',
    y='rs_mean',
    color=alt.Color('line:N', scale=scale, title='')
)

line_std = base.mark_line(color="yellow").encode(
    x='week:N',
    y='plus_164std',
    color=alt.Color('shade1:N', scale=scale, title='')
)

band_std= base.mark_area(
    opacity=0.5, color='gray'
).encode(
    x='week:N',
    y=alt.Y('rs_mean',title="number of cases"),
    y2='plus_164std',
#     color=alt.Color('shade2:N', scale=scale, title=''),
)

# chart_hist_std = alt.layer(line_std, band_std, line_avg).properties(
#     width=500,
#     height=300,
#     title = "Historical average and 1.64std"
# )
# chart_hist_std
# # chart_hist_std.save(os.path.join(plot_dir,f"{iso3}_histavg_std.png"))
```

### Cases monitoring period

```python
color_twentyone='#7f2100'
```

```python
base_mon = alt.Chart(df_date_sel).transform_calculate(
    cases="'cases monitoring'",
)
base_mon2 = alt.Chart(df_date_sel2).transform_calculate(
    cases="'cases monitoring'",
)
scale_mon = alt.Scale(domain=["hist mean", "hist +1.64std","cases monitoring"], range=['red', 'yellow',color_twentyone])
```

#TODO: facet

```python
bar_mon = base_mon.mark_bar(color=color_twentyone).encode(
    x='week:N',
    y=alt.Y('cases_number',title="number of cases"),
    color=alt.Color('cases:N', scale=scale_mon, title='')
)
bar_mon2 = base_mon2.mark_bar(color=color_twentyone).encode(
    x='week:N',
    y=alt.Y('cases_number',title="number of cases"),
    color=alt.Color('cases:N', scale=scale_mon, title='')
)

chart_mon = (bar_mon + line_std + band_std + line_avg).properties(
    width=600,
    height=300,
    title = "Number of cases during 2022 and historical average"
)
chart_mon
# chart_mon.save(os.path.join(plot_dir,f"{iso3}_cases_histavg_{mon_end_date.strftime('%Y%m%d')}.svg"))
```

```python
chart_mon2 = (bar_mon2 + line_std + band_std + line_avg).properties(
    width=600,
    height=300,
    title = "Number of cases during 2023 and historical average"
)

chart_mon2
```

### Pneunomic cases
Chart should be improved

```python
plt_pneumonic = alt.Chart(df_date[df_date.year.isin(range(2018,2023))]).mark_line().encode(
    x='week:N',
    y='pp_cases',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3",color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Pneumonic cases from 2018 to 2022"
)
# plt_pneumonic.save(os.path.join(plot_dir,f"{iso3}_pneunomic_cases_{mon_end_date.strftime('%Y%m%d')}.svg"))
plt_pneumonic
```

### Urban cases

```python
print(f"Sum of urban cases in monitoring period: {int(df_date_sel.urb_cases.sum())}")
```
