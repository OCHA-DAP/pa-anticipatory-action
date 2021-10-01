<!-- #region -->
### Analysis of plague data in Madagascar
This notebook explores plague data in Madagascar.


Note that this data is not publicly available. 
<!-- #endregion -->

The data already includes up to 26-09, i.e. week 38


Missing data:
- 2014-2016 data
- aggregation by sex and age
- resistance to antibiotics

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
```

```python
def preprocess_plague_data(path,list_cases_class=None):
    df = pd.read_csv(path, delimiter = ";")
    df.columns=df.columns.str.lower()
    df.rename(columns={"mdg_com_code":"ADM3_PCODE"},inplace=True)
    #to make pcodes correspond with shp file
    df.ADM3_PCODE=df.ADM3_PCODE.str.replace("MDG","MG")
    #In old data, there is an entry of week 53 in 2021, 
    #this cannot be correct so drop it 
    #not neatest method but good enough for now
    df=df[~((df.year==2021)&(df.week==53))]
    #create a datetime from the year and week as this is easier with plotting
    df["date"]=df.apply(lambda x: date.fromisocalendar(x.year,x.week,1),axis=1)
    df["date"]=pd.to_datetime(df["date"])
    if list_cases_class is not None:
        df=df[df.cases_class.isin(list_cases_class)]
    return df
```

```python
def plague_group_by_date(df):
    #group by date
    df_date=df.groupby(["date","year","week"],as_index=False).sum()
    df_date.set_index("date",inplace=True)
    #fill the weeks that are not included with 0, else they will be ignored when computing the historical average
    df_date=df_date.asfreq('W-Mon').fillna(0)
    #compute the year and week numbers from the dates
    df_date[["year","week"]]=df_date.index.isocalendar()[["year","week"]]
    df_date.reset_index(inplace=True)
    return df_date
```

```python
#define period of current interest
sel_start_date = "2021-08-02"
sel_end_date = "2021-09-20"
sel_start_week = 31
sel_end_week = 38
```

```python
#was suggested to only look at the probable and confirmed cases, not the suspected 
incl_cases_class=["PROB","CONF"]
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
country_data_private_processed_dir = os.path.join(config.DATA_DIR, config.PRIVATE_DIR,config.PROCESSED_DIR,iso3)
plot_dir = os.path.join(country_data_private_processed_dir,"plots","plague")
Path(plot_dir).mkdir(parents=True, exist_ok=True)
```

```python
plague_data_filename = "Madagascar_IPM_Plague_cases_Aggregated_2021-09-28.csv"
plague_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / config.RAW_DIR / iso3 / "institut_pasteur"
plague_path = plague_dir / plague_data_filename
```

```python
df=preprocess_plague_data(plague_path,list_cases_class=incl_cases_class)
```

```python
df=df.sort_values(["year","week"])
df
```

Read in urban classification for ADM3 areas.

```python
urban_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.PROCESSED_DIR / iso3 / "urban_classification"
urban_filename = "mdg_adm3_urban_classification.csv"
urban_path = urban_dir / urban_filename
```

```python
adm3_urban = pd.read_csv(urban_path, index_col=0)
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
df_date=plague_group_by_date(df)
```

```python
px.line(df_date,x="date",y="cases_number", title="Plague cases reported, 2017-2021")
```

### Compare new and old data set


We had an initital dataset and later received an updated one. Compare these two datasets. 
As can be seen in the graph, the cases in 2021 are slightly smoothed in the new dataset, which can be caused by newly available information. 
Numbers for previous years also changed for some dates, and we are thus far unclear why this occurred. 

```python
plague_data_filename_old = "Madagascar_IPM_Plague_cases_Aggregated_2021-09-24.csv"
plague_path_old = plague_dir / plague_data_filename_old
```

```python
df_old=preprocess_plague_data(plague_path_old,list_cases_class=incl_cases_class)

```

```python
df_old_date=plague_group_by_date(df_old)
```

```python
df_comb=df_date.merge(df_old_date,on="date",how="outer",suffixes=("_new","_old"))
```

```python
px.line(df_comb,x="date",y=["cases_number_new","cases_number_old"], title="Plague cases reported, 2017-2021")
```

```python
#small attempt to understand the differences between the two datasets
bla=pd.concat([df,df_old]).drop_duplicates(keep=False)
bla[bla.date>=df_old.date.min()].sort_values("date")
```

```python
# merge_cols=['district', 'commune', 'ADM3_PCODE', 'year', 'week', 'clinical_form',
#        'cases_class', 'status', 'date']
# bla=df.merge(df_old, on=merge_cols, how= 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
# bla[bla.date>=df_old.date.min()]
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
def plot_adm3(df,title="",predef_bins=[1,2,5,10,15,20,100,200]):
    fig,ax=plt.subplots(figsize=(15,10))
    gdf_adm3_merge=gdf_adm3.merge(df,on="ADM3_PCODE",how="outer")
    scheme = None
    norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    legend_kwds = None
    colors = None
    gdf_adm3_merge.plot(
            column="cases_number",
            legend=True,
            k=colors,
            cmap="YlOrRd",
            norm=norm,
            scheme=scheme,
            legend_kwds=legend_kwds,
            missing_kwds={
                "color": "lightgrey",
            },
        ax=ax
        )
    ax.set_axis_off()
    ax.set_title(title)
```

```python
df_adm = df.groupby("ADM3_PCODE",as_index=False).sum()[["ADM3_PCODE","cases_number"]]
```

```python
plot_adm3(df_adm)
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

Where the ADM3 codes are available, will add the urban classification for analysis. For ease, adding to `df` and recalculating `df_date` for use with urban/rural breakdown if necessary.

```python
df_urb = pd.merge(df, adm3_urban[["ADM3_PCODE", "urban_area"]], on="ADM3_PCODE", how="left")
df_urb = df_urb[df_urb.urban_area.notnull()]

#group by date
df_date_urb=df_urb.loc[df_urb.urban_area].groupby(["date","year","week"],as_index=False).sum()
df_date_urb.set_index("date",inplace=True)

#add 0 to scale to September since data is missing
df_date_urb = df_date_urb.append(pd.DataFrame([[0]],columns=["cases_number"], index=["2021-09-28"]))
df_date_urb.index.names=["date"]

#fill the weeks that are not included with 0, else they will be ignored when computing the historical average
df_date_urb=df_date_urb.asfreq('W-Mon').fillna(0)
#compute the year and week numbers from the dates
df_date_urb[["year","week"]]=df_date_urb.index.isocalendar()[["year","week"]]
df_date_urb.reset_index(inplace=True)
```

```python
px.line(
    df_date_urb,
    x="date",
    y="cases_number",
    title="Cases reported in urban areas, 2017 - 2021"
)
```

There are extremely few cases identified in urban areas since 2017, in fact, none in 2020 and only 1 at the beginning of 2021. Even a histogram can't display this well, the table below suffices:

```python
df_date.groupby("year",as_index=False).sum() \
    .drop("week",axis=1)
```

```python
df_date_urb.groupby("year",as_index=False).sum() \
    .drop("week",axis=1)
```

### Historical average


According to the bulletin the average is computed by using the data from 2014,2015,2016,2018,and 2019 is used. 
For each week the numbers of that week, the two weeks before, and the two weeks after are taken. I.e. it is a rolling sum with a window length of 5 and centred in the middle. 


We only have data starting from 2017.. We will therefore for now use the data we have from 2018 till 2020 which is 3 years.
2017 is excluded since this year saw a large outbreak, so it will influence the historical average in a non-representable way. 


Questions:
- Is the std computed over the years or over all values included, so before the rolling sum? The bulletin: For each week, the threshold is calculated by adding 1.64 standard deviations to the historical average
I now only computed the std over the years, so basically over 3 values, and not over 5*3=15 values


1.64std represents a 90% confidence interval. I.e. 5/100 events are above the 1.64 threshold

```python
hist_avg_years=[2018,2019,2020]
```

```python
#compute the historical average
df_hist_years=df_date[df_date.year.isin(hist_avg_years)]
df_hist_years["rolling_sum"]=df_hist_years.cases_number.rolling(window=5,center=True).mean()
```

```python
df_hist_weeks=df_hist_years.groupby("week")["rolling_sum"].agg(rs_mean="mean",rs_std="std").reset_index()
```

```python
df_hist_weeks["164std"]=df_hist_weeks.rs_std*1.64
df_hist_weeks["plus_164std"]=df_hist_weeks.rs_mean+df_hist_weeks["164std"]
df_hist_weeks["min_164std"]=df_hist_weeks.rs_mean+df_hist_weeks["164std"]
```

```python
df_hist_weeks.head()
```

```python
base = alt.Chart(df_hist_weeks).transform_calculate(
    line="'hist mean'",
    shade1="'hist +1.64std'",
)
scale = alt.Scale(domain=["hist mean", "hist +1.64std"], range=['red', 'yellow'])
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

alt.layer(line_std, band_std, line_avg).properties(
    width=500,
    height=300,
    title = "Historical average and 1.64std"
)
```

### Define functions key figures

```python
def plot_hist_avg(df):
    line = alt.Chart(df).mark_line(color="red").encode(
        x='week:N',
        y=alt.Y('rs_mean',title="number of cases"),

    )
    line_std = alt.Chart(df).mark_line(color="yellow").encode(
        x='week:N',
        y='plus_164std'
    )

    band_std = alt.Chart(df).mark_area(
        opacity=0.5, color='gray'
    ).encode(
        x="week:N",
        y='rs_mean',
        y2='plus_164std',
    )
    
    return alt.layer(line_std, band_std, line)
```

```python
def plot_adm3(df,title=""):
    fig,ax=plt.subplots(figsize=(15,10))
    gdf_adm3_merge=gdf_adm3.merge(df,on="ADM3_PCODE",how="outer")
    predef_bins=[1,2,5,10,15,20,100,1000]
    scheme = None
    norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    legend_kwds = None
    colors = None
    gdf_adm3_merge.plot(
            column="cases_number",
            legend=True,
            k=colors,
            cmap="YlOrRd",
            norm=norm,
            scheme=scheme,
            legend_kwds=legend_kwds,
            missing_kwds={
                "color": "lightgrey",
            },
        ax=ax
        )
    ax.set_axis_off()
    ax.set_title(title)
```

```python
def key_graphs(df,title="", all_weeks=False):
    df_status=df.groupby(['clinical_form', 'cases_class','status'])['cases_number'].sum().unstack()
    df_status.fillna(0,inplace=True)
    df_status["total"]=df_status.sum(axis=1)
    df_status["perc_decede"]=round(df_status.DECEDE/df_status.total*100,2)
    display(df_status)
    
    df_clin=df.drop(["year","week"],axis=1).groupby("clinical_form").sum()
    df_clin["cases_perc"]=round(df_clin["cases_number"]/sum(df_clin["cases_number"])*100,2)
    display(df_clin)
    
    #geographical distribution
    df_adm = df.groupby("ADM3_PCODE",as_index=False).sum()[["ADM3_PCODE","cases_number"]]
    plot_adm3(df_adm,title=title)
    
#     df_dist=df.groupby(['district','cases_class'])['cases_number'].sum().unstack()
#     df_dist.fillna(0,inplace=True)
#     df_dist["total"]=df_dist.sum(axis=1)
#     df_dist=df_dist.sort_values("total",ascending=False)
#     display(df_dist.head(n=10))
    
    #temporal distribution
    if all_weeks:
        df_hist_weeks_sel = df_hist_weeks
    else:
        df_hist_weeks_sel=df_hist_weeks[df_hist_weeks.week.isin(df.week.unique())]
    
    graph_hist = plot_hist_avg(df_hist_weeks_sel)
    
    #group by date
    df_date=df.groupby(["date","year","week"],as_index=False).sum()
    df_date.set_index("date",inplace=True)
    #fill the weeks that are not included with 0, else they will be ignored when computing the historical average
    df_date=df_date.asfreq('W-Mon').fillna(0)
    #compute the year and week numbers from the dates
    df_date[["year","week"]]=df_date.index.isocalendar()[["year","week"]]
    bar = alt.Chart(df_date).mark_bar().encode(
    x='week:N',
    y='sum(cases_number)'
    )
    graph = (bar+graph_hist).properties(
    width=500,
    height=300,
    title=title,
    )
    display(graph)
```

### Cases 2021
With a focus on 01-08-2021 to 26-09-2021

```python
df_sel=df[(df.date>=sel_start_date)&(df.date<=sel_end_date)]
```

```python
df_sel.head()
```

```python
df_sel.cases_number.sum()
```

```python
#have to change order of nb for this to work on restart
key_graphs(df_sel,title="Cases in week 31-38 2021")
```

```python
df_dist=df_sel.groupby(['district','cases_class'])['cases_number'].sum().unstack()
df_dist.fillna(0,inplace=True)
df_dist["total"]=df_dist.sum(axis=1)
df_dist
```

```python
df_sel.drop(["year","week"],axis=1).groupby("cases_class").sum()
```

Basic graphs from report, not being used so visuals are not optimized

```python
df_class_date=df_sel.groupby(["date","year","week","cases_class"],as_index=False).sum()
```

```python
alt.Chart(df_class_date).mark_bar().encode(
    x='week:N',
    y='sum(cases_number)',
    color = "cases_class:N"
)
```

```python
df_class_date=df_sel.groupby(["date","year","week","cases_class"],as_index=False).sum()
```

```python
alt.Chart(df_sel[df_sel.clinical_form=="PP"].
       groupby(["date","week","year","cases_class"],as_index=False).sum()).mark_bar().encode(
    x='week:N',
    y='sum(cases_number)',
    color = "cases_class:N"
)
```

```python
alt.Chart(df_sel[df_sel.clinical_form=="PB"].
       groupby(["date","week","year","cases_class"],as_index=False).sum()).mark_bar().encode(
    x='week:N',
    y='sum(cases_number)',
    color = "cases_class:N"
)
```

#### Figures to be shared on 2021

```python
color_twentyone='#7f2100'
```

TODO: add legend

```python
base_2021 = alt.Chart(df_date[df_date.year==2021]).transform_calculate(
    cases="'cases 2021'",
)
scale_2021 = alt.Scale(domain=["hist mean", "hist +1.64std","cases 2021"], range=['red', 'yellow',color_twentyone])
```

```python
bar_2021 = base_2021.mark_bar(color=color_twentyone).encode(
    x='week:N',
    y=alt.Y('cases_number',title="number of cases"),
    color=alt.Color('cases:N', scale=scale_2021, title='')
)

chart_2021 = (bar_2021 + line_std + band_std + line_avg).properties(
    width=600,
    height=300,
    title = "Number of cases in 2021 and historical average"
) 
chart_2021
##not working :( need some packages installed which is complicated
# chart_2021.save(os.path.join(plot_dir,f"{iso3}_cases_histavg_2021.png"))
```

```python
df_hist_weeks_sel = df_hist_weeks[df_hist_weeks.week.isin(range(sel_start_week,sel_end_week+1))]
line_avg_sel = alt.Chart(df_hist_weeks_sel).mark_line(color="red").encode(
    x='week:N',
    y='rs_mean'
)
line_std_sel = alt.Chart(df_hist_weeks_sel).mark_line(color="yellow").encode(
    x='week:N',
    y='plus_164std'
)

band_std_sel = alt.Chart(df_hist_weeks_sel).mark_area(
    opacity=0.5, color='gray'
).encode(
    x='week:N',
    y=alt.Y('rs_mean',title="number of cases"),
    y2='plus_164std',
)

bar_2021_sel = alt.Chart(df_sel).mark_bar(color=color_twentyone).encode(
    x='week:N',
    y=alt.Y('sum(cases_number)',title="number of cases")
)

(bar_2021_sel + line_std_sel + band_std_sel + line_avg_sel).properties(
    width=600,
    height=300,
    title="Cases from week 31 to 38"
) 
```

### Compare 2021 and 2017

```python
color_seventeen='#007ce0'
```

```python
chart_2017_2021=alt.Chart(df_date[df_date.year.isin([2017,2021])]).mark_line().encode(
    x='week:N',
    y=alt.Y('cases_number',title="number of cases"),
    color=alt.Color('year:N', scale=alt.Scale(range=[color_seventeen,color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Cases in 2017 and 2021"
)
chart_2017_2021
# # you need to have altair_saver installed to make this work
# # also the scale_factor is not working with all installations
# chart_2017_2021.save(os.path.join(plot_dir,f"{iso3}_cases_2017_2021.png"),scale_factor=20)
```

Rather than looking at standard deviation, because we have so few years to work with, I think more informative to look at the other years outside 2017 to show if a spike as we've seen that then recedes has been observed in the past.

```python
chart_2018_2021=alt.Chart(df_date[df_date.year.isin([2018,2019,2020,2021])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3",color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Cases from 2018 to 2021"
)
chart_2018_2021
# chart_2018_2021.save(os.path.join(plot_dir,f"{iso3}_cases_2018_till_2021.png"),scale_factor=20)
```

```python
chart_1721_sel = alt.Chart(df_date[(df_date.year.isin([2017,2021]))&(df_date.week.isin(range(sel_start_week,sel_end_week+1)))]).mark_bar(width=5).encode(
    x=alt.X('year:N', scale=alt.Scale(domain=['', 2017, 2021]),title=None),
    y=alt.Y('cases_number',axis=alt.Axis(grid=False)),
    color=alt.Color('year:N', scale=alt.Scale(range=[color_seventeen,color_twentyone])),
).properties(
    width=30,
    height=300
).facet(
    'week:N', spacing=0
).configure_view(
    strokeWidth=0
).properties(title="Cases in 2017 and 2021 in week 31-38")
chart_1721_sel

# #not working
# chart_1721_sel.save(os.path.join(plot_dir,f"{iso3}_cases_2017_2021_3138.png"))
```

#### Only pneunomic cases

```python
df_date_pp=df[df.clinical_form=="PP"].groupby(["date","year","week"],as_index=False).sum()
```

```python
chart_1721_pp = alt.Chart(df_date_pp[df_date_pp.year.isin([2017,2021])]).mark_line().encode(
    x='week:N',
    y=alt.Y('cases_number',title="number of cases"),
    color=alt.Color('year:N', scale=alt.Scale(range=[color_seventeen,color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Pneunomic cases in 2017 and 2021"
)
chart_1721_pp
# chart_1721_pp.save(os.path.join(plot_dir,f"{iso3}_pp_cases_2017_2021.png"))
```

```python
chart_2018_2021_pp=alt.Chart(df_date_pp[df_date_pp.year.isin([2018,2019,2020,2021])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3",color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Pneumonic cases from 2018 to 2021"
)
chart_2018_2021_pp
# chart_2018_2021_pp.save(os.path.join(plot_dir,f"{iso3}_pp_cases_2018_till_2021.png"))
```

```python
chart_1721_pp_sel = alt.Chart(df_date_pp[(df_date_pp.year.isin([2017,2021]))&(df_date_pp.week.isin(range(sel_start_week,sel_end_week+1)))]).mark_bar(width=5).encode(
    x=alt.X('year:N', scale=alt.Scale(domain=['', 2017, 2021]),title=None),
    y=alt.Y('cases_number',axis=alt.Axis(grid=False)),
    color=alt.Color('year:N', scale=alt.Scale(range=[color_seventeen,color_twentyone])),
).properties(
    width=30,
    height=300,
    title="Number of pneunomic cases in week 31 to 38"
).facet(
    'week:N', spacing=0
).configure_view(
    strokeWidth=0
).properties(title="Pneunomic cases in 2017 and 2021 in week 31-38")
chart_1721_pp_sel

# #not working
# chart_1721_pp_sel.save(os.path.join(plot_dir,f"{iso3}_pp_cases_2017_2021_3138.png"))
```

#### Compare years


Functions are very ugly as of now, have to be improved if we use them. Just used for some exploration now

```python
key_graphs(df[df.year==2017],title="Cases in 2017")
```

```python
key_graphs(df[df.year==2018],title="Cases in 2018")
```

```python
key_graphs(df[df.year==2019],title="Cases in 2019")
```

```python
key_graphs(df[df.year==2020],title="Cases in 2020")
```

```python
key_graphs(df[df.year==2021],title="Cases in 2021")
```

```python
# key_graphs(df[df.year==2018])
```

```python
key_graphs(df[df.year==2019],title="Cases in 2019")
```

```python

```

```python
key_graphs(df[(df.year==2017)&(df.clinical_form=="PP")],title="Pneumonic cases in 2017")
```

```python
key_graphs(df[(df.year==2019)&(df.clinical_form=="PP")],title="Pneumonic cases in 2019")
```

```python
key_graphs(df[(df.year==2021)&(df.clinical_form=="PP")],title="Pneumonic cases in Aug-Sep 2021")
```

```python
key_graphs(df[(df.year==2021)&(df.week>=31)],title="Cases in Aug-Sep 2021")
```

```python
df
```

### Conclusions
- Not exceptional that there is one week with cases above std (with current data), see e.g. 2019


### Questions:
- How do we average current numbers? The historical average is based on a rolling centred sum, we cannot do that with current numbers. Would taking a right rolling sum of 3 weeks (instead of 5) suffice? Though then probably underestimating.. Think if we do averaging of current numbers, should follow some methodology as for the historical average (i.e. have to change historical average method)


```python

```

```python

```

```python

```

### Archive

```python

```

```python

# line_2017_2021_sel= alt.Chart(df[(df.year.isin([2017,2021]))&(df.week.isin(range(sel_start_week,sel_end_week+1)))]).mark_line().encode(
#     x='week:N',
#     y='sum(cases_number)',
#     color=alt.Color('year:N', scale=alt.Scale(range=['#f2645a','#007ce0']))
# )

# line_bar = line_2017_2021_sel #+ line_std
# line_bar.properties(
#     width=600,
#     height=300
# )
```

```python
# #showing whole year as bar chart gets too messy
# chart = alt.Chart(df[df.year.isin([2017,2021])]).mark_bar(width=5).encode(
#     x=alt.X('year:N', scale=alt.Scale(domain=['', 2017, 2021]),title=None),
#     y=alt.Y('cases_number',axis=alt.Axis(grid=False)),
#     color=alt.Color('year:N', scale=alt.Scale(range=['#f2645a','#007ce0'])),
# ).properties(
#     width=30,
#     height=300
# ).facet(
#     'week:N', spacing=0
# ).configure_view(
#     strokeWidth=0
# )
# chart
```

```python
# df_yearweek=df[df.year.isin([2017,2021])].groupby(["year","week"],as_index=False).sum()
# df_yearweek.year=df_yearweek.year.astype("str")
# px.bar(df_yearweek, x="week", y="cases_number",
#              color='year', 
#        barmode='group',
#              height=400)
```

```python
# df_yearweek=df.groupby(["year","week"],as_index=False).sum()
# df_yearweek.year=df_yearweek.year.astype("str")
# px.bar(df_yearweek, x="week", y="cases_number",
#              color='year', 
#        barmode='group',
#              height=400)
```

Altairs definition of "ci" is a 95% confidence interval
