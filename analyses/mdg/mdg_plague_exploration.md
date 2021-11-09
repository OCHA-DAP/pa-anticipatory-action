<!-- #region -->
### Analysis of plague data in Madagascar
This notebook explores plague data in Madagascar.


Note that this data is not publicly available. 
<!-- #endregion -->

The data already includes up to 26-09, i.e. week 38


Missing data:
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
```

```python
def max_week(year):
    """
    Check the max week in a year to prevent
    date errors in fromisocalendar() when a
    week 53 doesn't exist.
    
    Based off this post: https://stackoverflow.com/questions/60945041/setting-more-than-52-weeks-in-python-from-a-date
    """
    has_week_53 = date.fromisocalendar(year, 52, 1) + timedelta(days=7) != date.fromisocalendar(year + 1, 1, 1)
    if has_week_53:
        return 53
    else:
        return 52

def preprocess_plague_data(path,list_cases_class=None, delimiter = ";"):
    df = pd.read_csv(path, delimiter = delimiter)
    df.columns=df.columns.str.lower()
    df.rename(columns={"mdg_com_code":"ADM3_PCODE"},inplace=True)
    #to make pcodes correspond with shp file
    df.ADM3_PCODE=df.ADM3_PCODE.str.replace("MDG","MG")
    #In old data, there is an entry of week 53 in 2021, 
    #this cannot be correct so drop it 
    #not neatest method but good enough for now
    df=df[~((df.year==2021)&(df.week==53))]
    #create a datetime from the year and week as this is easier with plotting
    #first, make sure no invalid iso weeks (sometimes had week as 53 when max
    #iso weeks in a year were 52)
    df["max_week"] = [max_week(x) for x in df.year]
    df["week"] = df[["week", "max_week"]].min(axis=1)
    df["date"]=df.apply(lambda x: date.fromisocalendar(x.year,x.week,1),axis=1)
    df["date"]=pd.to_datetime(df["date"])
    
    #simplify names if long so all datasets match
    df.cases_class.replace(
        to_replace = ["CONFIRME", "SUSPECTE", "PROBABLE"],
        value = ["CONF", "SUSP", "PROB"],
        inplace = True
    )
    if list_cases_class is not None:
        df=df[df.cases_class.isin(list_cases_class)]
    return df
```

```python
def plague_group_by_date(df, sel_start_date=None, sel_end_date=None):
    #group by date
    df_date=df.groupby(["date","year","week"],as_index=False).sum()
    df_date.set_index("date",inplace=True)
    if sel_start_date is not None:
        df_date = df_date.append(pd.DataFrame([[0]],columns=["cases_number"], index=[pd.to_datetime(sel_start_date, format='%Y-%m-%d')]))
        df_date=df_date.sort_index()
    if sel_end_date is not None:
        df_date = df_date.append(pd.DataFrame([[0]],columns=["cases_number"], index=[pd.to_datetime(sel_end_date, format='%Y-%m-%d')]))
    df_date.index.names=["date"]
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
sel_end_date = "2021-10-01"
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
plague_data_filename = "Madagascar_IPM_Plague_cases_Aggregated_2021-10-01.csv"
plague_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / config.RAW_DIR / iso3 / "institut_pasteur"
plague_path = plague_dir / plague_data_filename
plague_data_filename_old = "Madagascar_IPM_Plague_cases_Aggregated_2021-09-28.csv"
plague_path_old = plague_dir / plague_data_filename_old
```

```python
df=preprocess_plague_data(plague_path,list_cases_class=incl_cases_class,delimiter=",")
```

```python
# a separate 2012 to 2016 data for analysis
# this is the disaggregated 2012-2016 data
plague_path_2012_2016_v2 = plague_dir / "Madagascar_IPM_Plague_cases_Aggregated_historic_2021-10-18.csv"
df_2012_2016_v2 = preprocess_plague_data(plague_path_2012_2016_v2,list_cases_class=incl_cases_class,delimiter=";")


df = df.append(df_2012_2016_v2)
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
df_date=plague_group_by_date(df, sel_end_date="2021-11-08")
```

```python
px.line(df_date,x="date",y="cases_number", title="Plague cases reported, 2012-2021")
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

Where the ADM3 codes are available, will add the urban classification for analysis. For ease, adding to `df` and recalculating `df_date` for use with urban/rural breakdown if necessary. The urban classification is using `urban_area_weighted` which is urban areas defined as communes where the average raster cell value is above a threshold. We initially tried with `>= 15`, but the code below uses `>= 13` now which is a permissive threshold but the historical data still shows very few urban cases of pneumonic plague outside of 2017 even with that threshold. Using the threshold system rather than % of cells helps capture areas where the majority of raster cells are not urban, but there are still significant urban agglomerations within the commune by using the inherent weighting in the GHS classification figures.

```python
df_urb = pd.merge(df, adm3_urban[["ADM3_PCODE", "urban_area_weighted_13"]], on="ADM3_PCODE", how="left")
# first filter out rows where the join failed (i.e. those with only ADM2 pcodes rather than ADM3)
df_urb = df_urb[df_urb.urban_area_weighted_13.notnull()]
# then filter to only urban areas with pneumonic plague
df_urb = df_urb.loc[df_urb.urban_area_weighted_13 & (df_urb.clinical_form == "PP")]

#group by date
df_date_urb=plague_group_by_date(df_urb, sel_start_date="2012-01-01", sel_end_date="2021-11-08")
```

```python
urb = px.line(
    df_date_urb,
    x="date",
    y="cases_number",
    title="Pneumonic plague cases reported in urban areas, 2012 - 2021"
)

urb.add_hline(y=5,annotation_text="5 cases",annotation_position="top left")
# requires kaleido package on mac, might be different on other machines
# urb.write_image(os.path.join(plot_dir,f"{iso3}_urban_timeline.png"))
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

It's also important to understand the distribution of urban areas in Madagascar. Here is a map of urban areas identified through the methodology.

```python
#urb_map_df = gdf_adm3.merge(adm3_urban[["ADM3_PCODE", "urban_area"]], on="ADM3_PCODE", how="left")
#urb_map_df['color'] = np.where(urb_map_df.urban_area, "#151515", "#FAFAFA")
#
#urb_plot = urb_map_df.plot(
#    color=urb_map_df["color"],
#    categorical=True,
#    legend=True,
#    edgecolor="face",
#    figsize=(15,10)
#)
#
# need to adjust size for saving, currently screenshotting
# urb_plot.figure.savefig(os.path.join(plot_dir,f"{iso3}_urban_areas.png"), bbox_inches="tight")
```

```python
#urb_map_df = gdf_adm3.merge(adm3_urban[["ADM3_PCODE", "urban_area_weighted"]], on="ADM3_PCODE", how="left")
#urb_map_df['color'] = np.where(urb_map_df.urban_area_weighted, "#151515", "#ECECEC")
#
#fig, ax = plt.subplots(1, 1)
#fig.set_figheight(15)
#
#urb_plot = urb_map_df.plot(
#    ax = ax,
#    color=urb_map_df["color"],
#    categorical=True,
#    legend=True,
#    edgecolor="face",
#    figsize=(15,10),
#)
#
#urb_plot.text(47.8,-19,"Antananarivo")
#urb_plot.text(48.8,-17.9,"Toamisina")
#urb_plot.text(47.2,-20,"Antsirabe")
#urb_plot.text(48.3,-22.2,"Manakara")
#urb_plot.text(48.1,-22.9,"Farafangana")
#urb_plot.text(49.4,-13.2,"Vohemar")
#urb_plot.text(46.5,-15.8,"Mahajanga")
#urb_plot.text(43.9,-23.45,"Toliara")
#urb_plot.text(47.3,-21.55,"Fianarantsoa")
#
#font_dict = {
#    'fontsize': 14,
#    'fontweight' : 'bold',
#    'horizontalalignment' : 'center'
#    }
#
#ax.set_title("Madagascar urban areas, defined through GHS",font_dict)
#ax.axis('off')
# need to adjust size for saving, currently screenshotting
# urb_plot.figure.savefig(os.path.join(plot_dir,f"{iso3}_urban_areas_weighted.png"), bbox_inches="tight")
```

```python
urb_map_df = gdf_adm3.merge(adm3_urban[["ADM3_PCODE", "urban_area_weighted_13"]], on="ADM3_PCODE", how="left")
urb_map_df['color'] = np.where(urb_map_df.urban_area_weighted_13, "#151515", "#ECECEC")

fig, ax = plt.subplots(1, 1)
fig.set_figheight(15)

urb_plot = urb_map_df.plot(
    ax = ax,
    color=urb_map_df["color"],
    categorical=True,
    legend=True,
    edgecolor="face",
    figsize=(15,10),
)

urb_plot.text(47.8,-19,"Antananarivo")
urb_plot.text(48.8,-17.9,"Toamisina")
urb_plot.text(47.2,-20,"Antsirabe")
urb_plot.text(48.3,-22.2,"Manakara")
urb_plot.text(48.1,-22.9,"Farafangana")
urb_plot.text(49.4,-13.2,"Vohemar")
urb_plot.text(46.5,-15.8,"Mahajanga")
urb_plot.text(47.8,-15.6,"Boriziny")
urb_plot.text(49,-15.9,"Mandritsara")
urb_plot.text(43.9,-23.45,"Toliara")
urb_plot.text(47.3,-21.55,"Fianarantsoa")
urb_plot.text(43.6,-20.1,"Morondava")
urb_plot.text(48.6,-12.1,"Antsiranana")
urb_plot.text(47.8,-13.2,"Nosy Be")
urb_plot.text(47.5,-20.6,"Ambositra")
urb_plot.text(47.7,-19.5,"Ambatolampy")
urb_plot.text(48.8,-17.2,"Mahambo")

font_dict = {
    'fontsize': 14,
    'fontweight' : 'bold',
    'horizontalalignment' : 'center'
    }

ax.set_title("Madagascar urban areas, defined through GHS",font_dict)
ax.axis('off')
# need to adjust size for saving, currently screenshotting
# urb_plot.figure.savefig(os.path.join(plot_dir,f"{iso3}_urban_areas_weighted.png"), bbox_inches="tight")
```

### Historical average


According to the bulletin the average is computed by using the data from 2014,2015,2016,2018,and 2019 is used. 
For each week the numbers of that week, the two weeks before, and the two weeks after are taken. I.e. it is a rolling sum with a window length of 5 and centred in the middle. 


We have data starting from 2012.. We will use the full data we have from 2012 till 2020 which is 8 years.
2017 is excluded since this year saw a large outbreak, so it will influence the historical average in a non-representable way. 


Questions:
- Is the std computed over the years or over all values included, so before the rolling sum? The bulletin: For each week, the threshold is calculated by adding 1.64 standard deviations to the historical average
I now only computed the std over the years, so basically over 3 values, and not over 5*3=15 values


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
df_hist_weeks.head()
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

chart_hist_std = alt.layer(line_std, band_std, line_avg).properties(
    width=500,
    height=300,
    title = "Historical average and 1.64std"
)
chart_hist_std
# chart_hist_std.save(os.path.join(plot_dir,f"{iso3}_histavg_std.png"))
```

### Define functions key figures

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
def plot_hist_avg(df):
    line = alt.Chart(df).mark_line(color="red").encode(
        x='week:N',
        y=alt.Y('rs_mean',title="number of cases"),

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
def plot_hist_cur(df_cur,df_hist,title=""):
    graph_hist = plot_hist_avg(df_hist)
    
    #group by date
    df_date=df_cur.groupby(["date","year","week"],as_index=False).sum()
    df_date.set_index("date",inplace=True)
    #fill the weeks that are not included with 0, else they will be ignored when computing the historical average
    df_date=df_date.asfreq('W-Mon').fillna(0)
    #compute the year and week numbers from the dates
    df_date[["year","week"]]=df_date.index.isocalendar()[["year","week"]]
    bar = alt.Chart(df_date).mark_bar().encode(
    x='week:N',
    y='sum(cases_number)'
    )
    graph_hist_cur = (bar+graph_hist).properties(
    width=500,
    height=300,
    title=title,
    )
    return graph_hist_cur
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
    graph_hist_cur = plot_hist_cur(df,df_hist_weeks_sel,title=title)
    display(graph_hist_cur)
```

### Cases 2021
With a focus on 01-08-2021 to 26-09-2021

```python
color_twentyone='#7f2100'
```

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

### Compare years

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
#chart_2017_2021.save(os.path.join(plot_dir,f"{iso3}_cases_2017_2021.png"),scale_factor=20)
```

Rather than looking at standard deviation, because we have so few years to work with, we look at the other years outside 2017 to show if a spike as we've seen that then recedes has been observed in the past. We do see that the 2021 is slightly more exceptional, but we have seen similair spikes in the past. 

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
#chart_2018_2021.save(os.path.join(plot_dir,f"{iso3}_cases_2018_till_2021.png"),scale_factor=20)
```

```python
chart_2012_2021=alt.Chart(df_date[df_date.year.isin([2012,2013,2014,2015, 2016, 2021])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3",color_twentyone]))
).properties(
    width=600,
    height=300,
    title="Cases from 2012 to 2016 and 2021"
)
chart_2012_2021
#chart_2012_2021.save(os.path.join(plot_dir,f"{iso3}_cases_2012_till_2016_2021.png"),scale_factor=20)
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
#chart_1721_sel.save(os.path.join(plot_dir,f"{iso3}_cases_2017_2021_3138.png"))
```

#### Only pneunomic cases

```python
df_date_pp=plague_group_by_date(df[df.clinical_form=="PP"], sel_end_date="2021-11-08")
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

### Defining a trigger
The idea is to have a trigger consisting of two parts: 
1) Based on total number of cases
2) Based on pneunomic cases in urban areas

We explore different options for both parts. Optimally the trigger would be reached as early as possible in September 2017 while not having been reached at any other point


#### All cases
One suggestion is to look at the historical mean and standard deviation. When the number of cases is larger than the mean+1.64std then we would trigger (i.e. 5/100 chance). What makes this problematic is that we only have 3 years of data to compute the std on which makes it fluctuate a lot. Another question is why we would want to trigger based on the std as this would mean a certain number of cases is "less bad" in one period of the year than in another.  

```python
def comp_std_consec(df_date,df_hist_avg,std=1.64):
    df_hist_avg[f"{std}std"]=df_hist_weeks.rs_std*std
    df_hist_avg[f"plus_{std}std"]=df_hist_weeks.rs_mean+df_hist_weeks[f"{std}std"]
    df_date=df_date.merge(df_hist_avg[["week",f"plus_{std}std"]],on="week",how="right")
    #do larger than instead of larger or equal than cause else when std=0 it would trigger
    df_date["thresh_reached"]=np.where(df_date.cases_number>df_date[f"plus_{std}std"],1,0)
    df_date=df_date.sort_values("date")
    df_date['consecutive'] = df_date[f"thresh_reached"].groupby( \
    (df_date[f"thresh_reached"] != df_date[f"thresh_reached"].shift()).cumsum()).transform('size') * \
    df_date[f"thresh_reached"]
    df_date["thresh_reached_str"]=df_date.thresh_reached.replace({0:"no",1:"yes"})
    return df_date
```

```python
df_164=comp_std_consec(df_date,df_hist_weeks)
```

Below the occurrences of cases>mean+1.64std are shown. We can see that this occurs quite commonly. We would therefore have to require 3 or 4 consecutive weeks where the condition is met

```python
heatmap_164 = alt.Chart(df_164).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title="larger than +1.64 std")),
).properties(
    title="> average + 1.64 std cases"
)
heatmap_164
#heatmap_164.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_std164.png"))
```

```python
df_164[df_164.consecutive>=4]
```

```python
chart_2018_2021_grey=alt.Chart(df_date[df_date.year.isin([2018,2019,2020,2021])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3"]))
)
```

```python
std_164 = alt.Chart(df_hist_weeks).mark_line(color="yellow").encode(
    x='week:N',
    y='plus_164std',
)
```

```python
chart_2018_2021_grey + std_164
```

Instead of looking at the fluctuations compared to the std, we can also look at the absolute number of observed cases. 

```python
def comp_abs_consec(df_date,cap=10,cases_col="cases_number"):
    df_date["thresh_reached"]=np.where(df_date[cases_col]>=cap,1,0)
    df_date=df_date.sort_values("date")
    df_date['consecutive'] = df_date[f"thresh_reached"].groupby( \
    (df_date[f"thresh_reached"] != df_date[f"thresh_reached"].shift()).cumsum()).transform('size') * \
    df_date[f"thresh_reached"]
    df_date["thresh_reached_str"]=df_date.thresh_reached.replace({0:"no",1:"yes"})
    return df_date
```

When we require >= 10 cases, we can see this occurring in 2017 and at 3 other points of time. However in 2017 this lasts for a consecutive period of time, so requiring 2 or 3 weeks of 10 consecutive cases might be a suitable trigger. 

With 2 weeks you might still risk having a false alarm while with 3 weeks you might be a bit late with the trigger

```python
df_cap10=comp_abs_consec(df_date,cases_col="cases_number")
```

```python
heatmap_abs10 = alt.Chart(df_cap10).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 10 cases")),
).properties(
    title=">= 10 cases"
)
heatmap_abs10
# heatmap_abs10.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_abs10.png"))
```

```python
week_window=3
df_date["rolling_mean"]=df_date.cases_number.rolling(window=week_window).mean()
df_date["rolling_sum"]=df_date.cases_number.rolling(window=week_window).sum()
```

Another idea would be to instead of looking at cases in one week, we look at the rolling sum


Lets have a  look at the density distribution of the 3-week cumsum years during all weeks and all years except 2017

```python
df_cap50_rolling_sum=comp_abs_consec(df_date,cap=50,cases_col="rolling_sum")
```

```python
df_cap50_sel_year = df_cap50_rolling_sum[df_cap50_rolling_sum.year!=2017]

density_chart = alt.Chart(df_cap50_rolling_sum[df_cap50_rolling_sum.year!=2017]).transform_density(
    'rolling_sum',
    as_=['rolling_sum', 'density'],
).mark_area().encode(
    x=alt.X("rolling_sum:Q",title="cumsum cases 3 weeks",scale=alt.Scale(domain=[0, 60])),
    y='density:Q',
).properties(title="Density distribution for 2012-2016, 2018-2021 of the 3-week cumsum")
line = alt.Chart(pd.DataFrame({'x': [df_cap50_rolling_sum.rolling_sum.mean()+2*df_cap50_rolling_sum.rolling_sum.std()]})).mark_rule(color="red").encode(x='x')
density_chart # + line
```

```python
df_cap50_sel_2012 = df_cap50_rolling_sum[df_cap50_rolling_sum.year<2017]

density_chart_2012 = alt.Chart(df_cap50_sel_2012).transform_density(
    'rolling_sum',
    as_=['rolling_sum', 'density'],
).mark_area().encode(
    x=alt.X("rolling_sum:Q",title="cumsum cases 3 weeks",scale=alt.Scale(domain=[0, 60])),
    y='density:Q',
).properties(title="Density distribution for 2012-2016 of the 3-week cumsum")
density_chart_2012
```

```python
df_cap50_sel_2018 = df_cap50_rolling_sum[df_cap50_rolling_sum.year>2017]

density_chart_2018 = alt.Chart(df_cap50_sel_2018).transform_density(
    'rolling_sum',
    as_=['rolling_sum', 'density'],
).mark_area().encode(
    x=alt.X("rolling_sum:Q",title="cumsum cases 3 weeks",scale=alt.Scale(domain=[0, 60])),
    y='density:Q',
).properties(title="Density distribution for 2018-2021 of the 3-week cumsum")
density_chart_2018
```

We see that for most weeks 2018 - 2021 we have a very small number of cum cases, the peak being around 2. The maximum sum we have seen was 25 cases. However, the distribution is much wider from 2012 to 2016, and the peak is closer to 3.5 or 4.


Next, lets see how the cumsum was in 2017 compared to the other years

```python
 alt.Chart(df_cap50_rolling_sum).mark_line().encode(
    x='week:N',
    y=alt.Y('rolling_sum',title="3-week rolling sum"),
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3",color_seventeen,"#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3"]))
).properties(    
     width=600,
     height=300,
     title="3-week rolling sum in 2012-2021")
```

```python
df_cap50_rolling_sum[(df_cap50_rolling_sum.year==2017)&(df_cap50_rolling_sum.week>=32)]
```

We can see that the cumsum in 2017 was significantly higher. In week 37 it was higher than we ever saw before, with 43 cases. 


Based on this simple analysis, using a threshold of at least 30 or 40 cases during 3 weeks might serve as a good trigger, or even 35. Where exactly the cap should be, is something that should be discussed with the wider team as it is a trade-off between detection and the risk of false alarms. These triggers would have also been triggered 2012 - 2014, but would have only triggered in 2017 from 2015 onward. It may be that the nature of the disease has


When using this method, the trigger would be reached in week 37 in 2017

```python
heatmap_abs_cumsum50 = alt.Chart(df_cap50_rolling_sum).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 50 cases last 3 weeks")),
).properties(
    title=">= 50 cases in last 3 weeks"
)
heatmap_abs_cumsum50
# heatmap_abs_cumsum50.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_cumsum50.png")),
```

#### Urban areas and pneumonic cases
Let's do similar calculations on urban areas, first looking at just triggering when there is a single case, primarily as a reference, and then looking at triggering on absolute cases and rolling sums.

```python
df_urb_cap1=comp_abs_consec(df_date_urb,cap=1,cases_col="cases_number")

heatmap_urb_abs1 = alt.Chart(df_urb_cap1).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 1 cases")),
).properties(
    title=">= 1 pneumonic plague cases in urban areas"
)
# heatmap_urb_abs1.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_urb_abs1.png"))
heatmap_urb_abs1
```

```python
df_urb_cap5=comp_abs_consec(df_date_urb,cap=5,cases_col="cases_number")

heatmap_urb_abs5 = alt.Chart(df_urb_cap5).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 5 cases")),
).properties(
    title=">= 5 pneumonic plague cases in urban areas"
)
# heatmap_urb_abs5.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_urb_abs5.png"))
heatmap_urb_abs5
```

```python
df_date_urb["rolling_sum"]=df_date_urb.cases_number.rolling(window=3).sum()

df_urb_cumsum10=comp_abs_consec(df_date_urb,cap=10,cases_col="rolling_sum")

heatmap_urb_cumsum10 = alt.Chart(df_urb_cumsum10).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 10 cases")),
).properties(
    title=">= 10 pneumonic plague cases in urban areas in last 3 weeks"
)
# heatmap_urb_cumsum10.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_urb_cumsum10.png"))
heatmap_urb_cumsum10
```

```python
 alt.Chart(df_date).mark_line().encode(
    x=alt.X('week:N',title="Week"),
    y=alt.Y('rolling_sum', title = "3-week rolling sum"),
    color=alt.Color('year:N', scale=alt.Scale(range=["#151515","#151515","#151515","#151515","#151515","#FF0000","#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3"]))
)
# cases_data + std_164
```

```python
df_cap10_rolling_sum=comp_abs_consec(df_date,cap=30,cases_col="rolling_sum")
```

```python
heatmap_abs_cumsum30 = alt.Chart(df_cap10_rolling_sum).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 30 cases last 3 weeks")),
).properties(
    title=">= 30 cases in last 3 weeks"
)
heatmap_abs_cumsum30
#heatmap_abs_cumsum30.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_cumsum30.png"))
```

```python
date.fromisocalendar(2017, 36, 1)
```

```python
df_cap50_rolling_sum=comp_abs_consec(df_date,cap=50,cases_col="rolling_sum")
heatmap_abs_cumsum50 = alt.Chart(df_cap50_rolling_sum).mark_rect().encode(
    x="week:N",
    y="year:N",
    color=alt.Color('thresh_reached_str:N',scale=alt.Scale(range=["#D3D3D3",color_twentyone]),legend=alt.Legend(title=">= 50 cases last 3 weeks")),
).properties(
    title=">= 50 cases in last 3 weeks"
)
heatmap_abs_cumsum50
#heatmap_abs_cumsum50.save(os.path.join(plot_dir,f"{iso3}_heatmap_trigger_cumsum50.png"))
```

When using the rolling sum, the cap would be reached one week later in 2017. So that is kind of the same result as requiring 2 consec weeks with more than 10 cases. 

```python
 alt.Chart(df_date).mark_line().encode(
    x='week:N',
    y='rolling_sum',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3"]))
)
# cases_data + std_164
```

```python
cases_data = alt.Chart(df_date[df_date.year.isin([2017])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3"]))
)
cases_data + std_164
```

```python
chart_2018_2021_grey=alt.Chart(df_date[df_date.year.isin([2018,2019,2020,2021])]).mark_line().encode(
    x='week:N',
    y='cases_number',
    color=alt.Color('year:N', scale=alt.Scale(range=["#D3D3D3","#D3D3D3","#D3D3D3","#D3D3D3"]))
)
```

```python
chart_2018_2021_grey + std_164
```

### Plot time series of cases and trigger activations

```python
urb = px.line(
    df_date_urb,
    x="date",
    y="cases_number",
    title="Absolute cases trigger, pneumonic plague in urban areas"
)

urb.add_trace(
    go.Scatter(y=[5],x=[pd.to_datetime("2017-09-11")], marker={"size":10}, showlegend=False),
)

urb.add_annotation(
    y=5,x=pd.to_datetime("2017-09-11"),text="Trigger, 6 cases",font={"size":12}
)

urb.add_hline(y=5,annotation_text="5 cases trigger",annotation_position="top left")

# requires kaleido package on mac, might be different on other machines
# urb.write_image(os.path.join(plot_dir,f"{iso3}_urban_timeline_abs_trigger.png"))
```

```python
urb_cumsum = px.line(
    df_date_urb,
    x="date",
    y="rolling_sum",
    title="3 week sum trigger, pneumonic plague in urban areas"
)

urb_cumsum.add_trace(
    go.Scatter(y=[10],x=[pd.to_datetime("2017-09-11")], marker={"size":10}, showlegend=False),
)

urb_cumsum.add_annotation(
    y=10,x=pd.to_datetime("2017-09-11"),text="Trigger, 10 cases",font={"size":12}
)

urb_cumsum.add_hline(y=10,annotation_text="10 cases trigger",annotation_position="top left")

# requires kaleido package on mac, might be different on other machines
# urb_cumsum.write_image(os.path.join(plot_dir,f"{iso3}_urban_timeline_cumsum_trigger.png"))
```

```python
abs_ov = px.line(
    df_date,
    x="date",
    y="rolling_sum",
    title="3 week sum trigger, pneumonic and bubonic plague"
)

abs_ov.add_trace(
    go.Scatter(y=[53,
                  51,
                  53],
               x=[pd.to_datetime("2012-11-12"),
                  pd.to_datetime("2013-11-11"),
                  pd.to_datetime("2017-09-18")],
               mode="markers",
               marker={"size":10},
               showlegend=False),
)

abs_ov.add_annotation(
    y=43,x=pd.to_datetime("2017-09-18"),text="Trigger, 53 cases",font={"size":12}
)

abs_ov.add_hline(y=50,annotation_text="50 cases trigger",annotation_position="top left")

# requires kaleido package on mac, might be different on other machines
# abs_ov.write_image(os.path.join(plot_dir,f"{iso3}_overall_timeline_cumsum_trigger.png"), width = 1200, height = 300)
```

```python
df_date[df_date.rolling_sum >= 50]
```

```python
std_ov = px.line(
    df_164,
    x="date",
    y="cases_number",
    title="Standard deviation trigger, pneumonic and bubonic plague",
)

std_ov.add_trace(
    go.Scatter(y=[25,4,2],
               x=[pd.to_datetime("2017-09-11"),
                  pd.to_datetime("2012-04-23"),
                  pd.to_datetime("2012-07-23")],
               mode="markers",
               marker={"size":10},
               showlegend=False)
)

std_ov.add_annotation(
    y=25,x=pd.to_datetime("2017-09-11"),text="Trigger, 3rd week above baseline",font={"size":12}
)


std_ov.add_annotation(
    y=2,x=pd.to_datetime("2012-04-23"),text="Triggers",font={"size":12}
)


std_ov.add_trace(
    go.Scatter(y=df_164["plus_1.64std"],x=df_164["date"],marker={"color":"grey"},name="Rolling mean + std. dev.")
)

# std_ov.write_image(os.path.join(plot_dir,f"{iso3}_overall_timeline_std_trigger.png"))
```

```python
df_164[(df_164.cases_number >= df_164["plus_1.64std"]) & (df_164.year == 2012)]
```

### Plot some key graphs for selected data

```python
# key_graphs(df[df.year==2017],title="Cases in 2017")
```

```python
df_164[df_164.cases_number >= 100]
```

### Conclusions
- Not exceptional that there is one week with cases above std (with current data)
- Need several consecutive weeks above std
- It works relatively well if you require 3 consecutive weeks, but also activates in the "off season" twice in 2012 where cumulative cases over that period were extremely low.
- Because of this, could an absolute threshold also be a solution? 


### Questions:
- How do we average current numbers? The historical average is based on a rolling centred sum, we cannot do that with current numbers. Would taking a right rolling sum of 3 weeks (instead of 5) suffice? Though then probably underestimating.. Think if we do averaging of current numbers, should follow some methodology as for the historical average (i.e. have to change historical average method)



### Data quality
Some checks to better understand the data and the different files we received


#### Compare new and old data set


We had an initital dataset and later received an updated one. Compare these two datasets. 
As can be seen in the graph, the cases in 2021 are slightly smoothed in the new dataset, which can be caused by newly available information. 
Numbers for previous years also changed for some dates, and we are thus far unclear why this occurred. 

```python
#df_old=preprocess_plague_data(plague_path_old,list_cases_class=incl_cases_class)
#df_old_date=plague_group_by_date(df_old, sel_end_date="2021-09-28")

# THIS DATA OUTDATED BY NEW DATA PROVIDED IN 2021-10-18 FILE
# bring in 2012 to 2016 data for analysis
plague_path_2012_2016 = plague_dir / "Madagascar_IPM_Plague_cases_Aggregated_historic_2021-10-13.csv"
df_2012_2016 = preprocess_plague_data(plague_path_2012_2016,list_cases_class=incl_cases_class,delimiter=";")
```

```python
dd = plague_group_by_date(df_2012_2016, sel_end_date="2016-12-31")
dd2 = plague_group_by_date(df_2012_2016_v2, sel_end_date="2016-12-31")
df_comb=dd.merge(dd2,on="date",how="outer",suffixes=("_new","_old"))
```

```python
dd2
```

```python
px.line(df_comb,x="date",y=["cases_number_new","cases_number_old"], title="Plague cases reported, 2012-2021")
```

```python
#small attempt to understand the differences between the two datasets
#df_comb_concat=pd.concat([df,df_old]).drop_duplicates(keep=False)
#df_comb_concat[df_comb_concat.date>=df_old.date.min()].sort_values("date")
```
