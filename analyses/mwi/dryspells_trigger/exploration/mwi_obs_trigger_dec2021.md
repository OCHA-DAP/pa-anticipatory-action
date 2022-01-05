### Inspect rainfall December 2021
There were signs that there were dry spells in December 2021. While according to the framework we only start monitoring form 25th of December, we do a short analysis to understand what happened in December 2021 for future learning. 

```python
%load_ext autoreload
%autoreload 2

import os
from pathlib import Path
import sys
from datetime import date
import pandas as pd
import altair as alt
from datetime import date

path_mod = f"{Path.cwd().parents[3]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells

import geopandas as gpd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
```

We will look at the rainfall and dry spell data during December 2021. For this analysis we use the centroid method. 
We look at the data over all of the December month, as well as from the start of the period we are monitoring which is the 25th of December. 

```python
## Global variables for all monitoring

POLY_PATH = Path(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'cod_ab',
    'mwi_drought_adm2.gpkg'
)


MONITORING_END = date.today()
RANGE_X = ("32E", "36E")
RANGE_Y = ("20S", "5S")

# centroid method
arc2_centr_dec = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = "2021-12-01",
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y
)

arc2_centr_mon = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = "2021-12-25",
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y
)

#arc2_centr_dec.download()
#arc2_centr_dec.aggregate_data()
#arc2_centr_dec.calculate_rolling_sum()
#df_ds=arc2_centr_dec.identify_dry_spells()
```

We compute the rainy season onset. Several definitions exist but we use "the occurrence of at least 40 mm of rainfall accumulated within 10 days after 1st November and not to be followed by a dry spell of 10 or more consecutive days within one month." [source](https://tjet.udsm.ac.tz/index.php/tjet/article/view/418)

```python
daily_df=arc2_centr_dec.load_aggregated_data()
daily_df=daily_df.rename(columns={"T":"date"})
```

```python
gdf_adm2_south=gpd.read_file(POLY_PATH)
daily_df=daily_df.merge(gdf_adm2_south[["ADM2_PCODE","ADM1_EN"]],on="ADM2_PCODE",how="left")
```

```python
daily_df.sort_values(by = ['ADM2_PCODE', 'date'], axis = 0, inplace = True, ignore_index = True)

# rolling forwards

#last 10 days at least 40mm
daily_df['rainfall_40mm_10d_prior'] = daily_df \
    .groupby('ADM2_PCODE')['mean_ADM2_PCODE'] \
    .apply(lambda x: x.rolling(window=10).sum() >= 40) \
    .reset_index(drop = True)

#not perse needed, but for exploratory analysis
daily_df['rainfall_40mm_10d_prior_abs'] = daily_df \
    .groupby('ADM2_PCODE')['mean_ADM2_PCODE'] \
    .apply(lambda x: round(x.rolling(window=10).sum(),2)) \
    .reset_index(drop = True)

# rolling backwards

#next 10 days less than 2mm
daily_df['rainfall_2mm_10d_ahead'] = daily_df \
    .groupby('ADM2_PCODE')['mean_ADM2_PCODE'] \
    .apply(lambda x: x.rolling(window=10).sum().shift(-9) < 2) \
    .reset_index(drop = True)

# no 10 consecutive less than 2mm in next 30 days
daily_df['no_10_consec'] = daily_df \
    .groupby('ADM2_PCODE')['rainfall_2mm_10d_ahead'] \
    .apply(lambda x: x.rolling(window=21).sum().shift(-20).shift(-1) == 0) \
    .reset_index(drop = True)

# get rainy season onset

rainy_onset_df = daily_df[daily_df.rainfall_40mm_10d_prior & daily_df.no_10_consec]
rainy_onset_df = rainy_onset_df[(rainy_onset_df.date.dt.month == 12) | (rainy_onset_df.date.dt.month <= 2) | ((rainy_onset_df.date.dt.month == 11) & (rainy_onset_df.date.dt.day >= 10))]
rainy_onset_df['season'] = np.where(rainy_onset_df.date.dt.month >= 11,
                                    rainy_onset_df.date.dt.strftime('%Y') + "-" + (rainy_onset_df.date.dt.year + 1).map(str),
                                    (rainy_onset_df.date.dt.year - 1).map(str) + "-" + rainy_onset_df.date.dt.year.map(str))

rainy_onset_df.reset_index(drop=True,inplace=True)

rainy_onset_df = rainy_onset_df \
    .loc[rainy_onset_df.groupby(['ADM2_PCODE', 'season'])['date'].idxmin()] \
    .reset_index(drop=True) \
    [['ADM2_PCODE', 'date', 'season']] \
    .assign(year = lambda x: x.season.str.split('-').str[1].map(int),
            #substract 9 days for onset as should be first day of the 10 day period with >=40mm
            rainy_season_onset = lambda x: x.date - pd.DateOffset(days=9)) \
    .rename(columns = {'ADM2_PCODE':'pcode'}) \
    .drop('date', axis = 1)
```

Only in 2 adm2's the rainy season has started according to the definition by 02-01-2022.     

It should be noted here that this means that by 02-01-2022 the start of the rainy season hadn't been confirmed yet. This thus can mean that the rainy season might have already started in more admins, but given the definition we can only confirm this 40 days after the occurrence. 

```python
rainy_onset_df[rainy_onset_df.year==2022]
```

The requirement of 30 days without 10 day long dry spells is quite strict. We therefore also check if and when the first condition of the onset, namely 40 mm in 10 days has been met for the different admins.    
We can see this has been met for all admins. This occured already quite early in November as the dates shown are the last of the 10 day period. 

```python
daily_df[(daily_df.date>="2021-11-01")&(daily_df.rainfall_40mm_10d_prior)].groupby("ADM2_PCODE").first()
```

We plot the distribution of rainfall to understand slightly better why the rainy season hasn't started for most admins. As can be seen from the graph many admin2's experienced a 10 day dry spell between 3 Dec and 12 Dec. According to our definitions this breaks the start of the rainy season. 

Moreover, from this graph we can see that many admins got very little rain between the 25th and 30th of Dec, but all received rain thereafter. 

```python
daily_df_south=daily_df[daily_df.ADM1_EN=="Southern"]
daily_df_south_sel=daily_df_south[daily_df_south.date>="2021-11-01"]
```

```python
chart_rain_2021=alt.Chart(daily_df_south_sel).mark_area().encode(
    x='date:T',
    y=alt.Y('mean_ADM2_PCODE',title='Daily rainfall')
).properties(
    width=200,#600,
    height=100,
    title="Rainfall from 01-11 2021 till Jan 2022"
).facet(
    facet='ADM2_PCODE:N',
    columns=3
)
chart_rain_2021
```

Nevertheless, we can see what the longest runs of consecutive days with less than 2mm were. 
When looking at all of december, we  can see that many admin2's experienced a dry spell of around 12 days, with MW301 even reaching 21 days. All these long dry spells took place around the beginning to middle of December. 

```python
df_dec_longest=arc2_centr_dec.find_longest_runs()
df_dec_longest[df_dec_longest.index.str.contains("MW3")]
```

When we only analyze the dates within our monitoring period, we can see that the longest observed dry spell was 8 days and that all others observed a 4 or 5 day long dry spell. The longest observed dryspell was in "MW301" which is the most northern admin2 area of the Southern admin1 area. 

```python
df_mon_longest=arc2_centr_mon.find_longest_runs()
df_mon_longest[df_mon_longest.index.str.contains("MW3")]
```

We inspect the cumulative rainfall from the 25th till the 30th, which is the time the shorter dry spells occurred and a notice was made by the prime minister. From this data we can see that the northern part of the Southern ADMIN1 received barely any rain. The South of the Southern ADMIN1 did receive a substantial amount of rain. 

```python
date_min=date(2021,12,25)
date_max=date(2021,12,30)
da_cum = arc2_centr_dec.cumulative_rainfall(date_min=date_min,date_max=date_max)
da_cum = da_cum.rio.clip(gdf_adm2_south.geometry)
da_cum = da_cum.where(da_cum.values >= 0, np.NaN)
da_cum=da_cum.drop("spatial_ref")
g=da_cum.plot(
    cmap='Blues',#'Greys',
    cbar_kwargs={"label":"Cumulative rainfall (mm)"},
    figsize=(6,10)
    )
gdf_adm2_south.plot(ax=g.axes, facecolor="none", alpha=0.5)
plt.title(f"Cumulative rainfall from {date_min} to {date_max}");
```

Lastly we inspect if the situation changed during the last days. From the plot below we can see that between 31 Dec and 2 Jan all of the Southern region received rain, thus ending the dry spells

```python
date_min=date(year=2021,month=12,day=31)
date_max=date(year=2022,month=1,day=2)
da_cum = arc2_centr_dec.cumulative_rainfall(date_min=date_min,date_max=date_max)
da_cum = da_cum.rio.clip(gdf_adm2_south.geometry)
da_cum = da_cum.where(da_cum.values >= 0, np.NaN)
da_cum=da_cum.drop("spatial_ref")
g=da_cum.plot(
    cmap='Blues',#'Greys',
    cbar_kwargs={"label":"Cumulative rainfall (mm)"},
    figsize=(6,10)
    )
gdf_adm2_south.plot(ax=g.axes, facecolor="none", alpha=0.5)
plt.title(f"Cumulative rainfall from {date_min} to {date_max}");
```


And for the email to the Malawi team, here's a faceted graph of cumulative rainfall across the administrative areas that clearly highlight the spread of rainfall in the latter parts of December and early January relative to early to mid December.

```python
# save path for figure and table
save_path = Path(
    os.getenv('AA_DATA_DIR'),
    'public',
    'processed',
    'mwi',
    'dry_spells',
    'v1'
)

def cum_plot_data(date_min, date_max):
    da_cum = arc2_centr_dec.cumulative_rainfall(date_min=date_min,date_max=date_max)
    da_cum = da_cum.rio.clip(gdf_adm2_south.geometry)
    da_cum = da_cum.where(da_cum.values >= 0, np.NaN)
    da_cum=da_cum.drop("spatial_ref")
    return da_cum


dates_list = [[date(2021,12,1),date(2021,12,7)],
              [date(2021,12,8),date(2021,12,14)],
              [date(2021,12,14),date(2021,12,21)],
              [date(2021,12,22),date(2021,12,28)],
              [date(2021,12,29),date(2022,1,4)]]
             
f,ax = plt.subplots(2, 3, figsize=(10,15))

for i in range(5):
    da_cum = cum_plot_data(dates_list[i][0], dates_list[i][1])
    da_cum.plot(
        cmap='Blues',
        vmin=0,
        vmax=120,
        cbar_kwargs={"label":"Cumulative rainfall (mm)"},
        ax=ax[i // 3, i % 3]
    )
    gdf_adm2_south.plot(ax=ax[i // 3, i % 3], facecolor="none", alpha=0.5)
    ax[i // 3, i % 3].title.set_text(f"{dates_list[i][0].strftime('%b %d')} - {dates_list[i][1].strftime('%b %d')}")

ax[1,2].set_visible(False)
f.suptitle(f"Cumulative rainfall from December 1 until January 4", size=20)
# f.savefig(save_path / "mwi_december_cum_rainfall.png")
```

Lastly, let's look at the number of dry days and lengths across December and January.

```python
# December to January 2022 dry spells
arc2_centr_check = DrySpells(
    country_iso3 = "mwi",
    polygon_path = POLY_PATH,
    bound_col = "ADM2_PCODE",
    monitoring_start = "2021-12-14",
    monitoring_end = MONITORING_END,
    range_x = RANGE_X,
    range_y = RANGE_Y
)

rainy_days = arc2_centr_check.count_rainy_days()
dry_days = arc2_centr_check.count_dry_days()
# days under 4mm
df_agg = arc2_centr_check.load_aggregated_data(True)
dry_days_4mm = df_agg.groupby("ADM2_PCODE").agg(
            dry_days_4mm=(
                "mean_ADM2_PCODE",
                lambda x: sum(x < 4),
            )
        )

long_run = arc2_centr_check.find_longest_runs()
admin_en_df = gdf_adm2_south[["ADM2_EN", "ADM2_PCODE"]].set_index("ADM2_PCODE")

df = pd.merge(rainy_days, dry_days_4mm, left_index=True, right_index=True) \
    .merge(dry_days, left_index=True, right_index=True) \
    .merge(long_run, left_index=True, right_index=True) \
    .merge(admin_en_df, left_index=True, right_index=True) \
    .reset_index(drop=True)
df = df[df.columns[::-1]]
df.columns = ["admin_area",
              "cumulative_days_under_2mm",
              "days_without_rainfall",
              "days_with_rainfall_lt_4mm",
              "days_with_rainfall_gte_4mm"]
df.to_csv(save_path / "mwi_december_2021_table.csv", index=False)
```
