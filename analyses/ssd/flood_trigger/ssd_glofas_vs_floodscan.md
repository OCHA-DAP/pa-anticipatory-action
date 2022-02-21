<!-- #region -->
### Comparing flood events with historical streamflow

This notebook looks at the correlation between peaks in historical streamflow (from GloFAS reanalysis) and the timing of past flood events identified from floodscan. 

This is done at the Malaka station, as this is the only station in South Sudan. 

The two data sources show partial correspondence in years with the highest peaks, though reaching a maximum precision and recall of 0.5. 

As we can see from the analysis the Floodscan data in Malaka shows a strange pattern. It is therefore a little hard to compare the two datasources here.

However, the peak years of the GloFas data don't fully correspond with the peak years of the county of interest either. Especially surprising is that no large peak has been seen in 2021, while according to all other sources this had the most flooding. It must be said though that at the beginning of 2021 there was already a large fraction flooded. I would nevertheless expect the river discharge to be high, but I am not sure if I understand well-enough how river discharge works (Monica, help;)). 


This notebook is inspired by the Malawi glofas vs floodscan analysis
<!-- #endregion -->

![afbeelding.png](attachment:673c27fd-83fb-4679-aa6d-a7d53b1703ee.png)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
import altair as alt

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.glofas import utils
from src.utils_general.statistics import get_return_periods_dataframe

config = Config()
mpl.rcParams['figure.dpi'] = 300

stations = ['Malakal']
```

```python
iso3="ssd"
```

```python
country_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
floodscan_path=country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm2_stats.csv'
```

Read in the historical GloFAS reanalysis and floodscan data

```python
ds_glofas_reanalysis = utils.get_glofas_reanalysis(
    country_iso3=COUNTRY_ISO3)
```

```python
df_floodscan=pd.read_csv(floodscan_path,parse_dates=['time'])
```

Plot out the historical streamflow against the flooded fraction from Floodscan

```python
start_slice = '1998-01-01'
end_slice = '2021-12-31'

def filter_event_dates(df_event, start, end):
    return df_event[(df_event['time']<str(end)) & (df_event['time']>str(start))].reset_index()

for station in stations: 
        
    fig, axs = plt.subplots(1, 
                            figsize=(10,6 * len(stations.values())), 
                            squeeze=False, sharex=True, sharey=True)
    fig.suptitle(f'Historical streamflow at {station}')
    
    da_plt = ds_glofas_reanalysis[station].sel(time=slice(start_slice, end_slice))
    df_floodscan_adm = filter_event_dates(
        df_floodscan[df_floodscan.ADM2_EN==station],start_slice, end_slice) 

    observations = da_plt.values
    x = da_plt.time

    ax = axs[0, 0]
    ax.plot(da_plt.time, da_plt.values, c='k', lw=0.75, alpha=0.75)
    ax.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
    ax.plot([], [], label="GloFas", color=f'black')
    ax.plot([], [], label="Floodscan", color=f'blue')
    ax2=ax.twinx()
    ax2.plot(df_floodscan_adm.time,df_floodscan_adm.mean_ADM2_PCODE)
    ax2.set_ylabel("Flooded fraction")

#plt rp lines
#         rp_list = [1.5, 2, 5]
#         df_glofas_return_period = utils.get_return_periods(ds_glofas_reanalysis, method='analytical')
#         for i in range(0,len(df_event['start_date'])):
#             ax.axvspan(np.datetime64(df_event['start_date'][i]), np.datetime64(df_event['end_date'][i]), alpha=0.25, color='#3ea7f7')
#         for irp, rp in enumerate(rp_list):
#             ax.axhline(df_return_period.loc[rp, station],  
#                        0, 1, color=f'C{irp+1}', alpha=1, lw=0.75, 
#                        label=f'1 in {str(rp)}-year return period')

    ax.figure.legend()
```

From the graph above we cannot immediately see a clear correspondence between the two datasources. However, we can see that the Floodscan data shows a strange pattern. Where it is almost always zero except for very sharp peaks. To me this is a strange pattern as I would expect flooding to appear and disappear more gradually. 

Despite that, we can see that the peak of the floodscan always comes before the peak of the discharge values, which is also the opposite of what I would expect. 


Next we compute the years during which the highest peaks were observed for Glofas and Floodscan. With the goal to see how well they correspond. 

```python
def compute_peak_rp(df,val_col):
    df_rolling=df.sort_values('time').set_index('time').groupby(
    'ADM2_EN',as_index=False)[val_col].rolling(
    10,min_periods=10).mean().reset_index().rename(
    columns={val_col:'mean_rolling'})
    df=df.merge(df_rolling,on=['time','ADM2_EN'])
    df['year']=df.time.dt.year
    #get one row per adm2-year combination that saw the highest mean value
    df_peak=df.sort_values('mean_rolling', ascending=False).drop_duplicates(['year','ADM2_EN'])
    years = [3,5]#np.arange(1.5, 6, 0.5)
    df_rp=df_peak[df_peak.ADM2_EN.isin(stations)].copy()
    for adm in stations:
        df_adm=df_peak[df_peak.ADM2_EN==adm].copy()
        df_rps_ana=get_return_periods_dataframe(df_adm, rp_var="mean_rolling",years=years,
                                                method="analytical",round_rp=False)
        df_rps_emp=get_return_periods_dataframe(df_adm, rp_var="mean_rolling",years=years,
                                                method="empirical",round_rp=False)
        for y in years:
            df_rp.loc[df_rp.ADM2_EN==adm,f'rp{y}']=np.where(
            (df_adm.mean_rolling>=df_rps_emp.loc[y,'rp']),1,0)
    return df_rp
```

```python
df_glofas=ds_glofas_reanalysis.to_dataframe()
df_glofas_stations=df_glofas[stations].reset_index().melt(id_vars=["time"], 
        var_name="ADM2_EN", 
        value_name="river_discharge")
```

```python
#compute years that reached 1 in 3 and 1 in 5 year return period peak
df_glofas_rp=compute_peak_rp(
    df_glofas_stations[df_glofas_stations.time.dt.year>=df_floodscan.time.dt.year.min()],'river_discharge')
```

```python
df_floodscan_rp=compute_peak_rp(df_floodscan,'mean_ADM2_PCODE')
```

```python
#combine the data
df_comb=df_glofas_rp.merge(df_floodscan_rp[['time','ADM2_EN','mean_rolling','year','rp3','rp5']],
                   on=['year','ADM2_EN'],suffixes=("_glofas","_floodscan"))
```

```python
#transform for plotting
df_comb_long=pd.melt(df_comb, id_vars=['year','ADM2_EN'], 
                      value_vars=['rp3_floodscan','rp5_floodscan','rp3_glofas','rp5_glofas'])
```

```python
#show 1 in 3 year return period years for both data sources
alt.Chart(df_comb_long[df_comb_long.variable.isin(
    ['rp3_floodscan','rp3_glofas'])]).mark_rect().encode(
    x="year:N",
    y=alt.Y("variable:N"),
    color=alt.Color('value:N',scale=alt.Scale(range=["#D3D3D3",'red']),
                    legend=alt.Legend(title="1 in 3 year rp"),
                   ),
).properties(
    title="1 in 3 year return period years"
)
```

```python
#show 1 in 3 year return period years for both data sources
alt.Chart(df_comb_long[df_comb_long.variable.isin(
    ['rp5_floodscan','rp5_glofas'])]).mark_rect().encode(
    x="year:N",
    y=alt.Y("variable:N"),
    color=alt.Color('value:N',scale=alt.Scale(range=["#D3D3D3",'red']),
                    legend=alt.Legend(title="1 in 5 year rp"),
                   ),
).properties(
    title="1 in 5 year return period years"
)
```

From the graphs above we can see confirmed what we saw in the timeseries graph. Namely, that the two datasources don't correspond very well. 
However, given the strange patttern in the floodscan data it is questionable whether this is a good comparison dataset. 

What is worrying nevertheless, is that we don't see a big peak in the 2021 glofas data. Also other years, such as 2014 and 2017 no peak discharge was observed whereas these were years that also saw flooding in the area around the Nile, i.e. south-east of Malakal. 
