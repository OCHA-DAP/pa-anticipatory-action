### Evaluating the forecast skill of GloFAS in Bangladesh

This notebook is to compare the forecast skill of GloFAS for various lead times. We are comparing the reforecast product (lead time 5-30 days) against the reanalysis product. This is an improvement on the ```process-glofas``` notebook and takes the processed GloFAS data created by ```get_glofas_data.py```. 

We're specifically interested in the forecast skill during times of potential flooding, here estimated to be between June - Oct. 

```python
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib as mpl
import xskillscore as xs

import read_in_data as rd
reload(rd)

mpl.rcParams['figure.dpi'] = 300
```

### Read in forecast and reanalysis
The data is interpolated to a daily granularity and shifted according to the forecast lead time. 

```python
da_glofas_reanalysis = rd.get_glofas_reanalysis()
da_glofas_forecast = rd.get_glofas_forecast()
da_glofas_reforecast = rd.get_glofas_reforecast()

```

Let's take a sample of some of the data to check that it all looks like we would expect. 

```python
# Slice time and get mean of ensemble members for simple plotting
start = '2000-06-01'
end = '2000-10-31'
rf_list_slice = da_glofas_reforecast.sel(time=slice(start,end))
ra_slice = da_glofas_reanalysis.sel(time=slice(start, end))

rf_list_slice.mean(axis=1).plot.line(x='time', add_legend=True)
ra_slice.plot.line(label='Historical', c='k')
#plt.legend()
plt.show()
```

#### Compute the measure(s) of forecast skill

We'll compute forecast skill using the ```xskillscore``` library and focus on the CRPS (continuous ranked probability score) value, which is similar to the mean absolute error but for probabilistic forecasts. This is also what GloFAS uses in evaluating their own forecast skill.

```python
# Need to keep going from here
```

```python
df_crps = pd.DataFrame(columns=['leadtime', 'crps', 'year'])
years = list(range(1999,2019))

for year in years:
    for lt in rf_list:
        start = str(year) + '-06-01'
        end = str(year) + '-10-30'
        rf_slice = rf_list[lt].sel(time=slice(start, end))
        ra_slice = ra.sel(time=slice(start, end))
        crps_ensemble = xs.crps_ensemble(ra_slice, rf_slice)
        df_crps = df_crps.append([{'leadtime':lt, 'crps':crps_ensemble.values.round(2), 'year': year}], ignore_index=True)

# Save results disaggregated by year
df_crps.to_csv(os.path.join(GLOFAS_DIR, f'{STATION}_crps.csv'))
```

```python
# Quick plot to summarize the results across all years
known_skill = SKILL_DIR / 'forecast_skill.csv'
df_skill = pd.read_csv(known_skill, header=None)

df_crps_avg = df_crps.groupby('leadtime').mean().reset_index()

plt.plot(df_crps_avg.leadtime/24, df_crps_avg.crps)
plt.plot(df_skill[0], df_skill[1])
plt.title("GloFAS forecast skill at Bahadurabad:\n 1999-2018 reforecast, June-Oct average")
plt.xlabel("Lead time (days)")
plt.ylabel("CRPS")
plt.show()
```

### Bias: rank histogram

```python

```
