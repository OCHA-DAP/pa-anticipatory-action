## Observational trigger frequency

```python
from pathlib import Path
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt

mpl.rcParams['figure.dpi'] = 300
```

Set config values and parameters.

```python
path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

config = Config()
parameters = config.parameters('mwi')
COUNTRY_ISO3 = parameters["iso3_code"]
DATA_DIR = Path(config.DATA_DIR)

RAW_DIR =  DATA_DIR / config.PUBLIC_DIR / config.RAW_DIR / COUNTRY_ISO3
ARC2_DIR = DATA_DIR / config.PUBLIC_DIR / 'exploration' / COUNTRY_ISO3 / 'arc2'
DRY_SPELLS = 'mwi_arc2_centroid_dry_spells.csv'
PLOT_DIR = DATA_DIR / 'public' / 'processed' / COUNTRY_ISO3 / 'plots' / 'dry_spells' / 'arc2'

N_YEARS = 21 # Dry spells based on 21 years of ARC2 data - from 2000-2020, inclusive
```

Read in the dry spells data and convert data types.

```python
df = pd.read_csv(ARC2_DIR / DRY_SPELLS)
df['dry_spell_confirmation'] = pd.to_datetime(df['dry_spell_confirmation'])
```

Keep only dry spells that were confirmed within the monitoring period of Jan 1 - Feb 28. This means that the dry spells could begin as early as Dec 18th.

```python
# The year doesn't matter here
monitoring_start = "2014-01-01"
monitoring_end = "2014-02-28" 

monitoring = pd.Series(pd.date_range(monitoring_start, monitoring_end))
monitoring_no_year = monitoring.map(lambda x: x.strftime("%m-%d"))
df["no_year"] = df['dry_spell_confirmation'].map(lambda x: x.strftime("%m-%d"))
no_year_mask = df['no_year'].isin(monitoring_no_year)
df_filtered = df[no_year_mask].sort_values(by='dry_spell_confirmation')
```

Also only keep the dry spells that are in the Southern region (MW3).

```python
df_filtered = df_filtered[df_filtered['pcode'].str.startswith('MW3')]
```

Look into various scenarios for activating, based on different thresholds for number of regions experiencing a dry spell.

```python
df_activations = df_filtered.groupby('dry_spell_confirmation').size().reset_index(name='n_adm2')
df_activations['year'] = df_activations['dry_spell_confirmation'].dt.year
```

```python
df_activations
```

Output statistics include: 
- `thresh`: number of admin regions with a dry spell confirmed on a given date
- `tot_act`: total number of times the trigger would be met, *potentially in the same season*
- `tot_years`: total number of times the trigger would be met, *assuming it can only happen once / season*
- `freq_act`: `tot_act` / total number of monitoring years
- `freq_years`: `tot_years` / total number of monitoring yeats
- `mult_cases`: number of years that have multiple cases where the trigger would be met

```python
def calc_stats(df_activations, threshs = range(1,6)):
    
    df_stats = pd.DataFrame(columns = ['thresh', 'tot_act', 'tot_years', 'freq_act', 'freq_years', 'mult_cases'])
    
    for thresh in threshs: 

        df_activations_filter = df_activations[df_activations['n_adm2'] >= thresh]

        tot_act = len(df_activations_filter.index)
        tot_years = len(df_activations_filter.year.unique())
        freq_act = tot_act / N_YEARS
        freq_years = tot_years / N_YEARS

        count_mult = df_activations_filter.groupby('year').size().reset_index(name='n_mult_years')
        mult_cases = len(count_mult[count_mult['n_mult_years'] > 1])

        row = {
            'thresh': thresh, 
            'tot_act': tot_act, 
            'tot_years': tot_years, 
            'freq_act': freq_act, 
            'freq_years': freq_years, 
            'mult_cases': mult_cases
        }

        df_stats = df_stats.append(row, ignore_index=True)
    return df_stats
```

```python
df_stats_single = calc_stats(df_activations)
```

Now explore the results if we add a buffer around when we consider different admin regions to be simultaneously triggering. For example, many of the dry spells are confirmed within several days of each other, but they aren't counted as happening together with the above method. If we identify dry spells occurring in multiple admin regions within a matter of days of each other, we should probably count these together.

```python
buffer = 7 # Days of buffer to consider dry spells co-occurring

df_buf = df_activations.copy()
df_buf['match'] = 0

count = 1

for index, row in df_buf.iterrows():
    
    try: 
    
        d1 = df_buf.loc[index, 'dry_spell_confirmation']  
        d2 = df_buf.loc[index + 1, 'dry_spell_confirmation']
        
        df_buf.loc[index, 'match'] = count

        if (d1 + dt.timedelta(days = buffer)) >= d2:           
            df_buf.loc[index+1, 'match'] = count
            
        else:           
            count += 1
    
    except KeyError as e:
        print(f'KeyError at index {e}')
        
df_activations_buf = df_buf.groupby(['match'])['n_adm2'].sum().reset_index(name='n_adm2')
df_activations_buf = df_activations_buf.merge(df_buf[['match', 'year']], on='match', how='left').drop_duplicates()
```

Calculate the same statistics as above.

```python
df_stats_cumulative = calc_stats(df_activations_buf)
```

Make plots of all results. Compare across various thresholds and between the single vs cumulative method of dry spell detection.

```python
df_stats_single
```

```python
df_stats_cumulative
```

```python
plt.plot(df_stats_single['thresh'], df_stats_single['freq_act'], label='Single: Trigger mult / season')
plt.plot(df_stats_cumulative['thresh'], df_stats_cumulative['freq_act'], label='Cumulative: Trigger mult / season')
plt.legend()
plt.ylabel('Frequency of occurrence')
plt.xlabel('Number of admin regions with dry spell')
plt.show()
```
