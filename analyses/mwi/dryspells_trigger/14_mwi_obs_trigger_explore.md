## Observational trigger frequency

```python
from pathlib import Path
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import dataframe_image as dfi

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
DAILY_RAIN = 'mwi_arc2_precip_long_raw.csv'
PLOT_DIR = DATA_DIR / 'public' / 'processed' / COUNTRY_ISO3 / 'plots' / 'dry_spells' / 'arc2'

RAINY_SEASON = ARC2_DIR / 'mwi_arc2_rainy_seasons.csv'

N_YEARS = 21 # Dry spells based on 21 years of ARC2 data - from 2000-2020, inclusive
```

Read in the dry spells data and convert data types.

```python
df = pd.read_csv(ARC2_DIR / DRY_SPELLS)
daily_df = pd.read_csv(ARC2_DIR / DAILY_RAIN, parse_dates = ['date'])
df['dry_spell_confirmation'] = pd.to_datetime(df['dry_spell_confirmation'])
```

Keep only dry spells that were confirmed within the monitoring period of Jan 1 - March 7. This means that the dry spells could begin as early as Dec 18th and late as February 21.

```python
# The year doesn't matter here
monitoring_start = "2014-01-01"
monitoring_end = "2014-03-14" 

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

df_activations_save = df_activations.rename(columns={'dry_spell_confirmation': 'Confirmation date', 'n_adm2': 'Number ADM2', 'year': 'Year'})
dfi.export(df_activations_save, 'df_activations.png')
```

```python
df_activations_save
```

Let's examine what these additional activations in late February (confirmed in March) would look like.

```python
df_filtered[df_filtered.no_year.str.startswith("03")]
```

I guess the concern here is that we might be picking up dry spells that overlap with the dry season (as defined in ARC2). This is pretty clear on the long duration dry spells, but let's see how it looks elsewhere. By definition it must start on the 15th of March at the earliest, so we can see pretty clearly those that can't overlap with the dry season.

```python
daily_df.sort_values(by = ['ADM2_PCODE', 'date'], axis = 0, inplace = True, ignore_index = True)

# rolling backwards

daily_df['rolling_rainfall'] = daily_df \
    .groupby('ADM2_PCODE')['mean_cell'] \
    .apply(lambda x: x.rolling(window=15).sum().shift(-14)) \
    .reset_index(drop = True)

# capture rainy season groups, all code here piecemeal for checking
daily_df['rain_less_25'] = daily_df['rolling_rainfall'] <= 25
daily_df['year'] = daily_df.date.dt.year

# get all dry spells

rainy_df = daily_df[(daily_df.rain_less_25) & ((daily_df.date.dt.month >= 4) | ((daily_df.date.dt.month == 3) & (daily_df.date.dt.day >= 29)))] \
    .groupby(['ADM2_PCODE', 'year']) \
    .agg(
      pcode = ('ADM2_PCODE','unique'),
      year = ('year','unique'),
      dry_season_confirmation = ('date','min')
     ) \
    .reset_index(drop = True) \
    .assign(pcode = lambda x: x.pcode.str[0],
            year = lambda x: x.year.str[0],
            season = lambda x: (x.year-1).astype('str') + "-" + x.year.astype('str'),
            dry_season_first_date = lambda x: x.dry_season_confirmation - pd.to_timedelta(14, unit = 'd'))

rainy_df
```

Let's merge this back into the dry spell data to see how many of the dry spells confirmed in March would be overlapping with the dry season.

```python
df_filtered['year'] = df_filtered.dry_spell_confirmation.dt.year
new_ds = pd.merge(df_filtered,
                  rainy_df,
                  how = 'left',
                  on=['pcode', 'year'])


new_ds['overlap_dry_season'] = pd.to_datetime(new_ds.dry_spell_last_date) >= new_ds.dry_season_first_date
new_ds = new_ds[['pcode', 'dry_spell_confirmation', 'dry_spell_first_date', 'dry_spell_last_date', 'dry_spell_duration', 'overlap_dry_season', 'dry_season_first_date', 'no_year']]

new_ds[new_ds.no_year.str.startswith("03")]
```

From above, we can see that many of these dry spell activations into March overlap with the beginning of the dry season, as it would be defined using ARC2. Often, this occurs when the confirmation occurs past the 7th of March, so that more than half of the dry spell is taking place within March. This is the case in 2005, where all dry spells overlap with the dry season onset and in 2020, where 5 of 9 detected dry spells overlap with the onset of the dry season. The other 4 did not overlap with the dry season, although there was only a few day gap between the cessation of the dry spell and dry season onset.

The decision here really comes down to how we view these dry spells activated in 2020. Even ignoring dry season overlap, for all other years, there would be no change in activation based on these additional dry spells (there would be only 1 detected in 2002 and the trigger would've already been hit in 2005 and 2011).

So, how do we view the 2020 dry spells? Some general thoughts are:

- If we set a limit that at least half of the dry spell must occur in February (detect on or before March 7th), based on not wanting to overlap with the dry season onset, we would not have any additional activations based on our `>=3` trigger. We would still capture what appear to be legitimate dry spells in the rainy season in 2002 and 2011, while only capturing 2 dry spells in 2000. So it comes down to if we would have been okay missing the 2020 dry spells.

- For 2020, could we be conflating an early onset dry season with dry spells during the rainy season? I have seen this referenced in the literature as an issue (shortened rainy seasons), but is this something we want to capture with this trigger? The March 15th reference for rainy season cessation comes for a paper nearly 20 years old now, so it might be out of date regarding the seasonality in Malawi? It looks fairly clear that in 2005, for instance, the dry spells detected at the tail end of the period (only 1 day following in February) are simply capturing the dry season onset.

- If this is simply capturing a shortened rainy season, we could consider for V2 looking into capturing when the rainy season begins and the possibility of the cessation of the rainy season early. For instance, continuing to monitor for dry spells following March 7th and if we detect a situation like in 2020, triggering something based on the detection of a shortened rainy season (depending on its onset that year). For this round, we just want to be comfortable NOT activating in 2020 even though there were 9 separate ADM2 areas with a dry spell detected.

My intuition is that we are okay not activating the trigger for 2020. Cursory searches for documents/news about Malawi dry spells in 2020 have produced no collaborating documentation on high impact dry spells in that year. While this is obviously quite ancedotal, it's quite easy to found references to dry spells and their potential impacts even in the year they are observed, such as 2005 or 2010. In fact, documentation from the time, such as this [April 2020 FEWS-NET report](https://reliefweb.int/sites/reliefweb.int/files/resources/MW_FSOU_April_2020_Final.pdf) references above average seasonal rainfall and subsequent crop production.

I therefore would recommend that we DO include dry spells confirmed in March, but only monitoring until March 7th. Maintaining the `>=3` ADM2 trigger would keep the same return period and observed years where we would meet the action trigger, while also capturing some late onset dry spells without significant overlap into the dry season (only one instance in our observed data from 2020).

### Rainy season onset

```python
daily_df.sort_values(by = ['ADM2_PCODE', 'date'], axis = 0, inplace = True, ignore_index = True)

# rolling forwards

daily_df['rainfall_40mm_10d_prior'] = daily_df \
    .groupby('ADM2_PCODE')['mean_cell'] \
    .apply(lambda x: x.rolling(window=10).sum() >= 40) \
    .reset_index(drop = True)

# rolling backwards

daily_df['rainfall_2mm_10d_ahead'] = daily_df \
    .groupby('ADM2_PCODE')['mean_cell'] \
    .apply(lambda x: x.rolling(window=10).sum().shift(-9) < 2) \
    .reset_index(drop = True)

# no 10 consecutive in 30

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
            rainy_season_onset = lambda x: x.date - pd.DateOffset(days=9)) \
    .rename(columns = {'ADM2_PCODE':'pcode'}) \
    .drop('date', axis = 1)
            
rainy_onset_df
```

Merge the dry season onset and rainy season data frames for saving.

```python
arc2_rs_df = pd.merge(rainy_onset_df, rainy_df[['pcode', 'season', 'dry_season_first_date']], on = ['pcode', 'season'], how = 'left')
arc2_rs_df.to_csv(RAINY_SEASON, index=False)
```

Okay, now we can merge back in with old dataset to compare with our triggers.

```python
new_onset_ds = pd.merge(df_filtered,
                        rainy_onset_df,
                        how = 'left',
                        on=['pcode', 'year'])


new_onset_ds['overlap_dry_season'] = pd.to_datetime(new_onset_ds.dry_spell_first_date) <= new_onset_ds.rainy_season_onset
new_onset_ds = new_onset_ds[['pcode', 'dry_spell_confirmation', 'dry_spell_first_date', 'dry_spell_last_date', 'dry_spell_duration', 'overlap_dry_season', 'rainy_season_onset', 'no_year']]

new_onset_ds[new_onset_ds.overlap_dry_season]
```

## Output statistics

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
