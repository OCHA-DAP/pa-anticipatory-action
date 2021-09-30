## MWI processing for HDX

Code to process MWI ARC2 rainy season and dry spell data frames for upload to HDX.

```python
import pandas as pd
import os
import fuzzyjoin
import numpy as np

data_dir = os.environ["AA_DATA_DIR"]

rainy_df_path = os.path.join(data_dir, 'public', 'exploration', 'mwi', 'arc2', 'mwi_arc2_rainy_seasons.csv') 
rainy_df = pd.read_csv(rainy_df_path)

dry_spell_path = os.path.join(data_dir, 'public', 'exploration', 'mwi', 'arc2', 'mwi_arc2_centroid_dry_spells.csv')
dry_spell_df = pd.read_csv(dry_spell_path, index_col=0, parse_dates=['dry_spell_last_date', 'dry_spell_first_date', 'dry_spell_confirmation'])

daily_path = os.path.join(data_dir, 'public', 'exploration', 'mwi', 'arc2', 'mwi_arc2_precip_long_raw.csv')
daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=['date'])
daily_df = daily_df[['ADM2_PCODE', 'date', 'mean_cell']]

adm_path = os.path.join(data_dir, 'public', 'raw', 'mwi', 'cod_ab', 'mwi_district_lookup.xls')
adm_df = pd.read_excel(adm_path)

# Saving file paths

RAINY_SEASON = os.path.join(data_dir, 'public', 'processed', 'mwi', 'arc2', 'mwi_arc2_rainy_seasons_2000_2020.csv')
DRY_SPELLS = os.path.join(data_dir, 'public', 'processed', 'mwi', 'arc2', 'mwi_arc2_dry_spells_2000_2020.csv')
```

### Rainy season processing

First, let's process the rainy season data.

```python
rainy_merged_df = pd.merge(rainy_df, adm_df[['ADM2_PCODE', 'ADM2_32EN', 'ADM1_EN']], left_on = 'pcode', right_on = 'ADM2_PCODE') \
    .rename(columns={'ADM2_32EN':'ADM2_EN',
                     'season':'season_approx',
                     'ADM1_EN':'region'})
                     
rainy_merged_df['onset_date'] = pd.to_datetime(rainy_merged_df.rainy_season_onset)    
rainy_merged_df['cessation_date'] = pd.to_datetime(rainy_merged_df.dry_season_first_date) - pd.to_timedelta(1,unit='d')
rainy_merged_df['onset_month'] = rainy_merged_df.onset_date.dt.month
rainy_merged_df['cessation_month'] = rainy_merged_df.cessation_date.dt.month
rainy_merged_df['rainy_season_duration'] = (rainy_merged_df.cessation_date - rainy_merged_df.onset_date).dt.days

# filter rows for seasons fully calculated
rainy_filtered_df = rainy_merged_df[~rainy_merged_df.season_approx.isin(['1999-2000', '2020-2021'])]

# get precipitation for rainy season

ij_df = pd.merge(rainy_filtered_df, daily_df, how='inner', on='ADM2_PCODE')
ij_df = ij_df[(ij_df.date >= ij_df.onset_date) & (ij_df.date <= ij_df.cessation_date)].reset_index()
prec_df = ij_df.groupby(['ADM2_PCODE', 'season_approx'])['mean_cell'].sum().reset_index().rename(columns={"mean_cell":"rainy_season_rainfall"})

# finalize rainy data
rainy_season_df = pd.merge(rainy_filtered_df, prec_df, on = ['ADM2_PCODE', 'season_approx'])
rainy_season_df = rainy_season_df[[
    'ADM2_PCODE',
    'ADM2_EN',
    'season_approx',
    'onset_date',
    'onset_month',
    'cessation_date',
    'cessation_month',
    'rainy_season_duration',
    'rainy_season_rainfall',
    'region'
]]

# rainy_season_df.to_csv(RAINY_SEASON, index=False)
```

### Dry spell processing

Now we will wrap up the analysis of the dry spell datasets, joining up with the rainy season data to see if there is any overlap with a rainy season and tidying up for upload.

```python
# get all rainy season dates
rainy_filtered_df['rainy_season_dates'] = [pd.date_range(start, end, freq = 'd', closed = 'left').delete(0) for start, end in zip(rainy_filtered_df.onset_date, rainy_filtered_df.cessation_date)]
rainy_dates = rainy_filtered_df.explode('rainy_season_dates') \
                      .loc[:,['pcode', 'rainy_season_dates']]

# check overlap with rainy season

ds_overlap = pd.merge(dry_spell_df, rainy_dates, on = 'pcode', how='inner')
ds_overlap = ds_overlap[ds_overlap.rainy_season_dates == ds_overlap.dry_spell_confirmation].reset_index(drop=True)
ds_overlap = ds_overlap[['pcode', 'dry_spell_confirmation']].assign(during_rainy_season=True)

dry_spell_df = pd.merge(dry_spell_df, ds_overlap, on=['pcode', 'dry_spell_confirmation'], how='left').fillna(False)

# if year is 2000 and month < 4 for start date, assume overlap with rainy season

dry_spell_df['during_rainy_season'] = np.where(
    (dry_spell_df.dry_spell_first_date.dt.year == 2000) & (dry_spell_df.dry_spell_first_date.dt.month < 4),
    True,
    dry_spell_df.during_rainy_season
)

# get precipitation during dry spell
ij_ds_df = pd.merge(dry_spell_df, daily_df, how='inner', left_on='pcode', right_on='ADM2_PCODE')
ij_ds_df = ij_ds_df[(ij_ds_df.date >= ij_ds_df.dry_spell_first_date) & (ij_ds_df.date <= ij_ds_df.dry_spell_last_date)].reset_index()
ds_prec_df = ij_ds_df.groupby(['pcode', 'dry_spell_confirmation'])['mean_cell'].sum().reset_index().rename(columns={"mean_cell":"dry_spell_rainfall"})

dry_spell_df = pd.merge(dry_spell_df, ds_prec_df, on=['pcode', 'dry_spell_confirmation'], how='left')

# calculate season for dry spells overlapping rainy seasons

dry_spell_df['season_approx'] = np.where(
    dry_spell_df.dry_spell_first_date.dt.month > 6,
    dry_spell_df.dry_spell_first_date.dt.year.astype('str') + "-" + (dry_spell_df.dry_spell_first_date.dt.year + 1).astype('str'),
    (dry_spell_df.dry_spell_first_date.dt.year-1).astype('str') + "-" + dry_spell_df.dry_spell_first_date.dt.year.astype('str')
)

dry_spell_df['season_approx'] = np.where(
    dry_spell_df.during_rainy_season,
    dry_spell_df.season_approx,
    "Not during a rainy season"
)

# final naming
dry_spell_df = pd.merge(dry_spell_df, adm_df[['ADM2_PCODE', 'ADM2_32EN']], left_on = 'pcode', right_on = 'ADM2_PCODE') \
    .rename(columns = {"ADM2_32EN":"ADM2_EN"})

dry_spell_df = dry_spell_df[[
    'ADM2_PCODE',
    'ADM2_EN',
    'season_approx',
    'dry_spell_first_date',
    'dry_spell_last_date',
    'dry_spell_duration',
    'dry_spell_rainfall',
    'during_rainy_season'
]]

# dry_spell_df.to_csv(DRY_SPELLS, index=False)
```
