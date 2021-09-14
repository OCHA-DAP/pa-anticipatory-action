## Data processing to help compare ARC2 and CHIRPS dry spells in Malawi

This notebook does some data processing to help draw compirasons between dry spells identified through ARC2 and CHIRPS data to be used in finalizing observational dry spell triggers in Malawi. Previously, I was also doing some visualizations and analysis but have refactored and put this all into an identically named `.Rmd` file.

### Global libraries

```python
from pathlib import Path
import sys
import os

import rioxarray
from shapely.geometry import mapping
import pandas as pd
import itertools
import numpy as np
```

### Local configuration

```python
path_mod = f"{Path(os.path.abspath('')).parents[3]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

config = Config()
parameters = config.parameters('malawi')
COUNTRY_ISO3 = parameters["iso3_code"]
DATA_DIR = Path(config.DATA_DIR)

PROCESSED_DIR =   DATA_DIR / config.PUBLIC_DIR / config.PROCESSED_DIR / COUNTRY_ISO3
EXPLORATION_DIR = DATA_DIR / config.PUBLIC_DIR / 'exploration' / COUNTRY_ISO3

ARC2_FILEPATH = EXPLORATION_DIR / 'arc2' / 'mwi_arc2_dry_spells.csv'
ARC2_CENTER_FILTERED_FILEPATH = EXPLORATION_DIR / 'arc2' / 'mwi_arc2_dry_spells_during_rainy_season.csv'
ARC2_TOUCHING_FILEPATH = EXPLORATION_DIR / 'arc2' / 'mwi_arc2_touching_dry_spells.csv'
ARC2_TOUCHING_FILTERED_FILEPATH = EXPLORATION_DIR / 'arc2' / 'mwi_arc2_touching_dry_spells_during_rainy_season.csv'
ARC2_PRECIP_FILEPATH = EXPLORATION_DIR / 'arc2' / 'mwi_arc2_precip_long.csv'
CHIRPS_FILEPATH = PROCESSED_DIR / 'dry_spells' / 'dry_spells_during_rainy_season_list_2000_2021_mean_back.csv'
RAINY_SEASON_FILEPATH = PROCESSED_DIR / 'dry_spells' / 'rainy_seasons_detail_2000_2021_mean_back.csv'

PLOT_DIR = PROCESSED_DIR / 'plots' / 'dry_spells' / 'arc2_chirps_compare'
```

### Data loading

```python
arc2_df = pd.read_csv(ARC2_FILEPATH, parse_dates = [1,2,4], index_col = 0)
chirps_df = pd.read_csv(CHIRPS_FILEPATH, parse_dates = [3,4])
rainy_df = pd.read_csv(RAINY_SEASON_FILEPATH, parse_dates = [4,6])
arc2_precip_df = pd.read_csv(ARC2_PRECIP_FILEPATH, index_col = 0, parse_dates = ['date'])
```

### Filter ARC2 data frame (centroids)

Since the ARC2 data frame contains all recorded dry spells, need to filter it to rainy season (identified through CHIRPS). By definition, that means the onset or cessation of any dry spell must fall within the rainy season (otherwise it certainly is not a rainy season).

```python
# get all rainy season dates
rainy_df = rainy_df[rainy_df.onset_date.notnull()]
rainy_df['rainy_season_dates'] = [pd.date_range(start, end, freq = 'd', closed = 'left').delete(0) for start, end in zip(rainy_df.onset_date, rainy_df.cessation_date)]
rainy_dates = rainy_df.explode('rainy_season_dates') \
                      .loc[:,['pcode', 'rainy_season_dates']]

# join to check for rainy season and get region from CHIRPS
arc2_df = arc2_df.merge(rainy_dates, how = 'left', left_on = ['pcode', 'dry_spell_first_date'], right_on = ['pcode', 'rainy_season_dates']) \
                 .merge(rainy_dates, how = 'left', left_on = ['pcode', 'dry_spell_last_date'], right_on = ['pcode', 'rainy_season_dates']) \
                 .merge(chirps_df[['pcode', 'region', 'ADM2_EN']], how = 'left', on = 'pcode') \
                 .drop_duplicates()

arc2_center_df = arc2_df[arc2_df.rainy_season_dates_x.notnull() | arc2_df.rainy_season_dates_y.notnull()]
arc2_center_df = arc2_center_df[arc2_center_df.region == 'Southern'].reset_index(drop=True)

# rearrange columns and save
arc2_center_df = arc2_center_df[['pcode', 'ADM2_EN', 'dry_spell_first_date', 'dry_spell_confirmation', 'dry_spell_last_date', 'dry_spell_duration', 'season_approx']]
arc2_center_df.to_csv(ARC2_CENTER_FILTERED_FILEPATH)
```

### Calculate ARC2 dry spells using all raster cells touching

The above data frame from ARC2 was calculated using raster centroids contained within an ADM2 boundary. Given the small size of some ADM2 regions, particularly Zomba City which contains no centroid, this could be a useful comparison. It turns out the mean cells in the ARC2 precipation file have already been rolling summed, so no need to recalculate.

```python
# capture dry spell groups, all code here piecemeal for checking
arc2_precip_df['dry_spell'] = arc2_precip_df['mean_cell_touched'] <= 2
arc2_precip_df['dry_spell_diff'] = arc2_precip_df.groupby('ADM2_PCODE')['dry_spell'].diff() != 0
arc2_precip_df.sort_values(by = ['ADM2_PCODE', 'date'], axis = 0, inplace = True)
arc2_precip_df['dry_spell_groups'] = arc2_precip_df['dry_spell_diff'].cumsum()

# get all dry spells

arc2_touch_df = arc2_precip_df[arc2_precip_df.dry_spell] \
    .groupby('dry_spell_groups') \
    .agg(
      pcode = ('ADM2_PCODE','unique'),
      dry_spell_confirmation = ('date','min'),
      dry_spell_last_date = ('date','max')
     ) \
    .reset_index(drop = True) \
    .assign(pcode = lambda x: x.pcode.str[0],
            dry_spell_first_date = lambda x: x.dry_spell_confirmation - pd.to_timedelta(13, unit = 'd'),
            dry_spell_duration = lambda x: (x.dry_spell_last_date - x.dry_spell_first_date).dt.days)
```

### Filter ARC2 data frame (touching)

Conduct the same filtering done above based on CHIRPS identified rainy seasons.

```python
# join to check for rainy season and get region from CHIRPS
arc2_touch_df = arc2_touch_df \
    .merge(rainy_dates, how = 'left', left_on = ['pcode', 'dry_spell_first_date'], right_on = ['pcode', 'rainy_season_dates']) \
    .merge(rainy_dates, how = 'left', left_on = ['pcode', 'dry_spell_last_date'], right_on = ['pcode', 'rainy_season_dates']) \
    .merge(chirps_df[['pcode', 'region', 'ADM2_EN']], how = 'left', on = 'pcode') \
    .drop_duplicates()

arc2_touch_df.to_csv(ARC2_TOUCHING_FILEPATH)

arc2_touch_df = arc2_touch_df[arc2_touch_df.rainy_season_dates_x.notnull() | arc2_touch_df.rainy_season_dates_y.notnull()]
arc2_touch_df = arc2_touch_df[arc2_touch_df.region == 'Southern'].reset_index(drop=True)

# rearrange columns and save
arc2_touch_df = arc2_touch_df[['pcode', 'ADM2_EN', 'dry_spell_first_date', 'dry_spell_confirmation', 'dry_spell_last_date', 'dry_spell_duration']]
arc2_touch_df.to_csv(ARC2_TOUCHING_FILTERED_FILEPATH)
```
