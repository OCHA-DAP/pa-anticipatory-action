## ARC2 values checking

When looking at compiling data for HDX, I noticed some strange values in the aggregated daily precipitation CSV. These values were all averaged to -999, indicating that likely there was some issues with missing data. Let's have a look and see what we can find!


from shapely.geometry import mapping # need to import shapely first to avoid errors on MacOS
import rioxarray
import cftime
import numpy as np
import geopandas as gpd
import pandas as pd

```python
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

# data directories
data_dir = os.environ["AA_DATA_DIR"]
arc2_dir = os.path.join(data_dir, "public", "exploration", "mwi", "arc2")

# new and old downloads
arc2_filepath_new = os.path.join(arc2_dir, "arc2_20002021_approxmwi.nc")
arc2_filepath_old = os.path.join(arc2_dir, "arc2_20002020_approxmwi.nc")

# aggregates
arc2_daily_new = os.path.join(arc2_dir, "mwi_arc2_precip_long_raw.csv")
arc2_daily_old = os.path.join(arc2_dir, "mwi_arc2_precip_long.csv")
```

## Aggregated daily precipitation

First, let's have a look at the actual values identified in the CSV to quickly check what I saw before and display here.

```python
daily_old = pd.read_csv(arc2_daily_old, parse_dates=["date"])
daily_old = daily_old[['ADM2_EN', 'ADM2_PCODE', 'date', 'mean_cell']]
daily_new = pd.read_csv(arc2_daily_new, parse_dates=["date"])
daily_new = daily_new[['ADM2_EN', 'ADM2_PCODE', 'date', 'mean_cell']]

daily_new[daily_new.mean_cell==-999]
```

Alright, so we can see that we have approximately 240 rows where the raster average is `-999`. How many dates does this correspond to?

```python
daily_new[daily_new.mean_cell==-999].date.unique()
```

Just 8 dates. However, it's interesting to note that all of these dates except the one in April lie within a possible rainy season, and 5 of them within our monitoring period. Now, this aggregation was based off of a re-analysis done to confirm what was found. What was in the original data that was produced and used in the trigger development?

```python
daily_old[daily_old.mean_cell==-999]
```

Well, there's at least no `-999` values. So what does it look like on those dates?

```python
daily_old[daily_old.date == "2001-01-09"]
```

In fact, there is no data for that day. And it turns out, there is no daily precipitation data for the following 13 days! This is because the original analysis was doing a rolling sum at the raster cell level, so all sums over the missing value were themselves classified as missing. Let's see what dates these are?

```python
daily_new[(~daily_new.date.isin(daily_old.date)) & (daily_new.date >= "2000-01-14") & (daily_new.date <= "2021-03-31")] \
    .reset_index(drop = True) \
    .date \
    .unique()
```

So there we can see all of them missing dates from the previous analysis.

## Input ARC2 data

So very quickly, what does the input ARC2 data look like? Let's have a look at the values on the first date where we have issues, the 9th of January 2001. Checking the values in a new download as well as an old one to ensure that we check that nothing has potentially changed.

```python
#open the data
ds_new=rioxarray.open_rasterio(arc2_filepath_new,masked=True).squeeze().to_dataset()
ds_old=rioxarray.open_rasterio(arc2_filepath_old,masked=True).squeeze().to_dataset()
missing_date = cftime.DatetimeGregorian(2001, 1, 9, 12, 0, 0, 0)

ds_new.where(ds_new.T == missing_date, drop = True).est_prcp.values
```

```python
ds_old.where(ds_old.T == missing_date, drop = True).est_prcp.values
```

## What next?

So, there's clearly an issue where we have missing data in the ARC2 inputs that we need to deal with explicitly. We need to impute the data. We could:

- Impute using `0`, which would bias the analysis to possibly mis-detect dry spells when data is missing.
- Linearly interpolate the values so that we essentially take the average of the surrounding two dates. You could also weight this across a wider range of dates to smooth the average.

However, it would be best to understand why these values are missing! Almost all of them, as mentioned above, occur during the rainy season so could it be high levels of precipitation that could not be accurately captured so were dropped? Understanding the phenomena behind the missing values can help us hopefully choose a logical solution.
