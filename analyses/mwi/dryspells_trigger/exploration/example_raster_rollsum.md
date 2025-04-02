Really simple notebook used to generate an example of rolling sum identified dry spells kept at a raster level. These are then used for mapping in R as an example plot that could be used during the observational trigger monitoring for Malawi.

```python
from pathlib import Path
import sys
import os
import rioxarray
import pandas as pd
import numpy as np

explore_dir = os.path.join(os.getenv("AA_DATA_DIR"), "public", "exploration", "mwi", "arc2")
arc2_filepath = os.path.join(explore_dir, "arc2_20002020_approxmwi.nc")

#open the data
ds=rioxarray.open_rasterio(arc2_filepath,masked=True).squeeze()

ds_rolling=ds.rolling(T=14,min_periods=14).sum().dropna(dim="T",how="all")

ds_filter = ds_rolling.loc[ds_rolling.indexes["T"] >= pd.to_datetime("2018-01-01"),:,:]
ds_filter = ds_filter.loc[ds_filter.indexes["T"] <= pd.to_datetime("2018-01-31"),:,:]
((ds_filter <= 2).sum(axis = 0) > 0).to_netcdf(os.path.join(explore_dir, "example_arc2_rollsum_ds.nc"))
```
