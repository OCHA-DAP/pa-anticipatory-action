### Flooded fraction in the region of interest
As we can see in the notebook `ssd_floodscan_adm2`, there is a lot of fluctuation between the counties.
At the same time, the country level, as analyzed in `ssd_floodscan_country` might be too large. 
We therefore also examine a subset of counties, referred to as the region of interest. 

This notebook looks at the flooded fraction over time, and sees if the flooded fraction earlier in the season gives a signal for more extensive flooding later on. 

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import sys
from datetime import timedelta
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.util.testing import assert_frame_equal
import geopandas as gpd
import hvplot.xarray
import matplotlib as mpl
import numpy as np
from scipy import stats
from functools import reduce
import altair as alt
import panel.widgets as pnw
import calendar
import rioxarray

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.indicators.flooding.floodscan import floodscan
from src.utils_general.raster_manipulation import compute_raster_statistics
from src.utils_general.statistics import get_return_periods_dataframe


# mpl.rcParams['figure.dpi'] = 300
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
```

#### define functions

```R
plotFloodedFraction <- function (df,y_col,facet_col,title){
df %>%
ggplot(
aes_string(
x = "time",
y = y_col
)
) +
stat_smooth(
geom = "area",
span = 1/4,
fill = "#ef6666"
) +
scale_x_date(
date_breaks = "3 months",
date_labels = "%b"
) +
facet_wrap(
as.formula(paste("~", facet_col)),
scales="free_x",
ncol=5
) +
ylab("Flooded fraction")+
xlab("Month")+
labs(title = title)+
theme_minimal()
}
```

```python
iso3="ssd"
config=Config()
parameters = config.parameters(iso3)
country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / iso3
country_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / iso3
country_data_processed_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "processed" / iso3
country_data_public_exploration_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / "exploration" / iso3
bentiu_bound_path=country_data_processed_dir / "bentiu" / "bentiu_bounding_box.gpkg"
```

```python
gdf_bentiu=gpd.read_file(bentiu_bound_path)
```

```python
fs_clip=xr.load_dataset(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan.nc')
#I dont fully understand why, these grid mappings re-occur and what they mean
#but if having them, later on getting crs problems when computing stats
fs_clip.SFED_AREA.attrs.pop('grid_mapping')
fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
fs_clip=fs_clip.rio.write_crs("EPSG:4326",inplace=True)
```

```python
da_clip=fs_clip.SFED_AREA
```

```python
df_floodscan_reg=compute_raster_statistics(
        gdf=gdf_bentiu,
        bound_col="id",
        raster_array=da_clip,
        lon_coord="lon",
        lat_coord="lat",
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=True,
    )
df_floodscan_reg['year']=df_floodscan_reg.time.dt.year
```

We can plot the data over all years. 
We see a yearly pattern where some years the peak is higher than others (though a max of 1.75% of the country is flooded). 

We see that some peaks have very high outliers, while others are wider. Which to classify as a flood, I am unsure about. With the method of std, we are now looking at the high outliers. 

```python
#should document but for now removing 2022 as it is not a full year
#but does have very high values till feb, so computations get a bit skewed with that
df_floodscan_reg=df_floodscan_reg[df_floodscan_reg.year<=2021]
```

```python
df_floodscan_reg['mean_rolling']=df_floodscan_reg.sort_values('time').mean_id.rolling(10,min_periods=10).mean()
```

```python
df_floodscan_reg['month'] = pd.DatetimeIndex(df_floodscan_reg['time']).month
df_floodscan_reg_rainy = df_floodscan_reg.loc[(df_floodscan_reg['month'] >= 7) & (df_floodscan_reg['month'] <= 10)]
```

```python
fig, ax = plt.subplots(figsize=(20,6))
sns.lineplot(data=df_floodscan_reg, x="time", y="mean_id", lw=0.25, label='Original')
sns.lineplot(data=df_floodscan_reg, x="time", 
             y="mean_rolling", lw=0.25, label='10-day moving\navg')   
ax.set_ylabel('Flooded fraction')
ax.set_xlabel('Date')
ax.set_title(f'Flooding in Bentiu, 1998-2021')
ax.legend()
```

```python
# df_floodscan_reg.to_csv(
#     country_data_exploration_dir / "floodscan" / "bentiu_flood.csv"
# )
```

```python

```
