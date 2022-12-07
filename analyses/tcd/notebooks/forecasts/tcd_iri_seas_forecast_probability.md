# IRI forecast and frequency of a probability higher than 42.5%
For TCD the trigger was set at a probability of 42.5% of below average rainfall.
However, we only have IRI forecasts since 2017 and during those years this trigger was never met. 
Simeltaneously no large drought occured during those years. We can therefore not compute representable performance metrics. 

Instead of focusing on our region of interest, this notebooks investigates how often a probability of 42.5% has occurred globally since 2017. 
While far from perfect, this could potentially serve as an estimate for a return period. 


#### Load libraries and set global constants

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import pandas as pd
import rioxarray
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta
import hvplot.xarray
from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

# TODO: update to using toolbox, but need to upgrade hxl version for that..
from src.indicators.drought.iri_rainfallforecast import get_iri_data
```

## Inspect forecasts


We load all IRI data and plot the distribution of below average probabilities. 
We see that the median lies around 34% and the values are not normally distributed. 
We don't see large differences in the distributions between leadtimes

```python
config = Config()
```

```python
# C indicates the tercile (below-average, normal, or above-average).
# F indicates the publication month, and L the leadtime
ds_iri = get_iri_data(config, download=False)
ds_iri = ds_iri.rio.write_crs("EPSG:4326", inplace=True)
da_iri = ds_iri.prob
da_iri_bavg = da_iri.sel(C=0).drop("spatial_ref", axis=1)
```

```python
da_iri_bavg.median()
```

```python
da_iri_bavg.hvplot.hist("prob", alpha=0.5).opts(
    ylabel="Occurence below average",
    title="Forecasted probabilities of below average \n at raster level in the whole world across all seasons and leadtimes, 2017-2021",
)
```

```python
da_iri_bavg.hvplot.violin("prob", by="L", color="L", cmap="Category20").opts(
    ylabel="Probability below average",
    xlabel="leadtime",
    title="Observed probabilities of bavg at raster level globally",
)
```

### Frequency of meeting 42.5%
We can see that the frequency of meeting the 42.5% at raster cell level globally is surprisingly
low, only 5%, i.e. 1 in every 20. 

```python
thresh = 42.5
```

```python
da_iri_bavg.where((da_iri_bavg >= thresh)).count() / da_iri_bavg.count()
```

### Frequency of reaching 42.5% during any of the leadtimes
The trigger is evaluated at each leadtime, i.e. from 1 to 4 months leadtime
We just want to know if the frequency of reaching the threshold increases
if we only require the threshold to be met during one of the 4 leadtimes. 

We can see that this increases the frequency, i.e. 1 in every 10 times this is true

```python
datetimeindex = da_iri_bavg.indexes["F"].to_datetimeindex()
da_iri_bavg["F"] = datetimeindex
```

```python
# We convert the dataarray to a dataframe to make computations easier
df_bavg = da_iri_bavg.to_dataframe().reset_index()
```

```python
# drop nan values
df_bavg = df_bavg[~df_bavg.prob.isnull()]
```

```python
# this takes some time
# comp month the forecast is predicting
# in reality it is a 3 month period for which this is the start month
df_bavg.loc[:, "pred_month"] = df_bavg.apply(
    lambda x: x["F"] + relativedelta(months=int(x["L"])), axis=1
)
```

```python
df_bavg["meet_thresh"] = np.where(df_bavg.prob >= thresh, True, False)
```

```python
# get sum of leadtimes that predict above the threshold
# for the given predicted period and coordinates
df_bavg_group_thresh = df_bavg.groupby(
    ["pred_month", "latitude", "longitude"], as_index=False
).meet_thresh.sum()
```

```python
df_bavg_group_thresh.meet_thresh.value_counts()
```

```python
# probability of 1 of the leadtimes meeting the threshold
df_bavg_group_thresh[
    df_bavg_group_thresh.meet_thresh >= 1
].meet_thresh.count() / df_bavg_group_thresh.meet_thresh.count()
```

Look at the two separate triggers separately for reporting in the final phase.

```python
df_trigger1 = (
    df_bavg[df_bavg.L.isin([4, 3])]
    .groupby(["pred_month", "latitude", "longitude"], as_index=False)
    .meet_thresh.sum()
)

df_trigger1[
    df_trigger1.meet_thresh >= 1
].meet_thresh.count() / df_trigger1.meet_thresh.count()
```

```python
df_trigger2 = (
    df_bavg[df_bavg.L.isin([2, 1])]
    .groupby(["pred_month", "latitude", "longitude"], as_index=False)
    .meet_thresh.sum()
)

df_trigger2[
    df_trigger2.meet_thresh >= 1
].meet_thresh.count() / df_trigger2.meet_thresh.count()
```

### Thoughts
These return periods feel way lower than based on experience/feeling. 
Especially adding that this is at raster cell level but we require a larger area
to reach the threshold, i.e. the actual return period would only be lower.
I am therefore wondering if we are doing this correctly..

Thought maybe it difers per time, but doesn't seem to matter too much. Could geographical area also differ? 

```python
df_bavg_group_thresh_month_interest = df_bavg_group_thresh[
    df_bavg_group_thresh.pred_month.dt.month == 7
]
```

```python
# probability of 1 of the leadtimes meeting the threshold
df_bavg_group_thresh_month_interest[
    df_bavg_group_thresh_month_interest.meet_thresh >= 1
].meet_thresh.count() / df_bavg_group_thresh_month_interest.meet_thresh.count()
```

```python

```
