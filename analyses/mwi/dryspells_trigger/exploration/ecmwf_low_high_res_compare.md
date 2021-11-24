This notebook is for checking that the ECMWF data downloaded
from their API is equivalent and has been processed in a similar
manner to the data from CDS that we've been using so far.

```python
from pathlib import Path
import os
import sys
from datetime import datetime

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.ecmwf_seasonal import processing
```

```python
ISO3 = 'mwi'
VAR = 'precip'
```

```python
# Read in the two dataframes: One from CDS and the other from ECMWF API
da_cds = processing.get_ecmwf_forecast(ISO3)[VAR]
da_ecmwf =  processing.get_ecmwf_forecast(ISO3, source_cds=False)[VAR]
```

```python
# Need to flatten the data arrays to compare parameters across x-y coordinates.
# Can play around with these to view different times, ensemble member numbers,
# and forecast steps. 
time = datetime(1999, 1, 1)
number = 3
step = 2

data_cds = da_cds.sel(time=time, number=number, step=step)
data_ecmwf = da_ecmwf.sel(time=time, number=number, step=step)
```

```python
# CDS data has 1 deg resolution
data_cds.plot()
```

```python
data_ecmwf.latitude
```

```python
# ECMWF has 0.4 deg resolution, but coarsen only lets you combine
# integer number of pixels so can't reproduce the CDS resolution exactly.
# But it does look pretty close.
data_ecmwf.coarsen(longitude=2, latitude=2).mean().plot()
```
