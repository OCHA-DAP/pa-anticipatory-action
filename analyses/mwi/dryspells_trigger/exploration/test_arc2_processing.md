## ARC2 processing

This notebook is meant to provide simple examples and a space for testing the ARC2 processing script as it's developed and until a more proper space for examples is built/designed. The idea behind the processing is that the ARC2 class is defined for a specific country, date range, and geographical bounds. Each time the class is instantiated, the processes do all the work behind the scenes to load in and update new raw data and process it, maintaining all within a master file. Let's first look at just the raw data.

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import sys
from datetime import date

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.arc2_precipitation import DrySpells
from src.utils_general.raster_manipulation import compute_raster_statistics


poly_path = os.path.join(
    os.getenv('AA_DATA_DIR'),
    'public',
    'raw',
    'mwi',
    'cod_ab',
    'mwi_adm_nso_20181016_shp',
    'mwi_admbnda_adm2_nso_20181016.shp'
)
```

### Raw data downloading

So, first, let's just get a class and download data for the 2 days specified. Since we are specifying the class `DrySpells` rather than just `ARC2`, it will automatically ensure sufficient data is downloaded to calculate the rolling sum (and potential dry spells) from the start to end of our monitoring period.

```python
arc2_test = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2021-09-02",
    monitoring_end = "2021-09-03",
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S")
)

arc2_test.download_data(master=True)

ds = arc2_test.load_raw_data()
ds.indexes['T']
```

We can see here that we have data corrresponding to the 2nd and 3rd of September. Now, let's say we expand our `monitoring_start` and `monitoring_end` to be a day earlier and later respectively.

```python
arc2_test = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2000-01-01",
    monitoring_end = "2021-09-04",
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S"),
)

arc2_test.download_data()

ds2 = arc2_test.load_raw_data()
ds2.indexes['T']
```

In fact, now we can see that the data for the 2 extra days, on either side of the raw downloads was loaded into the master file. This master file is always accessible using the `load_raw_data()` method. Rather than managing a variety of downloads, this will always give access to the latest master (if the download has been done). So, of course, in the pipeline process we would want to call `date_max` based on the latest available day.

We can also see that if we process the data that we can calculate rolling sums across monitoring and potentially capture dry spells.

```python
arc2_test.downsample_data(poly_path, "ADM2_PCODE", reprocess=True)
arc2_test.calculate_rolling_sum()
arc2_test.identify_dry_spells()


```

Of course, we're in the middle of the dry season so in fact everywhere identifies a dry spell! Now, let's say we are again wanting to re-run the system. We can do data from the beginning of the year until today, since `monitoring_end` (and `monitoring_start`) accepts strings *or* dates. We can also use just 2 simple functions to redownload, reprocess, and identify all dry spells.

```python
arc2_test = DrySpells(
    country_iso3 = "mwi",
    monitoring_start = "2021-01-01",
    monitoring_end = date.today(),
    range_x = ("32E", "36E"),
    range_y = ("20S", "5S")
)

arc2_test.downsample_data(poly_path, "ADM2_PCODE", redownload=True)
arc2_test.identify_dry_spells(reprocess=True)
```
