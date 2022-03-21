### Water level in downstream basins
One indicator that might give us an indication for increased risk of flooding in South Sudan, is the water level in basins further downstream. 

[DAHITI provides these water levels](https://dahiti.dgfi.tum.de/en/map/). Here we analyze the water levels for the relevant basins and compare them to the Floodscan data. 

Note: as of now we only analyze Lake Albert cause inable to access the others, send them an email about it. 

Note2: before being able to download the data, you need to make an account and set the user credentials as env vars

```python
%load_ext autoreload
%autoreload 2
```

```python
import requests
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
import sys
from pathlib import Path
path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.config import Config
from src.utils_general.statistics import get_return_periods_dataframe
```

```python
%load_ext rpy2.ipython
```

```R
library(tidyverse)
library(lubridate)
```

```python
iso3="ssd"
config=Config()
glb_data_exploration_dir = Path(config.DATA_DIR) / config.PRIVATE_DIR / "exploration" / "glb"
```

```python
def download_dahiti_water_level(dahiti_id):
    """Download DAHITI water levels. 
    
    Largely copied from their example. 
    The ID can be found when clicking on the data
    point on their website. 
    """
    
    url = 'https://dahiti.dgfi.tum.de/api/v1/'

    args = {}
    """ required options """
    args['username'] = os.environ['DAHITI_USERNAME']
    args['password'] = os.environ['DAHITI_PASSWORD']
    args['action'] = 'download-water-level'
    #lake albert
    args['dahiti_id'] = dahiti_id #'85'

    """ send request as method POST """
    response = requests.post(url, data=args)

    if response.status_code == 200:
        """ convert json string in python list """
        data = json.loads(response.text)
        df=pd.json_normalize(data['target']['data'])
        return df
    else:
        return response.status_code
```

```python
# df_albert=download_dahiti_water_level('85')
# df_albert['date']=pd.to_datetime(df_albert.date)
# df_albert['year']=df_albert.date.dt.year
# df_albert.to_csv(glb_data_exploration_dir/'dahiti'/'dahiti_water_levels_lake_albert_id_85.csv',index=False)
```

```python
df_albert=pd.read_csv(glb_data_exploration_dir/'dahiti'/'dahiti_water_levels_lake_albert_id_85.csv',parse_dates=['date'])
```

```python
# #mehh not working atm, sent an email
# df_victoria=download_dahiti_water_level('2')
```

We can plot the height of the water level. 
We can see that the difference between most years is at maximum two meters, with an exception for 2020 and 2021. 

We don't see a very clear seasonality. 

```python
df_albert.plot(x='date',y='height',figsize=(15,8),marker='o')
```

It is important to note that the number of reporting points differs highly per year, see the histogram below. 

The points are also not spread evenly over time during a year. For example in 2012 there weren't any points after March. 

```python
g=df_albert[['date','year']].groupby('year').count().plot(kind="hist",title="Number of points per year");
```

```R
plotWaterLevel <- function (df,title){
#copied from https://stackoverflow.com/questions/51033854/x-axis-duplicates-for-a-time-series-using-facet-wrap-in-ggplot
df_plot <- df %>%
mutate_at(vars(-date), function(x) as.numeric(as.character(x))) %>%
mutate(date = as.Date(date),Month=month(date),Day=day(date)) %>%
mutate(dummy_date = paste("1997", Month, Day) %>% ymd())

df_plot %>%
ggplot(
aes_string(
x = "dummy_date",
y = "height"
)
) +
geom_point()+
# stat_smooth(
# geom = "area",
# span = 1/9,
# fill = "#ef6666"
# ) +
    scale_x_date(date_breaks = "3 months", date_labels = "%b") +
facet_wrap(
~year,
# scales="free_x",
ncol=5
) +
ylab("Flooded fraction")+
xlab("Month")+
labs(title=title)+
theme_minimal()
}
```

```R magic_args="-i df_albert -w 40 -h 20 --units cm"
# df_plot <- df_albert_country %>%
# mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_ADM0_PCODE = mean_ADM0_PCODE*100)
plotWaterLevel(df_albert,"Water level in lake Albert")
```

```python
df_albert['month'] = pd.DatetimeIndex(df_albert['date']).month
# df_albert_rainy = df_albert.loc[(df_albert['month'] >= 7) & (df_albert['month'] <= 10)]
```

Next we compute the return period and check which years had a peak above the return period. 
It is discussable whether only looking at the peak is the best method.. 

```python
#get one row per adm2-year combination that saw the highest mean value
df_albert_peak=df_albert_rainy.sort_values('height', ascending=False).drop_duplicates(['year'])
```

```python
years = np.arange(1.5, 20.5, 0.5)
```

```python
df_rps_ana=get_return_periods_dataframe(df_albert_peak, rp_var="height",years=years, method="analytical",round_rp=False)
df_rps_emp=get_return_periods_dataframe(df_albert_peak, rp_var="height",years=years, method="empirical",round_rp=False)
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_ana.index, df_rps_ana["rp"], label='analytical')
ax.plot(df_rps_emp.index, df_rps_emp["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('Fraction flooded');
```

```python
df_albert_peak[df_albert_peak.height>=df_rps_emp.loc[3,'rp']].sort_values('year')
```

```python
df_albert_peak[df_albert_peak.height>=df_rps_emp.loc[5,'rp']].sort_values('year')
```
