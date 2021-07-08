#!/usr/bin/env python
# coding: utf-8

# ### Trigger mechanism for Somalia
#
# IPC trigger design as endorsed early 2020 (not clearly documented but was endorsed as using ML1 forecasts):
#
# - The projected national population in Phase 3 and above exceed 20%, AND
# - (The national population in Phase 3 is projected to increase by 5 percentage points, OR
# - The projected national population in Phase 4 or above is 2.5%)

import pandas as pd
import geopandas as gpd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.ipc_definemetrics import (
    define_trigger_percentage,
    define_trigger_increase,
)
from indicators.food_insecurity.utils import compute_percentage_columns


admin_level = 0
country = "somalia"
# suffix of filenames
suffix = ""
config = Config()
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
fewsnet_dir = os.path.join(
    country_folder, config.DATA_DIR, config.FEWSWORLDPOP_PROCESSED_DIR
)
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(
    country=country, admin_level=admin_level, suffix=suffix
)
globalipc_dir = os.path.join(
    country_folder, config.DATA_DIR, config.GLOBALIPC_PROCESSED_DIR
)
globalipc_path = os.path.join(
    globalipc_dir, f"{country}_globalipc_admin{admin_level}{suffix}.csv"
)

adm_bound_path = os.path.join(
    country_folder,
    config.DATA_DIR,
    config.SHAPEFILE_DIR,
    parameters[f"path_admin{admin_level}_shp"],
)


df_fn = pd.read_csv(os.path.join(fewsnet_dir, fewsnet_filename))
df_fn["source"] = "FewsNet"
df_gipc = pd.read_csv(globalipc_path)
df_gipc["source"] = "GlobalIPC"


df = pd.concat([df_fn, df_gipc])
df = df.replace(0, np.nan)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month


# the data of 2020-11 is an update and thus doesn't include CS data or projected periods
# add them here manually, where the CS data is set to that of 2020-10
CS_cols = [c for c in df.columns if "CS" in c]
for c in CS_cols:
    df.loc[df.date == "2020-11-01", c] = df.loc[
        df.date == "2020-10-01", c
    ].values
df[df.date == "2020-11-01"] = compute_percentage_columns(
    df[df.date == "2020-11-01"], config
)
df.loc[df.date == "2020-11-01", "period_ML1"] = "Nov - Jan 2021"
df.loc[df.date == "2020-11-01", "period_ML2"] = "Feb - May 2021"


df["trigger_ML1_3_20"] = df.apply(
    lambda x: define_trigger_percentage(x, "ML1", 3, 20), axis=1
)
df["trigger_ML1_3_5in"] = df.apply(
    lambda x: define_trigger_increase(x, "ML1", 3, 5), axis=1
)
df["trigger_ML1_4_2half"] = df.apply(
    lambda x: define_trigger_percentage(x, "ML1", 4, 2.5), axis=1
)

df["trigger_ML2_3_20"] = df.apply(
    lambda x: define_trigger_percentage(x, "ML2", 3, 20), axis=1
)
df["trigger_ML2_3_5in"] = df.apply(
    lambda x: define_trigger_increase(x, "ML2", 3, 5), axis=1
)
df["trigger_ML2_4_2half"] = df.apply(
    lambda x: define_trigger_percentage(x, "ML2", 4, 2.5), axis=1
)

# determine whether national trigger is met
df["threshold_reached_ML1"] = np.where(
    (df["trigger_ML1_3_20"] == 1)
    & ((df["trigger_ML1_3_5in"]) == 1 | (df["trigger_ML1_4_2half"] == 1)),
    1,
    0,
)
df["threshold_reached_ML2"] = np.where(
    (df["trigger_ML2_3_20"] == 1)
    & ((df["trigger_ML2_3_5in"]) == 1 | (df["trigger_ML2_4_2half"] == 1)),
    1,
    0,
)


df[df.date == "2020-11-01"]["pop_CS"]


df[df.date == "2020-11-01"][
    [
        "CS_3p",
        "perc_CS_3p",
        "CS_4p",
        "perc_CS_4p",
        "ML1_3p",
        "perc_ML1_3p",
        "ML1_4p",
        "perc_ML1_4p",
        "ML2_3p",
        "perc_ML2_3p",
        "ML2_4p",
        "perc_ML2_4p",
    ]
]


# FewsNet released an update of projections in November, though they don't differ much from those released in October.
# Combining the two, means our freshest data consists of a "current" assessment in Oct 2020, a short-range projection for Nov 2020-Jan 2021 and a long-range projection for Feb 2021 - May 2021.
# According to our methods, the figures are as follows
#
#

# | Period        | Population IPC 3+         | Percentage IPC 3+  |Population IPC 4+         | Percentage IPC 4+  |
# | ------------- |:-------------------------:| ------------------:|:------------------------:| ------------------:|
# | Oct 2020      | 3.4 million               | 21.6%              | 0                        |0                   |
# | Nov 2020 - Jan 2021     | 3.4 million     |   21.6%            | 44 thousand             |0.28%               |
# | Feb 2021 - May 2021 | 5.6 million      |    35.8% | 44 thousand | 0.28%

# There is a large discrepancy between the calculated population in IPC 3+ between our method and that of FewsNet. We don't know how FewsNet's population numbers are computed, however we do know that they assign a phase to an area if 20% of more of the population of that area is in the given phase or above. Since we don't have any information whether this number is between 20% and 100%, with our methodology we assign 100% of the population to the given phase. This might explain the discrepancy.
#
# To compute the percentage of the population in an IPC phase, we use the national population as reported by UNESA, which is 15.8 million for 2020.
#
# With these numbers, and the current trigger design, we would trigger for the Feb - May 2021 period, as more than 20% of the population is projected to be in IPC3+ and the increase of population in IPC 3+ compared to October is more than 5%. For the Nov 2020 -Jan 2021 the trigger was not met, since the percentage of population projected in IPC3+ didn't increase compared to the October situation, nor was the projected population in IPC 4+ more than 2.5%.

# Regarding the Deyr season of 2016,2017,2018 and 2019, we thought the data released in October and projecting up to May reflects the Deyr season the best. If you think this should instead be the data released in February (covering up to September), please let us know.
#
# The table below summarizes all the data. As you can see from here, the situation was significantly worse during Oct 2017 - May 2018 compared to now. However, compared to the other 3 years (2016,2017,2019), the sitation of this year (2021) is worse.
#
# For 2018, the trigger would be met both for the short-range and long-range projections. For 2017 the trigger would have been met for the long-range projections

# |Publication date  | Period        | Population IPC 3+         | Percentage IPC 3+  |Population IPC 4+         | Percentage IPC 4+  |
# |------------------:| ------------- |:-------------------------:| ------------------:|:------------------------:| ------------------:|
# |Oct 2015           | Oct 2015      | 1.4 million               | 10.6%              | 0                        |0%                   |
# |Oct 2016           | Oct 2016      | 1.4 million               |10.1%               |0                         |0%                   |
# |Oct 2017           | Oct 2017     | 3.4 million     |   21.6%            | 1.4 million             |9.4%
# |Oct 2018           | Oct 2018 | 1.3 million      |    8.6% | 0 | 0%
# | Oct 2020 | Oct 2020      | 3.4 million               | 21.6%              | 0                        |0                   |
# |.|  || || |
# |Oct 2015           | Oct 2015 - Dec 2015      | 1.5 million               | 11%              | 200 thousand                        |1.5%                   |
# |Oct 2016           | Oct 2016 - Jan 2017     |1.4 million |10.1% |0 |0%
# |Oct 2017           | Oct 2017 - Jan 2018     | 7.9 million | 54.6% |4.4 million | 30.5%|
# |Oct 2018           | Oct 2018 - Jan 2019 | 700 thousand | 4.6% | 200 thousand | 1.4%|
# | Oct 2020 | Nov 2020 - Jan 2021     | 3.4 million     |   21.6%            | 44 thousand             |0.28%               |
# |.|  || || |
# |Oct 2015           | Jan 2016 - Mar 2016      | 900 thousand | 6.6% | 0 | 0%|
# |Oct 2016           | Feb 2017 - May 2017 | 3.2 million | 22.4% | 0 |0%|
# |Oct 2017           | Feb 2018 - May 2018     |   10.6 million | 73.4% |6.4 million | 43.9% |
# |Oct 2018           | Feb 2019 - May 2019 | 1.8 million | 12.2% | 200 thousand | 1.4% |
# | Oct 2020| Feb 2021 - May 2021 | 5.6 million      |    35.8% | 44 thousand | 0.28%

df_neat_cs = df[["date", "CS_3p", "perc_CS_3p", "CS_4p", "perc_CS_4p"]].rename(
    columns={
        "date": "Publication date",
        "CS_3p": "Population IPC 3+",
        "perc_CS_3p": "Percentage IPC 3+",
        "CS_4p": "Population IPC 4+",
        "perc_CS_4p": "Percentage IPC 4+",
    }
)
df_neat_cs["Period"] = df_neat_cs["Publication date"]


df_neat_cs[
    df_neat_cs["Publication date"].isin(
        ["2016-02-01", "2017-02-01", "2018-02-01", "2019-02-01"]
    )
][
    [
        "Publication date",
        "Period",
        "Population IPC 3+",
        "Percentage IPC 3+",
        "Population IPC 4+",
        "Percentage IPC 4+",
    ]
].replace(
    np.nan, 0
).style.format(
    {
        "Publication date": "{:%Y-%m}",
        "Period": "{:%Y-%m}",
        "Population IPC 3+": "{:,.0f}",
        "Percentage IPC 3+": "{:.1f}%",
        "Population IPC 4+": "{:,.0f}",
        "Percentage IPC 4+": "{:.1f}%",
    }
)


p = "ML1"
df_neat_ml1 = df[
    [
        "date",
        f"period_{p}",
        f"{p}_3p",
        f"perc_{p}_3p",
        f"{p}_4p",
        f"perc_{p}_4p",
    ]
].rename(
    columns={
        "date": "Publication date",
        f"period_{p}": "Period",
        f"{p}_3p": "Population IPC 3+",
        f"perc_{p}_3p": "Percentage IPC 3+",
        f"{p}_4p": "Population IPC 4+",
        f"perc_{p}_4p": "Percentage IPC 4+",
    }
)


df_neat_ml1[
    df_neat_ml1["Publication date"].isin(
        ["2016-02-01", "2017-02-01", "2018-02-01", "2019-02-01"]
    )
][
    [
        "Publication date",
        "Period",
        "Population IPC 3+",
        "Percentage IPC 3+",
        "Population IPC 4+",
        "Percentage IPC 4+",
    ]
].replace(
    np.nan, 0
).style.format(
    {
        "Publication date": "{:%Y-%m}",
        "Population IPC 3+": "{:,.0f}",
        "Percentage IPC 3+": "{:.1f}%",
        "Population IPC 4+": "{:,.0f}",
        "Percentage IPC 4+": "{:.1f}%",
    }
)


p = "ML2"
df_neat_ml2 = df[
    [
        "date",
        f"period_{p}",
        f"{p}_3p",
        f"perc_{p}_3p",
        f"{p}_4p",
        f"perc_{p}_4p",
    ]
].rename(
    columns={
        "date": "Publication date",
        f"period_{p}": "Period",
        f"{p}_3p": "Population IPC 3+",
        f"perc_{p}_3p": "Percentage IPC 3+",
        f"{p}_4p": "Population IPC 4+",
        f"perc_{p}_4p": "Percentage IPC 4+",
    }
)


df_neat_ml2[
    df_neat_ml2["Publication date"].isin(
        ["2016-02-01", "2017-02-01", "2018-02-01", "2019-02-01"]
    )
][
    [
        "Publication date",
        "Period",
        "Population IPC 3+",
        "Percentage IPC 3+",
        "Population IPC 4+",
        "Percentage IPC 4+",
    ]
].replace(
    np.nan, 0
).style.format(
    {
        "Publication date": "{:%Y-%m}",
        "Population IPC 3+": "{:,.0f}",
        "Percentage IPC 3+": "{:.1f}%",
        "Population IPC 4+": "{:,.0f}",
        "Percentage IPC 4+": "{:.1f}%",
    }
)


df[
    df.date.isin(
        [
            "2015-10-01",
            "2016-10-01",
            "2017-10-01",
            "2018-10-01",
            "2019-10-01",
            "2020-11-01",
        ]
    )
][
    [
        "date",
        "period_ML1",
        "period_ML2",
        "CS_3p",
        "perc_CS_3p",
        "CS_4p",
        "perc_CS_4p",
        "ML1_3p",
        "perc_ML1_3p",
        "ML1_4p",
        "perc_ML1_4p",
        "ML2_3p",
        "perc_ML2_3p",
        "ML2_4p",
        "perc_ML2_4p",
    ]
]


df[df.threshold_reached_ML1 == 1][
    [
        "date",
        "perc_ML1_3p",
        "perc_ML1_4p",
        "perc_CS_3p",
        "perc_inc_ML1_3p",
        "source",
    ]
]


df[df.threshold_reached_ML2 == 1][
    [
        "date",
        "perc_ML2_3p",
        "perc_ML2_4p",
        "perc_CS_3p",
        "perc_inc_ML2_3p",
        "source",
    ]
]


df_gipc = df[df.source == "GlobalIPC"]


df_gipc[df_gipc.date.isin(["2017-01", "2017-07", "2018-01"])][
    ["date", "perc_ML1_3p", "perc_CS_3p"]
]
