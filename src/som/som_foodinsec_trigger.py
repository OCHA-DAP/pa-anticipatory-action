# Compute the trigger information needed for the dashboard. Ugly now for
# first mock-up, should be improved later on

import warnings

import numpy as np

# IPC trigger design as of 01-02-2021: The projected national population
# in Phase 3 and above exceed20 %, AND (The national population in Phase
# 3 is projected to increase by 5 percentage points, OR The projected
# national population in Phase 4 or above is 2.5 %)
import pandas as pd

warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import Config
from src.indicators.food_insecurity.ipc_definemetrics import (
    define_trigger_increase,
    define_trigger_percentage,
)

admin_level = 0
iso3 = "som"
# suffix of filenames
suffix_fn = ""  # "202106"
suffix_ipcg = ""
suffix_output = ""  # "202106"
config = Config()
parameters = config.parameters(iso3)

country_data_raw_dir = os.path.join(config.DATA_PUBLIC_RAW_DIR, iso3.lower())
glb_data_raw_dir = os.path.join(config.DATA_PUBLIC_RAW_DIR, "glb")
country_data_processed_dir = os.path.join(
    config.DATA_PUBLIC_PROCESSED_DIR, iso3.lower()
)

fewsnet_dir = os.path.join(
    country_data_processed_dir, config.FEWSWORLDPOP_PROCESSED_DIR
)
fewsnet_filename = config.FEWSWORLDPOP_PROCESSED_FILENAME.format(
    iso3=iso3, admin_level=admin_level, suffix=suffix_fn
)
globalipc_dir = os.path.join(
    country_data_processed_dir, config.GLOBALIPC_PROCESSED_DIR
)
globalipc_path = os.path.join(
    globalipc_dir, f"{iso3}_globalipc_admin{admin_level}{suffix_ipcg}.csv"
)

adm_bound_path = os.path.join(
    country_data_raw_dir,
    config.SHAPEFILE_DIR,
    parameters[f"path_admin{admin_level}_shp"],
)

output_dir = os.path.join(country_data_processed_dir, "trigger_metrics")
output_path = os.path.join(
    output_dir, f"{iso3}_foodinsec_trigger{suffix_output}.csv"
)


df_fn = pd.read_csv(os.path.join(fewsnet_dir, fewsnet_filename))
df_fn["source"] = "FewsNet"
df_gipc = pd.read_csv(globalipc_path)
df_gipc["source"] = "GlobalIPC"

df = pd.concat([df_fn, df_gipc])
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

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

df.loc[df.date == "2020-10-01", "period_ML1"] = "Oct 2020 - Jan 2021"
df.loc[df.date == "2020-10-01", "period_ML2"] = "Feb 2021 - May 2021"
df.to_csv(output_path, index=False)
