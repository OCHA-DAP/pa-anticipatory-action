import pandas as pd
from utils import parse_args, parse_yaml, config_logger
from pathlib import Path
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

# TODO: check that all cols, so CS ML1 ML2 1 to 5 and pop cols are present in input data
# TODO: quality check that perc cols add up to 100


def add_columns(df, source):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["Source"] = source

    # TODO: these are column names used by FewsNet, already change in process_fewsnet
    df = df.rename(columns={"adjusted_population": "pop_Country", "ADM1_EN": "ADMIN1"})

    # calculate percentage of population per analysis period and level
    for period in ["CS", "ML1", "ML2"]:
        # IPC level goes up to 5, so define range up to 6
        for i in range(1, 6):
            c = f"{period}_{i}"
            df[f"perc_{c}"] = df[c] / df[f"pop_{period}"] * 100
        # get pop and perc in IPC3+ and IPC2-
        # 3p = IPC level 3 or higher, 2m = IPC level 2 or lower
        df[f"{period}_3p"] = df[[f"{period}_{i}" for i in range(3, 6)]].sum(axis=1)
        df[f"perc_{period}_3p"] = df[f"{period}_3p"] / df[f"pop_{period}"] * 100
        df[f"{period}_2m"] = df[[f"{period}_{i}" for i in range(1, 3)]].sum(axis=1)
        df[f"perc_{period}_2m"] = df[f"{period}_2m"] / df[f"pop_{period}"] * 100
    df["perc_inc_ML2_3p"] = df["perc_ML2_3p"] - df["perc_CS_3p"]
    df["perc_inc_ML1_3p"] = df["perc_ML1_3p"] - df["perc_CS_3p"]
    return df


def get_trigger(row, period, level, perc):
    """
    Return 1 if percentage of population in row for period in level "level" or higher, equals or larger than perc
    """
    # range till 6 cause 5 is max level
    cols = [f"{period}_{l}" for l in range(level, 6)]
    if np.isnan(row[f"pop_{period}"]):
        return np.nan
    if round(row[cols].sum() / row[f"pop_{period}"] * 100) >= perc:
        return 1
    else:
        return 0


# TODO: we are not using the relative increase at the moment, do we want to remove it?
def get_trigger_increase_rel(row, level, perc):
    """
    Return 1 if population in row for >="level" at ML1 is expected to be larger than (current (CS) population in >=level) * (1+(perc/100))
    """
    # range till 6 cause 5 is max level
    cols_ml1 = [f"ML1_{l}" for l in range(level, 6)]
    cols_cs = [f"CS_{l}" for l in range(level, 6)]
    if row[["pop_CS", "pop_ML1"]].isnull().values.any():
        return np.nan
    elif row[cols_ml1].sum() == 0:
        return 0
    elif row[cols_ml1].sum() > 0 and row[cols_cs].sum() == 0:
        return 1
    elif (
        round((row[cols_ml1].sum() - row[cols_cs].sum()) / row[cols_cs].sum() * 100)
        >= perc
    ):
        return 1
    else:
        return 0


def get_trigger_increase(row, period, level, perc):
    """
    Return 1 if for "row", if the expected increase in the percentage of the population in "level" or higher at time "period" compared to currently (CS) is expected to be larger than "perc"
    For Global IPC the population analysed in ML2 is sometimes different than in CS. That is why we work dirrectly with percentages and not anymore with (pop period level+ - pop CS level+) / pop CS
    """
    # range till 6 cause 5 is max level
    cols_perc_proj = [f"perc_{period}_{l}" for l in range(level, 6)]
    cols_perc_cs = [f"perc_CS_{l}" for l in range(level, 6)]
    if row[["pop_CS", f"pop_{period}"]].isnull().values.any():
        return np.nan
    if row[cols_perc_proj].sum() == 0:
        return 0
    if round(row[cols_perc_proj].sum() - row[cols_perc_cs].sum()) >= perc:
        return 1
    else:
        return 0


def compute_trigger(df):
    # TODO: would be great if we can define in config or so which triggers to compute. Not sure how..

    # get yes/no for different thresholds, i.e. column value for row will be 1 if threshold is met and 0 if it isnt
    df["threshold_ML1_4_20"] = df.apply(lambda x: get_trigger(x, "ML1", 4, 20), axis=1)
    df["threshold_ML1_3_30"] = df.apply(lambda x: get_trigger(x, "ML1", 3, 30), axis=1)
    df["threshold_ML1_3_5i"] = df.apply(
        lambda x: get_trigger_increase(x, "ML1", 3, 5), axis=1
    )
    df["threshold_ML2_4_20"] = df.apply(lambda x: get_trigger(x, "ML2", 4, 20), axis=1)
    df["threshold_ML2_3_30"] = df.apply(lambda x: get_trigger(x, "ML2", 3, 30), axis=1)
    df["threshold_ML2_3_5i"] = df.apply(
        lambda x: get_trigger_increase(x, "ML2", 3, 5), axis=1
    )

    df["trigger_ML1"] = (df["threshold_ML1_4_20"] == 1) | (
        (df["threshold_ML1_3_30"] == 1) & (df["threshold_ML1_3_5i"] == 1)
    )
    df["trigger_ML2"] = (df["threshold_ML2_4_20"] == 1) | (
        (df["threshold_ML2_3_30"] == 1) & (df["threshold_ML2_3_5i"] == 1)
    )
    return df


def main(country_iso3, admin_level, config_file="config.yml"):
    parameters = parse_yaml(config_file)[country_iso3]
    country = parameters["country_name"]
    start_date = parameters["start_date"]
    end_date = parameters["end_date"]
    FEWS_PROCESSED_FOLDER = f"{country}/Data/FewsNetCombined/"
    GIPC_PROCESSED_FOLDER = f"{country}/Data/GlobalIPCProcessed/"
    processed_fews_path = f"{FEWS_PROCESSED_FOLDER}{country}_admin{admin_level}_fewsnet_combined_{start_date}_{end_date}.csv"
    processed_globalipc_path = (
        f"{GIPC_PROCESSED_FOLDER}{country}_globalipc_ADMIN{admin_level}.csv"
    )

    # 3p = IPC level 3 or higher, 2m = IPC level 2 or lower
    ipc_cols = [
        f"{period}_{i}"
        for period in ["CS", "ML1", "ML2"]
        for i in [1, 2, 3, 4, 5, "3p", "2m"]
    ] + [
        f"perc_{period}_{i}"
        for period in ["CS", "ML1", "ML2"]
        for i in [1, 2, 3, 4, 5, "3p", "2m"]
    ]
    pop_cols = [
        f"pop_{period}" for period in ["CS", "ML1", "ML2", f"ADMIN{admin_level}"]
    ]

    # TODO: implement ADMIN0 in preprocess scripts, and select it here as well
    adm_cols = [f"ADMIN{a}" for a in range(1, int(admin_level) + 1)]

    # initialize dataframes such that can later check if they are filled with data
    df_fewss = None
    df_gipcs = None

    if os.path.exists(processed_fews_path):
        df_fews = pd.read_csv(processed_fews_path, index_col=0)
        # TODO: adjust column names in process_fewsnet.py instead
        df_fews = df_fews.rename(
            columns={
                parameters["shp_adm1c"]: "ADMIN1",
                parameters["shp_adm2c"]: "ADMIN2",
                "adjusted_population": f"pop_ADMIN{admin_level}",
            }
        )
        df_fews = add_columns(df_fews, "FewsNet")
        df_fewss = df_fews[["date", "Source"] + adm_cols + pop_cols + ipc_cols]

    if os.path.exists(processed_globalipc_path):
        df_gipc = pd.read_csv(processed_globalipc_path, index_col=0)
        df_gipc = add_columns(df_gipc, "GlobalIPC")
        df_gipcs = df_gipc[["date", "Source"] + adm_cols + pop_cols + ipc_cols]

    if df_fewss is not None and df_gipcs is not None:
        df_comb = pd.concat([df_fewss, df_gipcs])
        df_comb_trig = compute_trigger(df_comb)

    elif df_fewss is not None:
        df_comb = df_fewss
        df_comb_trig = compute_trigger(df_comb)
        logger.warning("No Global IPC data found")

    elif df_gipcs is not None:
        df_comb = df_gipcs
        df_comb_trig = compute_trigger(df_comb)
        logger.warning("No FewsNet data found")

    else:
        df_comb_trig = pd.DataFrame()
        logger.warning("No data found")

    RESULT_FOLDER = f"{country}/Data/IPC_trigger/"
    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)
    df_comb_trig.to_csv(
        f"{RESULT_FOLDER}trigger_results_admin{admin_level}.csv", index=False
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="warning")
    main(args.country_iso3.upper(), args.admin_level)
