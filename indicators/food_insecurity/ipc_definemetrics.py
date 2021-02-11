import pandas as pd
from pathlib import Path
import logging
import numpy as np
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.utils import parse_args
from utils_general.utils import config_logger

logger = logging.getLogger(__name__)

# TODO: check that all cols, so CS ML1 ML2 1 to 5 and pop cols are present in input data
# TODO: quality check that perc cols add up to 100

def define_trigger_percentage(row, period, level, perc):
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

def define_trigger_increase_rel(row, level, perc):
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

def define_trigger_increase(row, period, level, perc):
    """
    Return 1 for "row", if the expected increase in the percentage of the population in "level" or higher at time "period" compared to currently (CS) is expected to be larger than "perc"
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

def main(country, admin_level, suffix, config=None):
    #TODO: now keeping for reference but remove or adjust in future
    """
    Compute all functions to return one dataframe with processed columns and if trigger is met for each data-source combination
    Args:
        country_iso3: string with iso3 code
        suffix: string to attach to the output files name
        admin_level: integer indicating which admin level to aggregate to
        suffix: string that is attached to the input file names and will be attached to the output file names
        config_file: path to config file
    """
    if config is None:
        config = Config()
    parameters = config.parameters(country)
    COUNTRY_FOLDER = f"../../analyses/{country}"

    FEWS_PROCESSED_FOLDER = f"{COUNTRY_FOLDER}/Data/FewsNetProcessed/"
    GIPC_PROCESSED_FOLDER = f"{COUNTRY_FOLDER}/Data/GlobalIPCProcessed/"
    processed_fews_path = (
        f"{FEWS_PROCESSED_FOLDER}{country}_fewsnet_admin{admin_level}{suffix}.csv"
    )
    processed_globalipc_path = (
        f"{GIPC_PROCESSED_FOLDER}{country}_globalipc_ADMIN{admin_level}{suffix}.csv"
    )

    RESULT_FOLDER = f"{COUNTRY_FOLDER}/Data/IPC_trigger/"
    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

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
        # TODO: adjust column names in process_fewsnet_subnatpop.py instead
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

    df_comb_trig.to_csv(
        f"{RESULT_FOLDER}trigger_results_admin{admin_level}{suffix}.csv", index=False
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="warning")
    main(args.country.lower(), args.admin_level, args.suffix)
