import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from utils import parse_args, parse_yaml, config_logger

logger = logging.getLogger(__name__)


def read_ipcglobal(parameters, ipc_path, shp_path, admin_level):
    """
    Process ipc data and do some checks
    Args:
        parameters: dict with parameters parsed from config
        ipc_path: path to ipc data
        shp_path: path to shapefile
        admin_level: integer indicating which admin level to aggregate to
    Returns:
        df_ipc: DataFrame with processed ipc data
    """

    # TODO: now assuming column names are already changed in the excel file. Might want to add something to change them automatically but fileformat is rather hard
    # seems ipc file columns are always on line 11
    df_ipc = pd.read_excel(ipc_path, header=[11])
    # remove rows with nan date
    df_ipc = df_ipc[
        (df_ipc["date"].notnull()) & (df_ipc[f"ADMIN{admin_level}"].notnull())
    ]

    # replace values in ipc df
    # mainly about differently spelled admin regions
    if "replace_dict" in parameters:
        replace_dict = parameters["replace_dict"]
        df_ipc = df_ipc.replace(replace_dict)

    if len(df_ipc[f"ADMIN{admin_level}"].dropna().unique()) == 0:
        logger.warning(f"No admin {admin_level} regions found in the IPC file")

    # TODO: implement other adm levels
    if admin_level == 1:
        df_ipc_agg = df_ipc.groupby(["ADMIN1", "date"], as_index=False).sum()
    elif admin_level == 2:
        df_ipc_agg = df_ipc.groupby(["date", "ADMIN1", "ADMIN2"], as_index=False).sum()
    else:
        df_ipc_agg = df_ipc.copy()

    ipc_cols = [
        f"{period}_{i}" for period in ["CS", "ML1", "ML2"] for i in [1, 2, 3, 4, 5]
    ]
    pop_cols = [f"pop_{period}" for period in ["CS", "ML1", "ML2"]]
    # TODO: add ADMIN0
    adm_cols = [f"ADMIN{a}" for a in range(1, int(admin_level) + 1)]

    df_ipc_agg = df_ipc_agg[["date"] + adm_cols + ipc_cols + pop_cols]
    # TODO: implement getting population per admin region, already implemented in proces_fewsnet.py
    df_ipc_agg[f"pop_ADMIN{admin_level}"] = np.nan

    shp_admc = parameters[f"shp_adm{admin_level}c"]
    boundaries = gpd.read_file(shp_path)

    # Check that admin level names in the IPC data are all reasonable
    misspelled_names = np.setdiff1d(
        list(df_ipc_agg[f"ADMIN{admin_level}"].dropna()),
        list(boundaries[shp_admc].dropna()),
    )
    if misspelled_names.size > 0:
        logger.warning(
            f"The following admin {admin_level} regions from the IPC file are not found "
            f"in the boundaries file: {misspelled_names}"
        )

    return df_ipc_agg


def main(country_iso3, admin_level, suffix, config_file="config.yml"):
    """
    Define variables and save output
    Args:
        country_iso3: string with iso3 code
        admin_level: integer indicating which admin level to aggregate to
        config_file: path to config file
        suffix: string to attach to the output files name
    """
    parameters = parse_yaml(config_file)[country_iso3]
    country = parameters["country_name"]
    admin2_shp = parameters["path_admin2_shp"]
    GLOBALIPC_PATH = parameters["ipc_path"]

    COUNTRY_FOLDER = f"../../analyses/{country}"
    SHP_PATH = f"{COUNTRY_FOLDER}/Data/{admin2_shp}"
    IPC_PATH = f"{COUNTRY_FOLDER}/Data/{GLOBALIPC_PATH}"
    RESULT_FOLDER = f"{COUNTRY_FOLDER}/Data/GlobalIPCProcessed/"
    # create output dir if it doesn't exist yet
    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

    df_ipc = read_ipcglobal(parameters, IPC_PATH, SHP_PATH, admin_level)
    df_ipc.to_csv(f"{RESULT_FOLDER}{country}_globalipc_ADMIN{admin_level}{suffix}.csv")


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="warning")
    main(args.country_iso3.upper(), args.admin_level, args.suffix)
