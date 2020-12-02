import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.utils import parse_args, config_logger, get_globalipc_data

logger = logging.getLogger(__name__)


def read_ipcglobal(country_iso3, parameters, ipc_path, ipc_dir, shp_path, admin_level,config):
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

    # ipc file columns are always on line 11
    df_ipc = pd.read_excel(ipc_path, header=[11])
    # ipc excel file comes with horrible column names, so change them to better understandable ones
    df_ipc = df_ipc.rename(columns=config.GLOBALIPC_COLUMNNAME_MAPPING)
    #due to excel settings, the percentages are on the 0 to 1 scale so change to 0-100
    perc_cols = [c for c in df_ipc.columns if 'perc' in c]
    df_ipc[perc_cols]=df_ipc[perc_cols]*100

    #write to file such that user can check if column names are correct
    df_ipc.to_excel(os.path.join(ipc_dir,f"{country_iso3}_globalipc_newcolumnnames.xlsx"))
    # remove rows with nan date
    df_ipc = df_ipc[
        (df_ipc["date"].notnull()) & (df_ipc[f"ADMIN{admin_level}"].notnull())
    ]

    # replace values in ipc df
    # mainly about differently spelled admin regions
    if "globalipc_bound_mapping" in parameters:
        globalipc_bound_mapping = parameters["globalipc_bound_mapping"]
        df_ipc = df_ipc.replace(globalipc_bound_mapping)

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


def main(country_iso3, admin_level, suffix, download, config=None):
    """
    Define variables and save output
    Args:
        country_iso3: string with iso3 code
        admin_level: integer indicating which admin level to aggregate to
        config_file: path to config file
        suffix: string to attach to the output files name
    """
    if config is None:
        config = Config()
    parameters = config.parameters(country_iso3)
    # parameters = parse_yaml(config_file)[country_iso3]
    country = parameters["country_name"]
    country_iso2 = parameters["iso2_code"]
    admin2_shp = parameters["path_admin2_shp"]

    COUNTRY_FOLDER = f"../../analyses/{country}"
    SHP_PATH = f"{COUNTRY_FOLDER}/Data/{admin2_shp}"
    IPC_DIR = f"{COUNTRY_FOLDER}/Data/{config.GLOBALIPC_DIR}" #{GLOBALIPC_PATH}"
    IPC_PATH = os.path.join(IPC_DIR,config.GLOBALIPC_FILENAME.format(country_iso3=country_iso3))
    RESULT_FOLDER = f"{COUNTRY_FOLDER}/Data/GlobalIPCProcessed/"
    # create output dir if it doesn't exist yet
    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

    if download:
        get_globalipc_data(country_iso3, country_iso2, IPC_DIR, config)

    df_ipc = read_ipcglobal(country_iso3, parameters, IPC_PATH, IPC_DIR, SHP_PATH, admin_level,config)
    df_ipc.to_csv(f"{RESULT_FOLDER}{country}_globalipc_admin{admin_level}{suffix}.csv")


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country_iso3.upper(), args.admin_level, args.suffix, args.download_data)