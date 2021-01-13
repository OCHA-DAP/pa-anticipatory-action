import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.utils import parse_args, compute_percentage_columns
from utils_general.utils import config_logger, download_url

logger = logging.getLogger(__name__)

def compare_adm_names(df,country,config,parameters,admin_level):
    """
    check if names in df and boundaries file of country on admin_level match
    This is purely meant as a check and not required to produce a full match
    Args:
        df (pd.DataFrame): dataframe containing the data
        country (str): name of country of interest
        config (Config): Config class, if None initialize empty one
        parameters (dict): dict with parameters parsed from config
        admin_level (int): integer indicating which admin level to aggregate to
    """
    bound_file = parameters[f"path_admin{admin_level}_shp"]
    bound_path = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country, config.DATA_DIR, config.SHAPEFILE_DIR, bound_file)

    boundaries = gpd.read_file(bound_path)
    bound_admc = parameters[f"shp_adm{admin_level}c"]

    # Check that admin level names in the IPC data are all reasonable
    missing_gipcbound = np.setdiff1d(
        list(df[f"ADMIN{admin_level}"].dropna()),
        list(boundaries[bound_admc].dropna()),
    )
    if missing_gipcbound.size > 0:
        logger.warning(
            f"The following admin {admin_level} regions from the Global IPC file are not found "
            f"in the boundaries file. You can add a mapping in the globalipc_adm_mapping parameter of the country's config file."
            f" {missing_gipcbound}"
        )

    missing_boundgipc = np.setdiff1d(
        list(boundaries[bound_admc].dropna()),
        list(df[f"ADMIN{admin_level}"].dropna()),
    )
    if missing_boundgipc.size > 0:
        logger.warning(
            f"The following admin {admin_level} regions from the boundaries file are not found "
            f"in the Global IPC file.You can add a mapping in the globalipc_adm_mapping parameter of the country's config file."
            f" {missing_boundgipc}"
        )

def aggregate_adminlevel(df_ipc,admin_level,country,config):
    """
    Aggregate df_ipc to admin_level
    Args:
        df_ipc (pd.Dataframe): dataframe containing the data and the admin_level column
        admin_level (int): integer indicating which admin level to aggregate to
        country (str): name of country of interest
        config (Config): Config class, if None initialize empty one

    Returns:
        df_ipc_agg (pd.DataFrame): dataframe with data aggregated to admin_level
    """
    if admin_level == 0:
        #TODO: decide on method of admin0 aggregation
        #data of countries we have worked so far with enables us to aggregate to admin0 level with two methods
        #one is to take all the adm2 values and group them by date
        #the other is to use the already aggregated numbers on admin0 level that are present in the raw data
        #in theory they should give the same results, but in practice they dont... (for MWI and SOM)
        #TODO: aggregate from admin2 and use the other two methods only to raise warning if mismatch
        df_ipc_adm0_group = df_ipc[df_ipc[config.ADMIN0_COL].str.lower().str.fullmatch(country.lower())].groupby([config.ADMIN0_COL,"date"],as_index=False).sum()
        df_ipc_adm0_precalc = df_ipc[df_ipc[config.ADMIN0_COL].str.lower().str.match(f"{country.lower()}:")]
        #TODO: remove once aggregation method has been decided
        print(df_ipc_adm0_precalc[["ADMIN0","date","CS_1"]].sort_values("date"))
        print(df_ipc_adm0_group[["ADMIN0","date","CS_1"]].sort_values("date"))
        df_ipc_agg = df_ipc_adm0_precalc

    elif admin_level == 1:
        #TODO: aggregate from admin2 and use the other method only to raise warning if mismatch
        df_ipc_agg = df_ipc.groupby([config.ADMIN1_COL, "date"], as_index=False).sum()
        df_ipc_agg[config.ADMIN0_COL] = country

    elif admin_level == 2:
        df_ipc_agg = df_ipc.groupby(["date", config.ADMIN1_COL, config.ADMIN2_COL], dropna=False, as_index=False).sum()
        df_ipc_agg[config.ADMIN0_COL] = country
    else:
        logger.error(f"Admin level {admin_level} has not been implemented")

    return df_ipc_agg

def compute_population_admin(df,admin_level,config):
    print(df[["date","reported_pop_CS","reported_pop_ML1","reported_pop_ML2"]])
    for period in config.IPC_PERIOD_NAMES:
        #with min_count=1 NaN is returned instead of 0 if all values are NaN
        df[f"pop_{period}"]=df[[f"{period}_{i}" for i in range(1, 6)]].sum(axis=1,min_count=1)
        df_notmatch = df[abs((df[f"pop_{period}"]-df[f"reported_pop_{period}"])/df[f"reported_pop_{period}"])>0.05]
        if not df_notmatch.empty:
            dateadm_notmatch = df_notmatch.set_index(["date",f"ADMIN{admin_level}"]).index.unique()
            dateadm_notmatch = dateadm_notmatch.sort_values()
            # df_notmatch_numbers =
            cols = ["date",f"ADMIN{admin_level}",f"pop_{period}",f"reported_pop_{period}"]
            print([",".join("{}:{}".format(*t) for t in zip(cols, row)) for _, row in df_notmatch[cols].iterrows()])
            dateadm_notmatch_str = ",".join([f"[{da[0].strftime('%m-%Y')},{da[1]}]" for da in dateadm_notmatch])
            logger.warning(f"{period}: Not matching population numbers on the following date-admin{admin_level} combinations: {dateadm_notmatch_str}")
    # neg_dates = df[df[c] < 0].index.unique()
    # neg_dates = neg_dates.sort_values()
    # neg_dates_str = ",".join([n.strftime("%d-%m-%Y") for n in neg_dates])
    # logger.warning(f'{data_name}: Negative value in column {c} on {neg_dates_str}')
    return df


def download_globalipc(country,config,parameters,output_dir):
    #create directory if doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    min_year=2010 #first year to retrieve data for. Doesn't matter if global ipc only started including data for later years
    max_year=datetime.now().year #last date to retrieve data for. Doesn't matter if this is in the future
    url = config.GLOBALIPC_URL.format(min_year=min_year,max_year=max_year,country_iso2=parameters["iso2_code"])
    output_file=os.path.join(output_dir, config.GLOBALIPC_FILENAME_RAW.format(country=country))
    #have one file with all data, so also download if file already exists to make sure it contains the newest data (contrary to fewsnet)
    try:
        download_url(url, output_file)
    except Exception:
        logger.warning(f"Cannot download GlobalIPC data for {parameters['iso3_code']}")

def process_globalipc(country, admin_level, config, parameters,ipc_dir):
    """
    Process the global ipc data and aggregate to admin_level
    Args:
        country (str): name of country of interest
        admin_level (int): integer indicating which admin level to aggregate to
        config (Config): Config class, if None initialize empty one
        parameters (dict): dict with parameters parsed from config
        ipc_dir (str): absolute path to directory containing the raw ipc data

    Returns:
        df_ipc_agg (pd.DataFrame): dataframe with the processed data and aggregated to admin_level
    """
    #TODO: include analysis period

    # ipc file columns are always on line 11
    df_ipc = pd.read_excel(os.path.join(ipc_dir,config.GLOBALIPC_FILENAME_RAW.format(country=country)), header=[11])
    # ipc excel file comes with horrible column names, so change them to better understandable ones
    df_ipc = df_ipc.rename(columns=config.GLOBALIPC_COLUMNNAME_MAPPING)
    #might be read as float
    df_ipc['ADMIN2_ID']=df_ipc['ADMIN2_ID'].astype(str)
    #due to excel settings, the percentages are on the 0 to 1 scale so change to 0-100
    perc_cols = [c for c in df_ipc.columns if 'perc' in c]
    df_ipc[perc_cols]=df_ipc[perc_cols]*100
    df_ipc["date"] = pd.to_datetime(df_ipc["date"])

    #write to file such that user can check if column names are correct
    df_ipc.to_excel(os.path.join(ipc_dir,config.GLOBALIPC_FILENAME_NEWCOLNAMES.format(country=country)))

    # remove rows with nan date and nan adm value
    df_ipc = df_ipc[
        (df_ipc["date"].notnull()) & (df_ipc[f"ADMIN{admin_level}"].notnull())
        ]


    if len(df_ipc[f"ADMIN{admin_level}"].dropna().unique()) == 0:
        logger.warning(f"No admin {admin_level} regions found in the IPC file")

    df_ipc_agg = aggregate_adminlevel(df_ipc,admin_level,country,config)
    print(df_ipc[df_ipc.date=="2020-10-01"][["date","reported_pop_CS","reported_pop_ML1","reported_pop_ML2"]])
    df_ipc_agg = compute_population_admin(df_ipc_agg,admin_level,config)
    #recompute the percentage columns.
    # Already included in the ipc data but these are rounded numbers so want the unrounded numbers based on the raw population numbers
    df_ipc_agg = compute_percentage_columns(df_ipc_agg, config)

    #remove columns that haven't been processed
    ipc_cols = [f"{period}_{i}" for period in config.IPC_PERIOD_NAMES for i in [1, 2, 3, 4, 5]]
    pop_cols = [f"pop_{period}" for period in config.IPC_PERIOD_NAMES]
    adm_cols = [f"ADMIN{a}" for a in range(0, int(admin_level) + 1)]
    df_ipc_agg = df_ipc_agg[["date"] + adm_cols + ipc_cols + pop_cols]

    # replace values in ipc df
    # mainly about differently spelled admin regions
    if "globalipc_adm_mapping" in parameters["foodinsecurity"].keys():
        globalipc_bound_mapping = parameters["foodinsecurity"]["globalipc_adm_mapping"]
        df_ipc_agg = df_ipc_agg.replace(globalipc_bound_mapping)
    #give warning on not matching adm names global ipc and boundaries file
    compare_adm_names(df_ipc_agg,country,config,parameters,admin_level)

    # TODO: idea to also add total population per admin region, now only have population per admin that was included in the IPC analysis
    #  Would have to use WorldPop data for that, see process_fewsnet_worldpop.py

    return df_ipc_agg

#TODO: not sure if this function is useful or integrate it with process_globalipc
def retrieve_globalipc(country, admin_level, suffix="", download=False, config=None):
    """
    Retrieve the globalipc data and save it to a csv file
    Args:
        country (str): name of country of interest
        admin_level (int): integer indicating which admin level to aggregate to
        suffix (str): string to attach to the output file name
        download (bool): if True, download Global IPC data
        config (Config): Config class, if None initialize empty one
    """

    if config is None:
        config = Config()
    parameters = config.parameters(country)

    country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
    globalipc_dir = os.path.join(country_folder, config.DATA_DIR, config.GLOBALIPC_RAW_DIR)
    output_dir = os.path.join(country_folder, config.DATA_DIR, config.GLOBALIPC_PROCESSED_DIR)
    # create output dir if it doesn't exist yet
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if download:
        download_globalipc(country,config,parameters,globalipc_dir)

    if os.path.exists(os.path.join(globalipc_dir,config.GLOBALIPC_FILENAME_RAW.format(country=country))):
        df_ipc = process_globalipc(country, int(admin_level), config, parameters, globalipc_dir)
        df_ipc.to_csv(os.path.join(output_dir,config.GLOBALIPC_FILENAME_PROCESSED.format(country=country, admin_level=admin_level,suffix=suffix)))
    else:
        logger.error(f"File doesn't exist at path {os.path.join(globalipc_dir,config.GLOBALIPC_FILENAME_RAW.format(country=country))}. Download the data first by using the -d args")


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    retrieve_globalipc(args.country.lower(), args.admin_level, args.suffix, args.download_data)