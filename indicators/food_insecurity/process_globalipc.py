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

def test_mismatch_adminlevels(df_agg,df_precalc,admin_level,country, config):
    #TODO: prettify this function
    """
    There might be a mismatch between different levels of reporting on admin0 and admin1 level.
    The method we use is to compute the totals by aggregation of adm2 but raise a warning if doesn't match with reported adm0/1 numbers to make user aware
    But the numbers on admin0 and admin1 level are also reported directly
    This function compares the two methods of reporting and raises a warning if there is more than 5% difference between them
    We only raise a warning to make the user aware, but continue to use the aggregation from adm2 methodology
    Args:
        df_agg (pd.DataFrame): dataframe containing the data on admin_level aggregated from admin2
        df_precalc (pd.DataFrame): dataframe containing the data directly reported on admin_level
        admin_level (int): integer indicating admin level of interest
        country (str): name of country of interest
        config (Config): Config class
    """
    if admin_level==0:
        df_precalc = df_precalc[df_precalc[config.ADMIN0_COL].str.lower().str.match(f"{country.lower()}:")]
    elif admin_level==1:
        #assume admin1 sum is reported in the admin0 column and the rows corresponds to those that don't contain the admin0 name
        df_precalc = df_precalc[~df_precalc[config.ADMIN0_COL].str.lower().str.contains(country.lower())]
        #rename to make df_agg columns equal to df_precalc for merging
        df_agg = df_agg.drop(config.ADMIN0_COL, axis=1)
        df_agg=df_agg.rename(columns={config.ADMIN1_COL: config.ADMIN0_COL})
    #no aggregation for adm2 so tests don't make sense in that case
    else:
        return None
    if not df_precalc.empty and not df_agg.empty:
        df_merge = df_agg.merge(df_precalc, on=["date", config.ADMIN0_COL], suffixes=("_agg", "_precalc"))
        #only look at population and perc_ipc3+ for now, could add more
        match_cols = [f"pop_{p}" for p in config.IPC_PERIOD_NAMES] + [f"perc_{p}_3p" for p in config.IPC_PERIOD_NAMES]
        for c in match_cols:
            #select rows where the two methods differ by more than 5%
            if "perc" in c:
                df_notmatch = df_merge[abs(df_merge[f"{c}_agg"] - df_merge[f"{c}_precalc"]) > 5]
            else:
                df_notmatch = df_merge[abs((df_merge[f"{c}_agg"] - df_merge[f"{c}_precalc"]) / df_merge[f"{c}_precalc"]) > 0.05]
            if not df_notmatch.empty:
                #raise warning if any rows that differ by more than 5%
                dateadm_notmatch = df_notmatch.set_index(["date", config.ADMIN0_COL]).index.unique()
                dateadm_notmatch = dateadm_notmatch.sort_values()
                dateadm_notmatch_str = ",".join([f"[{da[0].strftime('%m-%Y')},{da[1]}]" for da in dateadm_notmatch])
                logger.warning(
                    f"{c} reported on admin{admin_level} and aggregated from admin2 to {admin_level} differs by more than 5% "
                    f"for the following date-admin{admin_level} combinations: {dateadm_notmatch_str}")

def aggregate_adminlevel(df_ipc,admin_level,country,config):
    """
    Aggregate df_ipc to admin_level
    Args:
        df_ipc (pd.Dataframe): dataframe containing the data and the admin_level column
        admin_level (int): integer indicating which admin level to aggregate to
        country (str): name of country of interest
        config (Config): Config class

    Returns:
        df_ipc_agg (pd.DataFrame): dataframe with data aggregated to admin_level
    """
    # we assume that the admin2 column is nan for rows where adm1/0 sums are reported.
    # Thus, this enable us to compute the total adm1/0s by summing the adm2s
    # From empirical testing it was shown that the aggregations from adm2 to adm1/0 don't always equal the reported adm1/0 sums
    # Since one methodology had to be chosen, we chose to always use the adm2 numbers as basis and aggregate from here
    # This enables us to be consistent across admin levels
    # The reasons for disagreement are not entirely clear. One cause is the inclusion of IDPs on adm1 and not on adm2, but this doesn't explain all the differences
    df_ipc_adm2=df_ipc[(df_ipc["date"].notnull()) & (df_ipc[f"ADMIN2"].notnull())]
    if admin_level == 0:
        df_ipc_agg = df_ipc_adm2.groupby("date",as_index=False).sum()
        df_ipc_agg[config.ADMIN0_COL] = country


    elif admin_level == 1:
        # we assume that the admin1_col contains the name of the adm1 region for each admin2,
        # but that the summed numbers of adm1s are given in the adm0 column and thus not in the admin1
        # thus we can groupby the admin1 col to get the sums of the adm2s per adm1
        df_ipc_agg = df_ipc_adm2.groupby([config.ADMIN1_COL, "date"], as_index=False).sum()
        df_ipc_agg[config.ADMIN0_COL] = country

    elif admin_level == 2:
        #it can occur that an adm2 name occurs in two adm1s, hence groupby adm1-adm2 combination and not only adm2
        df_ipc_agg = df_ipc_adm2.groupby(["date", config.ADMIN1_COL, config.ADMIN2_COL], dropna=False, as_index=False).sum()
        df_ipc_agg[config.ADMIN0_COL] = country
    else:
        logger.error(f"Admin level {admin_level} has not been implemented")

    return df_ipc_agg

def compute_population_admin(df,admin_level,config):
    """
    Compute the population assigned to an IPC phase per admin-date combination
    Global IPC reports the population that is analysed but this reported number doesn't always equal the sum of population over all the ipc phases
    Why there is this discrepancy, isn't entirely clear. But since the threshold are based on percentage of the population in each phase,
    we want to make sure the total analysed population equals the sum of the population of all ipc phases.
    Args:
        df (pd.DataFrame): dataframe containing the populatoin per ipc phase and the reported total population
        admin_level (int): integer indicating which admin level to aggregate to
        config (Config): Config class
    Returns:
        df (pd.DataFrame): input dataframe with column per period containing the sum of population of all IPC phases
    """

    for period in config.IPC_PERIOD_NAMES:
        #with min_count=1 NaN is returned instead of 0 if all values are NaN
        df[f"pop_{period}"]=df[[f"{period}_{i}" for i in range(1, 6)]].sum(axis=1,min_count=1)
        #raise a warning if the reported numbers and the sum over the ipc phases are not equal.
        #only do this for admin2 since we aggregate from those
        #user doesn't have to do anything with this information, but good to know as a quality indicator of the data
        if admin_level==2:
            df_notmatch = df[abs((df[f"pop_{period}"]-df[f"reported_pop_{period}"])/df[f"reported_pop_{period}"])>0.05]
            df_notmatch = df_notmatch[(df_notmatch[config.ADMIN2_COL].notnull())]
            if not df_notmatch.empty:
                dateadm_notmatch = df_notmatch.set_index(["date",f"ADMIN{admin_level}"]).index.unique()
                dateadm_notmatch = dateadm_notmatch.sort_values()
                dateadm_notmatch_str = ",".join([f"[{da[0].strftime('%m-%Y')},{da[1]}]" for da in dateadm_notmatch])
                logger.warning(f"{period}: The sum of population in the ipc phases differs from the reported total population by more than 5% "
                               f"for the following date-admin{admin_level} combinations in {period}: {dateadm_notmatch_str}")
    return df


def download_globalipc(country,config,parameters,output_dir):
    """
    Retrieve the Global IPC data from their Population Tracking Tool and save to output_file
    Args:
        country (str): name of country of interest
        config (Config): Config class
        parameters (dict): dict with parameters parsed from config
        output_dir (str): path to directory the file should be saved to
    """
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
        config (Config): Config class
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

    # remove rows with nan date, these often include textual information in other formats
    df_ipc = df_ipc[df_ipc["date"].notnull()]
    # df_ipc_adm2 = compute_population_admin(df_ipc_adm2,admin_level,config)



    #we map names but don't assign admin1's to admin2's if they are missing from the file (e.g. MWI)
    if len(df_ipc[f"ADMIN{admin_level}"].dropna().unique()) == 0:
        logger.error(f"No admin {admin_level} regions found in the IPC file")
        return None
    else:
        df_ipc = compute_population_admin(df_ipc,admin_level,config)
        df_ipc_agg = aggregate_adminlevel(df_ipc,admin_level,country,config)
        #recompute the percentage columns.
        # Already included in the ipc data but saw that these don't always match the numbers calculated based on the reported populations number, so recompute to be sure they match
        df_ipc_agg = compute_percentage_columns(df_ipc_agg, config)

        # replace values in ipc df
        # mainly about differently spelled admin regions
        if "globalipc_adm_mapping" in parameters["foodinsecurity"].keys():
            globalipc_bound_mapping = parameters["foodinsecurity"]["globalipc_adm_mapping"]
            df_ipc_agg = df_ipc_agg.replace(globalipc_bound_mapping)
        #give warning on not matching adm names global ipc and boundaries file
        #assume admin0 name is correct, cause quickly causes disrepancies due to added strings in Global IPC adm0 name
        if admin_level!=0:
            compare_adm_names(df_ipc_agg,country,config,parameters,admin_level)

        #aggregation from adm2 to adm0/1 doesn't always equal the numbers reported directly on adm0/1
        #raise a warning if this occurs
        #purely to be aware of the data quality, but can still continue to use the data based on aggregation from adm2
        test_mismatch_adminlevels(df_ipc_agg,df_ipc,admin_level,country,config)

        #remove columns that haven't been processed
        ipc_cols = [f"{period}_{i}" for period in config.IPC_PERIOD_NAMES for i in [1, 2, 3, 4, 5]]
        pop_cols = [f"pop_{period}" for period in config.IPC_PERIOD_NAMES]
        adm_cols = [f"ADMIN{a}" for a in range(0, int(admin_level) + 1)]
        perc_cols = [c for c in df_ipc_agg.columns if "perc" in c]
        df_ipc_agg = df_ipc_agg[["date"] + adm_cols + ipc_cols + perc_cols + pop_cols]

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
        if df_ipc is not None:
            df_ipc.to_csv(os.path.join(output_dir,config.GLOBALIPC_FILENAME_PROCESSED.format(country=country, admin_level=admin_level,suffix=suffix)),index=False)
    else:
        logger.error(f"File doesn't exist at path {os.path.join(globalipc_dir,config.GLOBALIPC_FILENAME_RAW.format(country=country))}. Download the data first by using the -d args")


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    retrieve_globalipc(args.country.lower(), args.admin_level, args.suffix, args.download_data)