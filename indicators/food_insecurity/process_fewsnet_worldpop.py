import pandas as pd
import os
import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.food_insecurity.config import Config
from indicators.food_insecurity.utils import parse_args, config_logger, get_fewsnet_data, get_worldpop_data

logger = logging.getLogger(__name__)


def merge_fewsnet_population(fews_path, adm_path, pop_path, date, period, adm1c, adm2c):
    """
    Compute the population per IPC phase per adm2 region for the data defined in fews_path
    Args:
        fews_path: path to the shapefile with FewsNet data
        adm_path: path to the shapefile with admin2 boundaries
        pop_path: path to the raster file with population data
        date: date of the data defined in fews_path
        period: type of FewsNet prediction: CS (current), ML1 (near-term projection) or ML2 (medium-term projection)
        adm1c: column name of the admin1 level name, in adm_path data
        adm2c: column name of the admin2 level name, in adm_path data

    Returns:
        df_gp: DataFrame with the population per IPC phase per Admin2
    """
    df_fews = gpd.read_file(fews_path)
    df_adm = gpd.read_file(adm_path)
    # get fewsnet area (livelihood) per admin region in df_adm (generally admin2)
    # overlay takes really long to compute, but could not find a better method
    df_fewsadm = gpd.overlay(df_adm, df_fews, how="intersection")

    # calculate population per "geometry"
    # in pop_path, the value per cell is the population of that cell, so we want the sum of them
    # in the calculation a cell is considered to belong to an area if the center of that cell is inside the area.
    # see https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy
    df_fewsadm["pop"] = pd.DataFrame(
        zonal_stats(vectors=df_fewsadm["geometry"], raster=pop_path, stats="sum")
    )["sum"]

    # convert the period values (1 to 5) to str
    df_fewsadm[period] = df_fewsadm[period].astype(int).astype(str)
    df_g = df_fewsadm.groupby([adm1c, adm2c, period], as_index=False).sum()
    # set the values of period as columns (1,2,3,4,5,99)
    df_gp = df_g.pivot(index=[adm1c, adm2c], columns=period, values="pop")
    df_gp = df_gp.add_prefix(f"{period}_")
    df_gp.columns.name = None
    # not all IPC levels will be present in all admin regions, so fill those with zeroes
    df_gp = df_gp.replace(np.nan, 0)
    df_gp = df_gp.reset_index()
    df_gp["date"] = pd.to_datetime(date, format="%Y%m")
    return df_gp


def combine_fewsnet_projections(
    country_iso3,
    dates,
    folder_fews,
    folder_pop,
    admin_path,
    shp_adm1c,
    shp_adm2c,
    region,
    regionabb,
    country_iso2,
    result_folder,
    suffix,
    config
):
    """
    Retrieve all FewsNet data, and calculate the population per IPC phase per date-admin combination
    The results are saved to a csv, one containing the admin2 calculations and one the admin1.
    Args:
        country_iso3: string with iso3 code
        dates: list of dates for which FewsNet data should be included
        folder_fews: path to folder that contains the FewsNet data
        folder_pop: path to folder that contains the population data
        admin_path: path to the shapefile with admin2 boundaries
        shp_adm1c: column name of the admin1 level name, in adm_path data
        shp_adm2c: column name of the admin2 level name, in adm_path data
        region: region that the fewsnet data covers, e.g. "east-africa"
        regionabb: abbreviation of the region that the fewsnet data covers, e.g. "EA"
        iso2_code: iso2 code of the country of interest
        result_folder: path to folder to which to save the output
        suffix: string to attach to the output files name
    """
    # all periods in the FewsNet data
    period_list = ["CS", "ML1", "ML2"]
    df = gpd.GeoDataFrame()
    # initialize progress bar
    pbar = tqdm(dates)
    # loop over dates
    for d in pbar:
        df_fews_list = []
        for period in period_list:
            pbar.set_description(f"Processing date {d}, period {period}")
            # path to fewsnet data
            # sometimes fewsnet publishes per region, sometimes per country
            fews_path = None
            fews_region_path = f"{folder_fews}{config.FEWSNET_FILENAME.format(region=region.lower(),regionabb=regionabb.upper(),date=d,period=period.upper())}"
            fews_country_path = f"{folder_fews}{config.FEWSNET_FILENAME.format(region=country_iso2.upper(),regionabb=country_iso2.upper(),date=d,period=period.upper())}"

            if os.path.exists(fews_country_path):
                fews_path = fews_country_path
            elif os.path.exists(fews_region_path):
                fews_path = fews_region_path


            # path to population data
            pop_path = f"{folder_pop}/{config.WORLDPOP_FILENAME.format(country_iso3=country_iso3,year=d[:4])}" #_1km_Aggregated_UNadj#{country_iso3.lower()}_ppp_{d[:4]}_UNadj.tif"

            if fews_path and os.path.exists(pop_path):
                df_fews = merge_fewsnet_population(
                    fews_path, admin_path, pop_path, d, period, shp_adm1c, shp_adm2c
                )
                df_fews_list.append(df_fews)
            elif not fews_path:
                logger.warning(
                    f"FewsNet file for {d} and {period} not found. Skipping to next date and period."
                )
            elif not os.path.exists(pop_path):
                logger.warning(
                    f"Worldpop file for {d} not found. Skipping to next date"
                )

        if df_fews_list:
            # concat the dfs of the different "periods", with an unique entry per date-adm1-adm2 combination
            df_listind = [
                df.set_index([shp_adm1c, shp_adm2c, "date"]) for df in df_fews_list
            ]
            df_comb = pd.concat(df_listind, axis=1).reset_index()

            # add ipc level columns that are not present in the data (commonly level 5 columns)
            ipc_cols = [f"{period}_{i}" for period in period_list for i in range(1, 6)]
            for i in ipc_cols:
                if i not in df_comb.columns:
                    df_comb[i] = 0

            # calculate total population of every admin, to use for comparison of population given by intersection of admin shape and fewsnet
            df_adm = gpd.read_file(admin_path)
            df_adm["pop"] = pd.DataFrame(
                zonal_stats(vectors=df_adm["geometry"], raster=pop_path, stats="sum")
            )["sum"]

            # calculate population per period over all IPC levels
            for period in period_list:
                # population that has an IPC level assigned
                df_comb[f"pop_{period}"] = df_comb[
                    [f"{period}_{i}" for i in range(1, 6)]
                ].sum(axis=1, min_count=1)

                # all columns related to the period, also areas that don't have an ipc phase assigned.
                # these can include 66 = water, 88 = parks, forests, reserves, 99 = No Data or Missing Data
                period_cols_all = [f"{period}_{i}" for i in [1, 2, 3, 4, 5, 66, 88, 99]]
                period_cols_data = [c for c in period_cols_all if c in df_comb.columns]
                df_comb[f"pop_Total_{period}"] = df_comb[period_cols_data].sum(
                    axis=1, min_count=1
                )

                # there can be a slight disperancy between the fewsnet and admin shapefile. Since we take the intersection, some areas might then be lost
                # here we calculate the total population based on the admin shape, and compare it to the total population of the overlay of the admin and fewsnet shapefiles
                # if the disperancy causes more than 5% of the population to be excluded, raise a warning
                pop_admfews = (
                    df_comb[f"pop_Total_{period}"].sum() / df_adm["pop"].sum() * 100
                )

                if pop_admfews < 95:
                    logger.warning(
                        f"For date {d} and period {period} only {pop_admfews:.2f}% of the country's population is included in the region covered by the FewsNet shapefile"
                    )

                # it can also be the case that FewsNet and Admin shapefile cover the same region
                # but that FewsNet hasn't assigned a phase to a large part of the population
                # if more than 50% doesn't have a phase assigned, raise a warning
                #TODO: if perc_ipcclass is really small for the country fewsnet data, check if region fewsnet data does contain data. Not sure how to.. But for SOM 201307 ML2 it is the case that the country fewsnet data does contain data but the east-africa not. Might happen other way around as well..
                perc_ipcclass = (
                    df_comb[f"pop_{period}"].sum() / df_adm["pop"].sum() * 100
                )
                if perc_ipcclass < 50:
                    logger.warning(
                        f"For period {period} and date {d} only {perc_ipcclass:.2f}% of the population is assigned to an IPC class"
                    )

            df = df.append(df_comb, ignore_index=True)

    if not df.empty:
        # set general admin names
        df.rename(columns={shp_adm1c: "ADMIN1", shp_adm2c: "ADMIN2"}, inplace=True)
        # TODO: decide what kind of filename we want to use for the output, i.e. do we always want to overwrite the output or not
        df.to_csv(
            f"{result_folder}{country_iso3.lower()}_admin2_fewsnet_worldpop{suffix}.csv"
        )
        # aggregate to admin1 by summing (and set to nan if no data for a date-adm1 combination
        df_adm1 = (
            df.drop("ADMIN2", axis=1)
            .groupby(["date", "ADMIN1"])
            .agg(lambda x: np.nan if x.isnull().all() else x.sum())
            .reset_index()
        )
        df_adm1.rename(columns={"pop_ADMIN2": "pop_ADMIN1"}, inplace=True)
        df_adm1.to_csv(
            f"{result_folder}{country_iso3.lower()}_admin1_fewsnet_worldpop{suffix}.csv"
        )
    else:
        logger.warning("No data found for the given dates")


def main(country_iso3, suffix, download, config_file="config.yml", config=None):
    """
    This script computes the population per IPC phase per data - admin2 region combination.
    The IPC phase is retrieved from the FewsNet data, which publishes their data in shapefiles, of three periods namely current situation (CS), near-term projection (ML1) and mid-term projection (ML2)
    The IPC phases range from 1 to 5, and missing values are indicated by 99.
    Args:
        country_iso3: string with iso3 code
        suffix: string to attach to the output files name
        config_file: path to config file
    """

    if config is None:
        config = Config()
    parameters = config.parameters(country_iso3)
    # parameters = parse_yaml(config_file)[country_iso3]

    country = parameters["country_name"]
    COUNTRY_FOLDER = f"../../analyses/{country}"

    region = parameters["region"]
    regioncode = parameters["regioncode"]
    country_iso2 = parameters["iso2_code"]
    dates = parameters["fewsnet_dates"]
    admin2_shp = parameters["path_admin2_shp"]
    shp_adm1c = parameters["shp_adm1c"]
    shp_adm2c = parameters["shp_adm2c"]

    # TODO: to make variables more generalizable with a config.py. Inspiration from pa-covid-model-parameterization
    # pop_dir = os.path.join(config.DIR_PATH, country, config.POP_DIR)
    FOLDER_FEWSNET = "Data/FewsNetRaw/"
    FOLDER_POP = f"{COUNTRY_FOLDER}/Data/WorldPop"
    ADMIN2_PATH = f"{COUNTRY_FOLDER}/Data/{admin2_shp}"
    RESULT_FOLDER = f"{COUNTRY_FOLDER}/Data/FewsNetWorldPop/"
    # create output dir if it doesn't exist yet
    Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

    if download:
        for d in dates:
            get_fewsnet_data(d,country_iso2,region,regioncode,FOLDER_FEWSNET)
        years=[x[:4] for x in dates]
        years_unique=set(years)
        for y in years_unique:
            get_worldpop_data(country_iso3, y, FOLDER_POP, config)

    combine_fewsnet_projections(
        country_iso3,
        dates,
        FOLDER_FEWSNET,
        FOLDER_POP,
        ADMIN2_PATH,
        shp_adm1c,
        shp_adm2c,
        region,
        regioncode,
        country_iso2,
        RESULT_FOLDER,
        suffix,
        config
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country_iso3.upper(), args.suffix, args.download_data)
