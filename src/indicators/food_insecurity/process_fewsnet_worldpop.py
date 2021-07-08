import pandas as pd
import os
import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)

from src.indicators.food_insecurity.config import Config
from src.indicators.food_insecurity.utils import (
    parse_args,
    download_fewsnet,
    download_worldpop,
    compute_percentage_columns,
)
from src.utils_general.utils import config_logger

logger = logging.getLogger(__name__)


def retrieve_admcols(admin_level, config):
    if admin_level == 0:
        adm_cols = [config.ADMIN0_COL]
    elif admin_level == 1:
        adm_cols = [config.ADMIN0_COL, config.ADMIN1_COL]
    elif admin_level == 2:
        adm_cols = [config.ADMIN0_COL, config.ADMIN1_COL, config.ADMIN2_COL]
    else:
        logger.error(f"Admin level {admin_level} hasn't been implemented")

    return adm_cols


def fewsnet_validperiod(row):
    """
    Add the period for which FewsNet's projections are valid.
    Till 2016 FN published a report 4 times a year, where each projection period had a validity of 3 months
    From 2017 this has changed to thrice a year, where each projection period has a validity of 4 months
    Args:
        row: row of dataframe containing the date the FN data was published (i.e. CS period) as timestamp

    Returns:

    """
    # make own mapping, to be able to use mod 12 to calculate months of projections
    month_abbr = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        0: "Dec",
    }
    year = row["date"].year
    month = row["date"].month
    if year <= 2015:
        if month == 10:
            year_ml2 = year + 1
        else:
            year_ml2 = year
        period_ML1 = (
            f"{month_abbr[month]} - {month_abbr[(month + 2) % 12]} {year}"
        )
        period_ML2 = f"{month_abbr[(month + 3) % 12]} - {month_abbr[(month + 5) % 12]} {year_ml2}"
    if year > 2015:
        if month == 2:
            year_ml1 = year
            year_ml2 = year
        elif month == 6:
            year_ml1 = year
            year_ml2 = year + 1
        elif month == 10:
            year_ml1 = year + 1
            year_ml2 = year + 1
        else:
            logger.info(
                f"Period of ML1 and ML2 cannot be added for non-regular publishing date {year}-{month}. Add manually."
            )
            row["period_ML1"] = np.nan
            row["period_ML2"] = np.nan
            return row
        period_ML1 = (
            f"{month_abbr[month]} - {month_abbr[(month + 3) % 12]} {year_ml1}"
        )
        period_ML2 = f"{month_abbr[(month + 4) % 12]} - {month_abbr[(month + 7) % 12]} {year_ml2}"
    row["period_ML1"] = period_ML1
    row["period_ML2"] = period_ML2
    return row


def compute_total_admin_population(df_fews, admin_path, pop_path, config):
    # calculate total population of every admin, to use for comparison of population given by intersection of admin shape and fewsnet
    df_adm = gpd.read_file(admin_path)
    df_adm["pop"] = pd.DataFrame(
        zonal_stats(vectors=df_adm["geometry"], raster=pop_path, stats="sum")
    )["sum"]

    # calculate population per period which has an IPC phase assigned
    for period in config.IPC_PERIOD_NAMES:
        # population that has an IPC level assigned
        df_fews[f"pop_{period}"] = df_fews[
            [f"{period}_{i}" for i in range(1, 6)]
        ].sum(axis=1, min_count=1)

        # calculate total population per period, also those that don't have an ipc phase assigned
        # these can include 66 = water, 88 = parks, forests, reserves, 99 = No Data or Missing Data
        period_cols_all = [
            f"{period}_{i}" for i in [1, 2, 3, 4, 5, 66, 88, 99]
        ]
        period_cols_data = [c for c in period_cols_all if c in df_fews.columns]
        df_fews[f"pop_total_{period}"] = df_fews[period_cols_data].sum(
            axis=1, min_count=1
        )

        # there can be a slight disperancy between the fewsnet and admin shapefile. Since we take the intersection of the fewsnet and admin shapefile, some small areas might hence be lost
        # here we calculate the total population based on the admin shape, and compare it to the total population of the overlay of the admin and fewsnet shapefiles
        # if the discrepancy causes more than 5% of the population to be excluded, raise a warning
        pop_admfews = (
            df_fews[f"pop_total_{period}"].sum() / df_adm["pop"].sum() * 100
        )
        date = "".join(df_fews["date"].dt.strftime("%Y-%m").unique())
        if pop_admfews < 95:
            logger.warning(
                f"For period {period} and date {date} only {pop_admfews:.2f}% of the country's population is included in the region covered by the FewsNet shapefile"
            )

        # In some cases a large fraction of the country's population doesn't have an IPC phase assigned (mostly 99).
        # This mostly occurs if FewsNet didn't cover the country. Due to small disreperancies between the FN and admin shapefile a small fraction can then still be assigned to an IPC phase.
        # If more than 50% doesn't have a phase assigned, raise a warning
        # TODO: if perc_ipcclass is really small for the country fewsnet data, check if region fewsnet data does contain data. Not sure how to.. But for SOM 201307 ML2 it is the case that the country fewsnet data does contain data but the east-africa not. Might happen other way around as well..
        # TODO: return NaN instead of zeroes if no data
        perc_ipcclass = (
            df_fews[f"pop_{period}"].sum() / df_adm["pop"].sum() * 100
        )
        if perc_ipcclass < 50:
            logger.warning(
                f"For period {period} and date {date} only {perc_ipcclass:.2f}% of the population is assigned to an IPC class"
            )
        if perc_ipcclass > 110:
            logger.warning(
                f"For period {period} and date {date} more than 100 ({perc_ipcclass:.2f}%) of the population is assigned to an IPC class. Check your FewsNet source data and processing"
            )
    return df_fews


def assign_population_fewsnet(
    adm_cols, fews_path, adm_path, pop_path, date, period, config, parameters
):
    """
    Compute the population per IPC phase per adm2 region by using the livelihoods defined in fews_path
    Args:
        fews_path: path to the shapefile with FewsNet data
        adm_path: path to the shapefile with admin boundaries
        pop_path: path to the raster file with population data
        date: date of the data in fews_path
        period: period of FewsNet prediction: CS (current), ML1 (near-term projection) or ML2 (medium-term projection)
        adm1c: column name of the admin1 level name, in adm_path data
        adm2c: column name of the admin2 level name, in adm_path data

    Returns:
        df_gp: DataFrame with the population per IPC phase per Admin2
    """
    df_fews = gpd.read_file(fews_path)
    df_adm = gpd.read_file(adm_path)
    df_adm = df_adm.rename(
        columns={
            parameters["shp_adm0c"]: config.ADMIN0_COL,
            parameters["shp_adm1c"]: config.ADMIN1_COL,
            parameters["shp_adm2c"]: config.ADMIN2_COL,
        }
    )
    # assign fewsnet area (livelihood) per admin region in df_adm. Sometimes livelihoods cross admin boundaries, in that case overlay splits the livelihood at the admin boundaries
    # TODO: overlay takes really long to compute, but could not find a better method
    # only use geometry and period since in newer FN data there might also admin columns be present that interfere with the admin columns of df_adm
    df_fewsadm = gpd.overlay(
        df_adm, df_fews[["geometry", period]], how="intersection"
    )
    # encountered with Chad (2018-06) that all polygons were included twice in the data. Not sure if this fixes all future cases, but only keep first occurence of same polygon and IPc phase seems safe enough to do
    df_fewsadm = df_fewsadm.drop_duplicates(subset=["geometry", period])
    # calculate population per "geometry", where geometry is the area of a livelihood within an admin zone
    # In pop_path, the value per raster cell is the population of that cell, so sum the cells within a geometry to get the total population
    # in the calculation a cell is considered to belong to an area if the center of that cell is inside the area.
    # see https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy for more explanation on the computation of zonal_stats
    df_fewsadm["pop"] = pd.DataFrame(
        zonal_stats(
            vectors=df_fewsadm["geometry"], raster=pop_path, stats="sum"
        )
    )["sum"]

    # convert the ipc phase values (1 to 5) to str
    df_fewsadm[period] = df_fewsadm[period].astype(int).astype(str)
    df_g = df_fewsadm.groupby(adm_cols + [period], as_index=False).sum()
    # set the values of ipc phases as columns (1,2,3,4,5,99)
    df_gp = df_g.pivot(index=adm_cols, columns=period, values="pop")
    df_gp = df_gp.add_prefix(f"{period}_")
    df_gp.columns.name = None
    # not all IPC levels will be present in all admin regions, so fill those with zeroes
    df_gp = df_gp.replace(np.nan, 0)
    df_gp = df_gp.reset_index()
    df_gp["date"] = pd.to_datetime(date, format="%Y%m")
    return df_gp


def process_fewsnet_worldpop(
    country,
    country_iso3,
    admin_level,
    dates,
    folder_fews,
    folder_pop,
    admin_path,
    region,
    regionabb,
    country_iso2,
    result_folder,
    suffix,
    config,
    parameters,
):
    """Retrieve all FewsNet data, and calculate the population per IPC phase
    per date-admin combination The results are saved to a csv, one containing
    the admin2 calculations and one the admin1.

    Args:
        dates: list of dates for which FewsNet data should be included
        folder_fews: path to folder that contains the FewsNet data
        folder_pop: path to folder that contains the population data
        admin_path: path to the shapefile with admin boundaries
        shp_adm1c: column name of the admin1 level name, in adm_path data
        shp_adm2c: column name of the admin2 level name, in adm_path data
        region: region that the fewsnet data covers, e.g. "east-africa"
        regionabb: abbreviation of the region that the fewsnet data covers, e.g. "EA"
        iso2_code: iso2 code of the country of interest
        result_folder: path to folder to which to save the output
        suffix: string to attach to the output files name
    """
    df = gpd.GeoDataFrame()
    adm_cols = retrieve_admcols(admin_level, config)
    # initialize progress bar
    pbar = tqdm(dates)
    # loop over dates
    for d in pbar:
        df_fews_list = []
        for period in config.IPC_PERIOD_NAMES:
            pbar.set_description(f"Processing date {d}, period {period}")
            # path to fewsnet data
            # sometimes fewsnet publishes per region, sometimes per country
            # if the country file exists, we prefer that one since it sometimes contains more detailed livelihood zones
            fews_path = None
            fews_region_path = os.path.join(
                folder_fews,
                f"{config.FEWSNET_FILENAME.format(region=region.lower(),regionabb=regionabb.upper(),date=d,period=period.upper())}",
            )
            fews_country_path = os.path.join(
                folder_fews,
                f"{config.FEWSNET_FILENAME.format(region=country_iso2.upper(),regionabb=country_iso2.upper(),date=d,period=period.upper())}",
            )
            if os.path.exists(fews_country_path):
                fews_path = fews_country_path
            elif os.path.exists(fews_region_path):
                fews_path = fews_region_path
            else:
                logger.warning(
                    f"FewsNet file for {d} and {period} not found. Skipping to next date and period."
                )

            # path to population data
            pop_path = f"{folder_pop}/{config.WORLDPOP_FILENAME.format(country_iso3=country_iso3,year=d[:4])}"
            # bit of ugly fix, but use worldpop data of previous year if that of current year does not exist
            if not os.path.exists(pop_path):
                pop_path = f"{folder_pop}/{config.WORLDPOP_FILENAME.format(country_iso3=country_iso3,year=int(d[:4])-1)}"
            if os.path.exists(pop_path) and fews_path:
                df_fews = assign_population_fewsnet(
                    adm_cols,
                    fews_path,
                    admin_path,
                    pop_path,
                    d,
                    period,
                    config,
                    parameters,
                )
                df_fews_list.append(df_fews)

            elif not os.path.exists(pop_path):
                logger.warning(
                    f"Worldpop file for {d} not found. Skipping to next date"
                )

        # if data exists for date d
        if df_fews_list:
            # concat the dfs of the different periods, with an unique entry per date-adm1-adm2 combination
            df_listind = [
                df.set_index(adm_cols + ["date"]) for df in df_fews_list
            ]
            df_comb = pd.concat(df_listind, axis=1).reset_index()

            # add ipc phase columns that are not present in the data (commonly phase 5 columns)
            ipc_cols = [
                f"{period}_{i}"
                for period in config.IPC_PERIOD_NAMES
                for i in range(1, 6)
            ]
            for i in ipc_cols:
                if i not in df_comb.columns:
                    df_comb[i] = 0

            df_comb = compute_total_admin_population(
                df_comb, admin_path, pop_path, config
            )
            df_comb = compute_percentage_columns(df_comb, config)

            df = df.append(df_comb, ignore_index=True)

    if not df.empty:
        df = df.apply(fewsnet_validperiod, axis=1)
        df = df.sort_values(by="date")
        df.to_csv(
            os.path.join(
                result_folder,
                config.FEWSWORLDPOP_PROCESSED_FILENAME.format(
                    country=country.lower(),
                    admin_level=str(admin_level),
                    suffix=suffix,
                ),
            ),
            index=False,
        )

    else:
        logger.warning("No data found for the given dates")


def retrieve_fewsnet_worldpop(
    country, admin_level, suffix="", download=False, config=None
):
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
    parameters = config.parameters(country)

    country_iso2 = parameters["iso2_code"]
    country_iso3 = parameters["iso3_code"]
    region = parameters["foodinsecurity"]["region"]
    regioncode = parameters["foodinsecurity"]["regioncode"]

    fewsnet_dates = config.FEWSNET_DATES
    if "fewsnet_dates_add" in parameters["foodinsecurity"].keys():
        fewsnet_dates = (
            fewsnet_dates + parameters["foodinsecurity"]["fewsnet_dates_add"]
        )
    if "fewsnet_dates_remove" in parameters.keys():
        fewsnet_dates = list(
            set(fewsnet_dates)
            - set(parameters["foodinsecurity"]["fewsnet_dates_remove"])
        )

    country_data_raw_dir = os.path.join(
        config.DATA_PUBLIC_RAW_DIR, parameters["iso3_code"].lower()
    )
    glb_data_raw_dir = os.path.join(config.DATA_PUBLIC_RAW_DIR, "glb")
    country_data_processed_dir = os.path.join(
        config.DATA_PUBLIC_PROCESSED_DIR, parameters["iso3_code"].lower()
    )

    fewsnet_raw_dir = os.path.join(glb_data_raw_dir, config.FEWSNET_DIR)
    worldpop_dir = os.path.join(country_data_raw_dir, config.WORLDPOP_DIR)

    if download:
        for d in fewsnet_dates:
            download_fewsnet(
                d, country_iso2, region, regioncode, fewsnet_raw_dir
            )
        years = [x[:4] for x in fewsnet_dates]
        years_unique = set(years)
        for y in years_unique:
            download_worldpop(country_iso3, y, worldpop_dir, config)

    adminbound_path = os.path.join(
        country_data_raw_dir,
        config.SHAPEFILE_DIR,
        parameters[f"path_admin{admin_level}_shp"],
    )
    output_dir = os.path.join(
        country_data_processed_dir, config.FEWSWORLDPOP_PROCESSED_DIR
    )
    # create output dir if it doesn't exist yet
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    process_fewsnet_worldpop(
        country,
        country_iso3,
        admin_level,
        fewsnet_dates,
        fewsnet_raw_dir,
        worldpop_dir,
        adminbound_path,
        region,
        regioncode,
        country_iso2,
        output_dir,
        suffix,
        config,
        parameters,
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    retrieve_fewsnet_worldpop(
        args.country.lower(),
        int(args.admin_level),
        args.suffix,
        args.download_data,
    )
