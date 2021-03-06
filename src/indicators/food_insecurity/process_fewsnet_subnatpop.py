import geopandas as gpd
import pandas as pd
import os
import numpy as np
import logging
import sys
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import Config
from src.indicators.food_insecurity.utils import parse_args, download_fewsnet
from src.utils_general.utils import config_logger, convert_to_numeric

logger = logging.getLogger(__name__)


def shapefiles_to_df(path, period, dates, region, regionabb, iso2_code):
    """
    Compile the shapefiles to a dataframe Args: path: path to directory
    that contains the FewsNet shapefiles period: type of FewsNet
    prediction: CS (current), ML1 (near-term projection) or ML2
    (medium-term projection) dates: list of dates for which FewsNet data
    should be included region: region that the fewsnet data covers, e.g.
    "east-africa" regionabb: abbreviation of the region that the fewsnet
    data covers, e.g. "EA" iso2_code: iso2 code of the country of
    interest

    Returns: df: DataFrame that contains all the shapefiles of Fewsnet
        for the given dates, period and regions
    """
    df = gpd.GeoDataFrame()
    for d in dates:
        # path to fewsnet data In most cases FewsNet publishes per
        # region, but sometimes also per country, so allow for both
        shape_region = os.path.join(
            path, f"{region}{d}/{regionabb}_{d}_{period}.shp"
        )
        shape_country = os.path.join(
            path, f"{iso2_code}{d}/{iso2_code}_{d}_{period}.shp"
        )
        if os.path.exists(shape_country):
            gdf = gpd.read_file(shape_country)
            gdf["date"] = pd.to_datetime(d, format="%Y%m")
            df = df.append(gdf, ignore_index=True)
        elif os.path.exists(shape_region):
            gdf = gpd.read_file(shape_region)
            gdf["date"] = pd.to_datetime(d, format="%Y%m")
            df = df.append(gdf, ignore_index=True)
    return df


def merge_admin2(df, path_admin, period, adm0c, adm1c, adm2c):
    """
    Merge the geographic boundary information shapefile with the FewsNet
    dataframe. Args: df: DataFrame with the Fewsnet data and geometries
    path_admin: path to file with admin(2) boundaries period: type of
    FewsNet prediction: CS (current), ML1 (near-term projection) or ML2
    (medium-term) adm0c: column name of the admin0 level name, in
    path_admin data adm1c: column name of the admin1 level name, in
    path_admin data adm2c: column name of the admin2 level name, in
    path_admin data

    Returns: overlap: dataframe with the regions per admin2 for each IPC
        level
    """
    admin2 = gpd.read_file(path_admin)
    if not all(x in admin2.columns for x in [adm0c, adm1c, adm2c]):
        logger.warning(
            "Not all admin columns defined in the config were found in the"
            " boundary shapefile. The boundary shapefile's columns are"
            f" {list(admin2.columns)}"
        )
    admin2 = admin2[[adm0c, adm1c, adm2c, "geometry"]]
    overlap = gpd.overlay(admin2, df, how="intersection")
    overlap = overlap.drop_duplicates()
    overlap["area"] = overlap["geometry"].to_crs("EPSG:3395").area
    columns = [adm0c, adm1c, adm2c, period, "date", "geometry", "area"]
    overlap = overlap[columns]
    return overlap


def return_max_cs(date, df, dfadcol, period, adm0c, adm1c, adm2c):
    """
    Return the IPC value that is assigned to the largest area (in m2)
    for the given Admin Level 2 region It is discussable if this is the
    best approach to select the IPC admin2 level. One could also try to
    work with more local population estimates Args: date: string with
    the date of the FewsNet analysis df: DataFrame that contains the
    geometrys per IPC level per Admin2 (output from merge_admin2)
    dfadcol: one row of the df period: type of FewsNet prediction: CS
    (current), ML1 (near-term projection) or ML2 (medium-term) adm0c:
    column name of the admin0 level name, in path_admin data adm1c:
    column name of the admin1 level name, in path_admin data adm2c:
    column name of the admin2 level name, in path_admin data


    Returns: row: row of the df which has the largest area within the
        given Admin Level 2 region (defined by dfadcol)
    """
    sub = df.loc[
        (df["date"] == date)
        & (df[adm1c] == dfadcol[adm1c])
        & (df[adm2c] == dfadcol[adm2c])
    ]
    # if there are nan (=0) values we prefer to take the non-nan values,
    # even if those represent a smaller area however if there are only
    # nans (=0s) in an admin2 region, we do return one of those rows
    if len(sub[sub[period] != 0]) > 0:
        sub = sub[sub[period] != 0]
    mx = sub["area"].max()
    row = sub[["date", adm0c, adm1c, adm2c, period]].loc[sub["area"] == mx]
    return row


def add_missing_values(df, period, dates, path_admin, adm0c, adm1c, adm2c):
    """
    Add dates which are in dates but not in current dataframe (i.e. not
    in raw FewsNet data) to dataframe and set period values to nan Args:
    df: DataFrame with the max IPC for period per adm1-adm2 combination
    for all dates that are in the FewsNet data period: type of FewsNet
    prediction: CS (current), ML1 (near-term projection) or ML2
    (medium-term) dates: list of dates for which FewsNet data should be
    included. The folders of these dates have to be present in the
    directory (ipc_path)! path_admin: path to file with admin(2)
    boundaries adm0c: column name of the admin0 level name, in
    path_admin data adm1c: column name of the admin1 level name, in
    path_admin data adm2c: column name of the admin2 level name, in
    path_admin data

    Returns: DataFrame which includes the dates in "dates" that were not
        in the input df
    """
    dates_dt = pd.to_datetime(dates, format="%Y%m")
    # check if
    if df.empty:
        diff_dates = set(dates_dt)
    else:
        # get dates that are in config list but not in df
        diff_dates = set(dates_dt) - set(df.date)

    if diff_dates:
        diff_dates_string = ",".join(
            [n.strftime("%d-%m-%Y") for n in diff_dates]
        )
        logger.warning(
            f"No FewsNet data found for {period} on {diff_dates_string}"
        )
        df_adm2 = gpd.read_file(path_admin)
        df_admnames = df_adm2[[adm0c, adm1c, adm2c]]
        for d in diff_dates:
            df_date = df_admnames.copy()
            df_date["date"] = d
            df_date[period] = np.nan
            df = df.append(df_date)
    return df


def gen_csml1m2(
    ipc_path,
    bound_path,
    period,
    dates,
    adm0c,
    adm1c,
    adm2c,
    region,
    regionabb,
    iso2_code,
):
    """
    Generate a DataFrame with the IPC level per Admin 2 Level, defined
    by the level that covers the largest area The DataFrame includes all
    the dates given as input, and covers one type of classification
    given by period Args: ipc_path: path to the directory with the
    fewsnet data bound_path: path to the file with the admin2 boundaries
    period: type of FewsNet prediction: CS (current), ML1 (near-term
    projection) or ML2 (medium-term) dates: list of dates for which
    FewsNet data should be included adm0c: column name of the admin0
    level name, in path_admin data adm1c: column name of the admin1
    level name, in path_admin data adm2c: column name of the admin2
    level name, in path_admin data region: region that the fewsnet data
    covers, e.g. "east-africa" regionabb: abbreviation of the region
    that the fewsnet data covers, e.g. "EA" iso2_code: iso2 code of the
    country of interest

    Returns: new_df: DataFrame that contains one row per Admin2-date
        combination, which indicates the IPC level
    """
    df_ipc = shapefiles_to_df(
        ipc_path, period, dates, region, regionabb, iso2_code
    )
    try:
        overlap = merge_admin2(df_ipc, bound_path, period, adm0c, adm1c, adm2c)
        # replace other values than 1-5 by 0 (these are 99,88,66 and
        # indicate missing values, nature areas or lakes)
        overlap.loc[overlap[period] >= 5, period] = 0
        new_df = pd.DataFrame(columns=["date", period, adm0c, adm1c, adm2c])

        for d in overlap["date"].unique():
            # all unique combinations of admin1 and admin2 regions
            # (sometimes an admin2 region can be in two admin1 regions)
            df_adm12c = overlap[[adm1c, adm2c]].drop_duplicates()
            for index, a in df_adm12c.iterrows():
                row = return_max_cs(d, overlap, a, period, adm0c, adm1c, adm2c)
                new_df = new_df.append(row)
        new_df.replace(0, np.nan, inplace=True)
        df_alldates = add_missing_values(
            new_df, period, dates, bound_path, adm0c, adm1c, adm2c
        )

    except AttributeError:
        logger.error(
            f"No FewsNet data for {period} for the given dates was found"
        )
        df_alldates = add_missing_values(
            df_ipc, period, dates, bound_path, adm0c, adm1c, adm2c
        )
    return df_alldates


def get_new_name(name, n_dict):
    """
    Return the values of a dict if name is in the keys of the dict
    Args:
        name: string of interest
        n_dict: dict with possibly "name" as key

    Returns:

    """
    if name in n_dict.keys():
        return n_dict[name]
    else:
        return name


def merge_ipcperiod(inputdf_dict, adm0c, adm1c, adm2c):
    """
    Merge the three types of IPC projections (CS, ML1, ML2) to one
    dataframe Args: inputdf_dict: dict with df for each period (CS, ML1,
    ML2) adm1c: column name of the admin1 level name, in fewsnet data
    adm2c: column name of the admin2 level name, in fewsnet data

    Returns: df_ipc: dataframe with the cs, ml1 and ml2 data combined
    """

    df = pd.DataFrame()
    for k in inputdf_dict.keys():
        if df.empty:
            df = inputdf_dict[k]
        else:
            df = df.merge(
                inputdf_dict[k], on=[adm0c, adm1c, adm2c, "date"], how="left"
            )

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.date
    return df


def check_missingadmins(
    adm_path,
    pop_path,
    shp_adm1c,
    shp_adm2c,
    pop_adm1c,
    pop_adm2c,
    pop_col,
    pop_bound_adm2_mapping,
    pop_bound_adm1_mapping,
):
    """Determine if there is any admin regions that are not in the admin
    boundaries or population file.

    This to circumvent part of the population not being assigned to an
    admin. Args: adm_path: pop_path: path to csv with population counts
    per admin2 region shp_adm1c:  column name of the admin1 level name,
    in admin boundary data shp_adm2c:  column name of the admin2 level
    name, in admin boundary data pop_adm1c: column name of the admin1
    level name, in population data pop_adm2c: column name of the admin2
    level name, in population data pop_col: column name that contains
    the population count pop_bound_adm2_mapping: dict of admin2level
    names that don't correspond in FewsNet and population data. Keys are
    FewsNet names, values population pop_bound_adm1_mapping: dict of
    admin1level names that don't correspond in FewsNet and population
    data. Keys are FewsNet names, values population
    """
    df_adm2 = gpd.read_file(adm_path)
    df_pop = load_popdata(
        pop_path,
        pop_adm1c,
        pop_adm2c,
        pop_col,
        pop_bound_adm2_mapping=pop_bound_adm2_mapping,
        pop_bound_adm1_mapping=pop_bound_adm1_mapping,
    )

    missing_adm2_popbound = np.setdiff1d(
        list(df_pop[pop_adm2c].dropna()), list(df_adm2[shp_adm2c].dropna())
    )
    if missing_adm2_popbound.size > 0:
        logger.warning(
            "The following admin2 regions of the pop file are not found in"
            f" the boundaries shapefile: {missing_adm2_popbound}. You can"
            " adjust the pop_bound_adm2_mapping in the config file to include"
            " them"
        )

    missing_adm2_boundpop = np.setdiff1d(
        list(df_adm2[shp_adm2c].dropna()), list(df_pop[pop_adm2c].dropna())
    )
    if missing_adm2_boundpop.size > 0:
        logger.warning(
            "The following admin2 regions of the boundaries shapefile are not"
            f" found in the pop file {missing_adm2_boundpop}"
        )

    missing_adm1_popbound = np.setdiff1d(
        list(df_pop[pop_adm1c].dropna()), list(df_adm2[shp_adm1c].dropna())
    )
    if missing_adm1_popbound.size > 0:
        logger.warning(
            "The following admin1 regions of the pop file are not found in"
            f" the boundaries shapefile: {missing_adm1_popbound}. You can"
            " adjust the pop_bound_adm1_mapping in the config file to include"
            " them"
        )

    missing_adm1_boundpop = np.setdiff1d(
        list(df_adm2[shp_adm1c].dropna()), list(df_pop[pop_adm1c].dropna())
    )
    if missing_adm1_boundpop.size > 0:
        logger.warning(
            "The following admin1 regions of the boundaries shapefile are not"
            f" found in the pop file {missing_adm1_boundpop}"
        )


def load_popdata(
    pop_path,
    pop_adm1c,
    pop_adm2c,
    pop_col,
    pop_bound_adm2_mapping=None,
    pop_bound_adm1_mapping=None,
):
    """

    Args: pop_path: path to csv with population counts per admin2 region
        pop_adm1c: column name of the admin1 level name, in population
        data pop_adm2c: column name of the admin1 level name, in
        population data pop_col: column name that contains the
        population count pop_bound_adm2_mapping: dict of admin2level
        names that don't correspond in FewsNet and population data. Keys
        are FewsNet names, values population pop_bound_adm1_mapping:
        dict of admin1level names that don't correspond in FewsNet and
        population data. Keys are FewsNet names, values population

    Returns: df_pop: DataFrame with population per admin2/admin1
        combination that corresponds with FewsNet names
    """
    # import population data
    df_pop = pd.read_csv(pop_path)
    # remove whitespace at end of string
    df_pop[pop_adm2c] = df_pop[pop_adm2c].str.rstrip()
    if pop_bound_adm2_mapping:
        df_pop[pop_adm2c] = df_pop[pop_adm2c].apply(
            lambda x: get_new_name(x, pop_bound_adm2_mapping)
        )
    if pop_bound_adm1_mapping:
        df_pop[pop_adm1c] = df_pop[pop_adm1c].apply(
            lambda x: get_new_name(x, pop_bound_adm1_mapping)
        )
    no_popdata = df_pop.loc[
        df_pop[pop_col].isin([0, np.nan]), pop_adm2c
    ].values
    if len(no_popdata) > 0:
        logger.warning(f"No population data for {', '.join(no_popdata)}")

    df_pop[pop_col] = convert_to_numeric(df_pop[pop_col])
    # 0 is here treated as missing data, since it is not realistic that
    # a region has no population and will make calculations later on
    # easier
    df_pop[pop_col] = df_pop[pop_col].replace(0, np.nan)

    # in case there are duplicate adm1-adm2 combinations
    df_pop = df_pop.groupby([pop_adm1c, pop_adm2c], as_index=False).sum()
    df_pop.rename(columns={pop_col: "Total"}, inplace=True)
    return df_pop


def create_histpopdict(df_data, country, histpop_path):
    """
    Retrieve the historical national population for the years that are
    present in df_data Args: df_data: DataFrame of interest country:
    Country of interest histpop_path: path to csv with historical
    national population

    Returns: dict with national population for each year
    """
    df_histpop = pd.read_csv(histpop_path, header=2)
    df_histpop.set_index("Country Name", inplace=True)
    df_histpopc = df_histpop.loc[country]
    # only select rows that contain a year-value (some have e.g. unnamed
    # or some other info that we don't need)
    df_histpopc.index = pd.to_datetime(df_histpopc.index, errors="coerce")
    df_histpopc = df_histpopc[df_histpopc.index.notnull()]
    df_histpopc.index = df_histpopc.index.year.astype(str)

    # get years that are in df_data
    data_years = [
        str(i)
        for i in range(
            df_data["date"].min().year, df_data["date"].max().year + 1
        )
    ]

    # get years that are in df_data but not in df_histpopc
    y_nothist = np.setdiff1d(data_years, df_histpopc.index)

    # set values of years in df_data but not in in histpop as the value
    # of the last year in histpop Note: assuming here that only missing
    # values in histpop are after the last entry (e.g. only up to 2019
    # while in 2020 but assuming e.g. 2013 cannot be missing).
    for y in y_nothist:
        df_histpopc[y] = df_histpopc[df_histpopc.index.max()]

    # only return the years that are in data_years
    df_histpopc_data = df_histpopc[data_years]
    return df_histpopc_data.to_dict()


def get_adjusted(row, perc_dict):
    """Compute the subnational population, adjusted to the country's
    national population of that year."""
    year = str(row["date"].year)
    adjustment = perc_dict[year]
    if pd.isna(row["Total"]):
        return row["Total"]
    else:
        return int(row["Total"] * adjustment)


def merge_ipcpop(
    df_ipc,
    df_pop,
    country,
    pop_adm1c,
    pop_adm2c,
    shp_adm1c,
    shp_adm2c,
    histpop_path,
):
    """

    Args: df_ipc: DataFrame with IPC data df_pop: DataFrame with
        subnational population data country: Name of country of interest
        pop_adm1c: column name of the admin1 level name, in df_pop
        pop_adm2c: column name of the admin1 level name, in df_pop
        shp_adm1c:  column name of the admin1 level name, in df_ipc
        shp_adm2c:  column name of the admin2 level name, in df_ipc

    Returns: df_ipcp: DataFrame with IPC level and population per admin2
        region, where the population is adjusted to historical national
        averages
    """
    df_ipcp = df_ipc.merge(
        df_pop[[pop_adm1c, pop_adm2c, "Total"]],
        how="left",
        left_on=[shp_adm1c, shp_adm2c],
        right_on=[pop_adm1c, pop_adm2c],
    )

    # dict to indicate relative increase in population over the years
    pop_dict = create_histpopdict(
        df_ipcp, country=country, histpop_path=histpop_path
    )
    # estimate percentage of population at given year in relation to the
    # national population given by the subnational population file
    pop_tot_subn = df_ipcp[df_ipcp.date == df_ipcp.date.unique()[0]][
        "Total"
    ].sum()
    perc_dict = {k: v / pop_tot_subn for k, v in pop_dict.items()}

    df_ipcp["adjusted_population"] = df_ipcp.apply(
        lambda x: get_adjusted(x, perc_dict), axis=1
    )
    if (
        df_ipcp[df_ipcp.date == df_ipcp.date.max()].Total.sum()
        != df_pop.Total.sum()
    ):
        logger.warning(
            "Population data merged with IPC doesn't match the original"
            f" population numbers. Original:{df_pop.Total.sum()}, Merged:"
            f"{df_ipcp[df_ipcp.date == df_ipcp.date.max()].Total.sum()}"
        )

    # add columns with population in each IPC level for CS, ML1 and ML2
    for period in ["CS", "ML1", "ML2"]:
        for level in [1, 2, 3, 4, 5]:
            ipc_id = "{}_{}".format(period, level)
            df_ipcp[ipc_id] = np.where(
                df_ipcp[period] == level,
                df_ipcp["adjusted_population"],
                (np.where(np.isnan(df_ipcp[period]), np.nan, 0)),
            )
        df_ipcp[f"pop_{period}"] = df_ipcp[
            [f"{period}_{i}" for i in range(1, 6)]
        ].sum(axis=1, min_count=1)
        df_ipcp[f"pop_{period}"] = df_ipcp[f"pop_{period}"].replace(0, np.nan)

    # TODO: sort values and test
    # df_ipcp=df_ipcp.sort_values(by=["date",shp_adm1c,shp_adm2c])

    return df_ipcp


def aggr_admin1(df, adm1c):
    """
    Aggregate dataframe to admin1 level Args: df: DataFrame of interest
    adm1c: column name of the admin1 level name in df

    Returns: df_adm: dataframe with number of people in each IPC class
        per Admin1 region
    """
    cols_ipc = [
        f"{s}_{lev}" for s in ["CS", "ML1", "ML2"] for lev in range(1, 6)
    ]
    df_adm = (
        df[["date", "Total", "adjusted_population", adm1c] + cols_ipc]
        .groupby(["date", adm1c])
        .agg(lambda x: np.nan if x.isnull().all() else x.sum())
        .reset_index()
    )
    for period in ["CS", "ML1", "ML2"]:
        df_adm[f"pop_{period}"] = df_adm[
            [f"{period}_{i}" for i in range(1, 6)]
        ].sum(axis=1, min_count=1)

    return df_adm


def main(country, suffix, download, config=None):
    """This script takes the FEWSNET IPC shapefiles provided by on
    fews.net and overlays them with an admin2 shapefile, in order to
    provide an IPC value for each admin2 district. In the case where
    there are multiple values per district, the IPC value with the
    maximum area is selected.

    In FEWSNET IPC, there are 3 possible categories of maps - 'CS'
    (Current State), 'ML1' (3 months projection), 'ML2' (6 months
    projection). Any one of these is compatible with the script.

    Possible IPC values range from 1 (least severe) to 5 (most severe,
    famine).

    Set all variables, run the function for the different forecasts, and
    save as csv Args: country_iso3: string with iso3 code suffix: string
    to attach to the output files name config_file: path to config file
    """

    if config is None:
        config = Config()
    parameters = config.parameters(country)

    iso2_code = parameters["iso2_code"]
    region = parameters["foodinsecurity"]["region"]
    regioncode = parameters["foodinsecurity"]["regioncode"]
    shp_adm0c = parameters["shp_adm0c"]
    shp_adm1c = parameters["shp_adm1c"]
    shp_adm2c = parameters["shp_adm2c"]

    pop_filename = parameters["foodinsecurity"]["pop_filename"]
    pop_adm1c = parameters["foodinsecurity"]["adm1c_pop"]
    pop_adm2c = parameters["foodinsecurity"]["adm2c_pop"]
    pop_col = parameters["foodinsecurity"]["pop_col"]
    pop_bound_adm1_mapping = parameters["foodinsecurity"]["pop_adm1_mapping"]
    pop_bound_adm2_mapping = parameters["foodinsecurity"]["pop_adm2_mapping"]

    fewsnet_dates = config.FEWSNET_DATES
    if "fewsnet_dates_add" in parameters["foodinsecurity"].keys():
        fewsnet_dates = (
            fewsnet_dates + parameters["foodinsecurity"]["fewsnet_dates_add"]
        )
    if "fewsnet_dates_remove" in parameters["foodinsecurity"].keys():
        fewsnet_dates = list(
            set(fewsnet_dates)
            - set(parameters["foodinsecurity"]["fewsnet_dates_remove"])
        )

    country_data_raw_dir = os.path.join(
        config.DATA_PUBLIC_RAW_DIR, parameters["iso3_code"].lower()
    )
    country_data_processed_dir = os.path.join(
        config.DATA_PUBLIC_PROCESSED_DIR, parameters["iso3_code"].lower()
    )
    glb_data_raw_dir = os.path.join(config.DATA_PUBLIC_RAW_DIR, "glb")

    pop_path = os.path.join(
        country_data_raw_dir, config.POPSUBN_DIR, pop_filename
    )
    fewsnet_raw_dir = os.path.join(glb_data_raw_dir, config.FEWSNET_DIR)

    admin2bound_path = os.path.join(
        country_data_raw_dir,
        config.SHAPEFILE_DIR,
        parameters["path_admin2_shp"],
    )
    histpop_path = os.path.join(
        glb_data_raw_dir, "worldbank", config.WB_POP_FILENAME
    )
    output_dir = os.path.join(
        country_data_processed_dir, config.FEWSADMPOP_PROCESSED_DIR
    )
    # create output dir if it doesn't exist yet
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if download:
        for d in fewsnet_dates:
            download_fewsnet(d, iso2_code, region, regioncode, fewsnet_raw_dir)

    perioddf_dict = {}
    for period in config.IPC_PERIOD_NAMES:
        perioddf_dict[period] = gen_csml1m2(
            fewsnet_raw_dir,
            admin2bound_path,
            period,
            fewsnet_dates,
            shp_adm0c,
            shp_adm1c,
            shp_adm2c,
            region,
            regioncode,
            iso2_code,
        )

    df_allipc = merge_ipcperiod(perioddf_dict, shp_adm0c, shp_adm1c, shp_adm2c)
    # check whether names of adm regions in boundary and population
    # files don't correspond
    check_missingadmins(
        admin2bound_path,
        pop_path,
        shp_adm1c,
        shp_adm2c,
        pop_adm1c,
        pop_adm2c,
        pop_col,
        pop_bound_adm2_mapping,
        pop_bound_adm1_mapping,
    )
    df_pop = load_popdata(
        pop_path,
        pop_adm1c,
        pop_adm2c,
        pop_col,
        pop_bound_adm2_mapping=pop_bound_adm2_mapping,
        pop_bound_adm1_mapping=pop_bound_adm1_mapping,
    )

    df_ipcpop = merge_ipcpop(
        df_allipc,
        df_pop,
        country.capitalize(),
        pop_adm1c,
        pop_adm2c,
        shp_adm1c,
        shp_adm2c,
        histpop_path,
    )

    df_ipcpop.to_csv(
        os.path.join(output_dir, f"{country}_fewsnet_admin2{suffix}.csv")
    )

    df_adm1 = aggr_admin1(df_ipcpop, shp_adm1c)
    df_adm1.to_csv(
        os.path.join(output_dir, f"{country}_fewsnet_admin1{suffix}.csv")
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country.lower(), args.suffix, args.download_data)
