import argparse
import logging
import os
import sys
import urllib.error
from pathlib import Path

import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.utils_general.utils import download_ftp, download_url, unzip

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("iso3", help="iso3 code of country of interest")
    parser.add_argument("-a", "--admin_level", default=1)
    # Prefix for filenames
    parser.add_argument(
        "-s",
        "--suffix",
        default="",
        type=str,
        help="Suffix for output files, and if applicable input files",
    )
    parser.add_argument(
        "-d",
        "--download-data",
        action="store_true",
        help=(
            "Download the raw data. FewsNet and WorldPop are currently"
            " implemented"
        ),
    )
    parser.add_argument(
        "-da",
        "--dates",
        default=None,
        nargs="+",
        type=str,
        help="List of strings of dates to be included in YM format",
    )
    return parser.parse_args()


def download_fewsnet(date, iso2_code, region, regioncode, output_dir):
    """Retrieve the raw fewsnet data.

    Depending on the region, this date is published per region or per
    country. This function tries to retrieve both. The download_url
    always downloads the given url, but sometimes this doesn't return a
    valid zip file. This means that data doesn't exist. This happens
    often, since for most countries the classifications are on earlier
    dates only published per region and later on per country. This is
    not bad, and the function will remove the invalid zip files Args:
    date: str in yyyymm format. Date for which to retrieve the fewsnet
    data. iso2_code: iso2 code of the country of interest region: region
    that the fewsnet data covers, e.g. "east-africa" regioncode:
    abbreviation of the region that the fewsnet data covers, e.g. "EA"
    output_dir: directory to save the files to
    """
    FEWSNET_BASE_URL_REGION = "https://fews.net/data_portal_download/download?data_file_path=http%3A//shapefiles.fews.net.s3.amazonaws.com/HFIC/"  # noqa: E501
    FEWSNET_BASE_URL_COUNTRY = "https://fdw.fews.net/api/ipcpackage/"
    url_country = f"{FEWSNET_BASE_URL_COUNTRY}?country_code={iso2_code.upper()}&collection_date={date[:4]}-{date[-2:]}-01"  # noqa: E501
    zip_filename_country = os.path.join(output_dir, f"{iso2_code}{date}.zip")
    output_dir_country = os.path.join(output_dir, f"{iso2_code}{date}")
    if not os.path.exists(output_dir_country):
        # var to check if country data exists
        country_data = False
        try:
            download_url(url_country, zip_filename_country)
        except Exception:
            logger.warning(
                f"Url of country level FewsNet data for {iso2_code}, {date} is"
                " not valid"
            )
        try:
            unzip(zip_filename_country, output_dir_country)
            logger.info(
                f'Downloaded "{url_country}" to "{zip_filename_country}'
            )
            logger.info(f"Unzipped {zip_filename_country}")
            country_data = True
        except Exception:
            # indicates that the url returned something that wasn't a
            # zip, happens often and indicates data for the given
            # country - date is not available
            logger.info(
                f"No country level FewsNet data for {iso2_code}, {date}, using"
                " regional data if available"
            )
        os.remove(zip_filename_country)

        url_region = (
            f"{FEWSNET_BASE_URL_REGION}{regioncode}/{region}{date}.zip"
        )
        zip_filename_region = os.path.join(output_dir, f"{region}{date}.zip")
        output_dir_region = os.path.join(output_dir, f"{region}{date}")
        if not country_data and not os.path.exists(output_dir_region):
            try:
                if not os.path.exists(zip_filename_region):
                    download_url(url_region, zip_filename_region)
            except Exception:
                logger.warning(
                    f"Url of regional level FewsNet data for {region}, {date}"
                    " is not valid"
                )
            try:
                unzip(zip_filename_region, output_dir_region)
                logger.info(
                    f'Downloaded "{url_region}" to "{zip_filename_region}'
                )
                logger.info(f"Unzipped {zip_filename_region}")
            except Exception:
                # indicates that the url returned something that wasn't
                # a zip, happens often and indicates data for the given
                # country - date is not available
                logger.warning(
                    f"No FewsNet data for date {date} found that covers"
                    f" {iso2_code}"
                )
            os.remove(zip_filename_region)


def download_worldpop(country_iso3, year, output_dir, config):
    # create directory if doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url = config.WORLDPOP_URL.format(
        country_iso3_upper=country_iso3.upper(),
        country_iso3_lower=country_iso3.lower(),
        year=year,
    )
    output_file = os.path.join(output_dir, url.split("/")[-1])
    if not os.path.exists(output_file):
        try:
            download_ftp(url, output_file)
        except urllib.error.URLError as e:
            logger.warning(
                f"{e}. Data of the year of interest might not exist on the"
                " WorldPop FTP."
            )


def fewsnet_validperiod(row):
    """
    Add the period for which FewsNet's projections are valid. Till 2016
    FN published a report 4 times a year, where each projection period
    had a validity of 3 months From 2017 this has changed to thrice a
    year, where each projection period has a validity of 4 months Args:
    row: row of dataframe containing the date the FN data was published
    (i.e. CS period) as timestamp

    Returns:

    """
    # make own mapping, to be able to use mod 12 to calculate months of
    # projections
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
        period_ML2 = (
            f"{month_abbr[(month + 3) % 12]} - {month_abbr[(month + 5) % 12]}"
            f" {year_ml2}"
        )
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
                "Period of ML1 and ML2 cannot be added for non-regular"
                f" publishing date {year}-{month}. Add manually."
            )
            row["period_ML1"] = np.nan
            row["period_ML2"] = np.nan
            return row
        period_ML1 = (
            f"{month_abbr[month]} - {month_abbr[(month + 3) % 12]} {year_ml1}"
        )
        period_ML2 = (
            f"{month_abbr[(month + 4) % 12]} - {month_abbr[(month + 7) % 12]}"
            f" {year_ml2}"
        )
    row["period_ML1"] = period_ML1
    row["period_ML2"] = period_ML2
    return row


def compute_percentage_columns(df, config):
    """
    calculate percentage of population per analysis period and level
    Args: df (pd.DataFrame): input df, should include columns of the
    IPC_PERIOD_NAMES for eah period in range(1,6) config (Config):
    food-insecurity config class Returns: df(pd.DataFrame): input df
    with added percentage columns
    """
    for period in config.IPC_PERIOD_NAMES:
        # IPC level goes up to 5, so define range up to 6
        for i in range(1, 6):
            c = f"{period}_{i}"
            df[f"perc_{c}"] = df[c] / df[f"pop_{period}"] * 100
        # get pop and perc in IPC3+ and IPC2-
        # 3p = IPC level 3 or higher, 2m = IPC level 2 or lower
        df[f"{period}_3p"] = df[[f"{period}_{i}" for i in range(3, 6)]].sum(
            axis=1
        )
        df[f"perc_{period}_3p"] = (
            df[f"{period}_3p"] / df[f"pop_{period}"] * 100
        )
        df[f"{period}_4p"] = df[[f"{period}_{i}" for i in range(4, 6)]].sum(
            axis=1
        )
        df[f"perc_{period}_4p"] = (
            df[f"{period}_4p"] / df[f"pop_{period}"] * 100
        )
        df[f"{period}_2m"] = df[[f"{period}_{i}" for i in range(1, 3)]].sum(
            axis=1
        )
        df[f"perc_{period}_2m"] = (
            df[f"{period}_2m"] / df[f"pop_{period}"] * 100
        )
    df["perc_inc_ML2_3p"] = df["perc_ML2_3p"] - df["perc_CS_3p"]
    df["perc_inc_ML1_3p"] = df["perc_ML1_3p"] - df["perc_CS_3p"]
    # round the percentage columns to 3 decimals
    perc_cols = [col for col in df.columns if "perc" in col]
    df[perc_cols] = df[perc_cols].round(3)
    return df
