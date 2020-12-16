import zipfile
import logging
import os
import argparse
import requests
import yaml
import coloredlogs
import locale
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
import datetime

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("country_iso3", help="Country ISO3")
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
        "-d", "--download-data", action="store_true", help="Download the raw data. FewsNet and WorldPop are currently implemented"
    )
    return parser.parse_args()


def parse_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def config_logger(level="INFO"):
    # Colours selected from here:
    # http://humanfriendly.readthedocs.io/en/latest/_images/ansi-demo.png
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        field_styles={
            "name": {"color": 8},
            "asctime": {"color": 248},
            "levelname": {"color": 8, "bold": True},
        },
    )

def download_url(url, save_path, chunk_size=128):
    # Remove file if already exists
    try:
        os.remove(save_path)
    except OSError:
        pass
    # Download
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_ftp(url, save_path):
    logger.info(f'Downloading "{url}" to "{save_path}"')
    urlretrieve(url, filename=save_path)

def unzip(zip_file_path, save_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

def convert_to_numeric(df_col,zone="en_US"):
    if df_col.dtype == "object":
        locale.setlocale(locale.LC_NUMERIC, zone)
        df_col = df_col.apply(lambda x: locale.atof(x))
        df_col = pd.to_numeric(df_col, errors="coerce")
    return df_col

def get_fewsnet_data(date, iso2_code, region, regioncode,output_dir):
    """
    Retrieve the raw fewsnet data. Depending on the region, this date is published per region or per country. This function tries to retrieve both.
    The download_url always downloads the given url, but sometimes this doesn't return a valid zip file. This means that data doesn't exist. This happens often, since for most countries the classifications are on earlier dates only published per region and later on per country. This is not bad, and the function will remove the invalid zip files
    Args:
        date: str in yyyymm format. Date for which to retrieve the fewsnet data.
        iso2_code: iso2 code of the country of interest
        region: region that the fewsnet data covers, e.g. "east-africa"
        regioncode: abbreviation of the region that the fewsnet data covers, e.g. "EA"
        output_dir: directory to save the files to
    """
    FEWSNET_BASE_URL_REGION = "https://fews.net/data_portal_download/download?data_file_path=http%3A//shapefiles.fews.net.s3.amazonaws.com/HFIC/"
    FEWSNET_BASE_URL_COUNTRY = "https://fdw.fews.net/api/ipcpackage/"
    url_country = f"{FEWSNET_BASE_URL_COUNTRY}?country_code={iso2_code}&collection_date={date[:4]}-{date[-2:]}-01"
    zip_filename_country = os.path.join(
        output_dir, f"{iso2_code}{date}.zip"
    )
    output_dir_country=os.path.join(output_dir, f"{iso2_code}{date}")
    if not os.path.exists(output_dir_country):
        #var to check if country data exists
        country_data = False
        try:
            download_url(url_country, zip_filename_country)
        except Exception:
            logger.warning(f"Url of country level FewsNet data for {iso2_code}, {date} is not valid")
        try:
            unzip(zip_filename_country, output_dir_country)
            logger.info(f'Downloaded "{url_country}" to "{zip_filename_country}')
            logger.info(f"Unzipped {zip_filename_country}")
            country_data=True
        except Exception:
            #indicates that the url returned something that wasn't a zip, happens often and indicates data for the given country - date is not available
            logger.info(f"No country level FewsNet data for {iso2_code}, {date}, using regional data if available")
        os.remove(zip_filename_country)

        url_region = f"{FEWSNET_BASE_URL_REGION}{regioncode}/{region}{date}.zip"
        zip_filename_region = os.path.join(
            output_dir, f"{region}{date}.zip"
        )
        output_dir_region = os.path.join(output_dir, f"{region}{date}")
        if not country_data and not os.path.exists(output_dir_region):
            try:
                if not os.path.exists(zip_filename_region):
                    download_url(url_region, zip_filename_region)
            except Exception:
                logger.warning(f"Url of regional level FewsNet data for {region}, {date} is not valid")
            try:
                unzip(zip_filename_region, output_dir_region)
                logger.info(f'Downloaded "{url_region}" to "{zip_filename_region}')
                logger.info(f"Unzipped {zip_filename_region}")
            except Exception:
                # indicates that the url returned something that wasn't a zip, happens often and indicates data for the given country - date is not available
                logger.warning(f"No FewsNet data for date {date} found that covers {iso2_code}")
            os.remove(zip_filename_region)

def get_worldpop_data(country_iso3, year, output_dir, config):
    #create directory if doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    url = config.WORLDPOP_URL.format(country_iso3_upper=country_iso3.upper(),country_iso3_lower=country_iso3.lower(),year=year)
    output_file=os.path.join(output_dir, url.split("/")[-1])
    if not os.path.exists(output_file):
        download_ftp(url, output_file)

def get_globalipc_data(country_iso3, country_iso2, output_dir, config):
    #create directory if doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    min_year=2010 #first year to retrieve data for. Doesn't matter if global ipc only started including data for later years
    max_year=datetime.datetime.now().year #last date to retrieve data for. Doesn't matter if this is in the future
    url = config.GLOBALIPC_URL.format(min_year=min_year,max_year=max_year,country_iso2=country_iso2)
    output_file=os.path.join(output_dir, config.GLOBALIPC_FILENAME.format(country_iso3=country_iso3))
    #have one file with all data, so also download if file already exists to make sure it contains the newest data (contrary to fewsnet)
    try:
        download_url(url, output_file)
    except Exception:
        logger.warning(f"Cannot download GlobalIPC data for {country_iso3}")