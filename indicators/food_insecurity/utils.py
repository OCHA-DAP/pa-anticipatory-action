import zipfile
import logging
import os
import argparse
import requests
import yaml
import coloredlogs

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
        "-d", "--download-fewsnet", action="store_true", help="Download the raw FewsNet data"
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
    logger.info(f'Downloaded "{url}" to "{save_path}"')

def unzip(zip_file_path, save_path):
    logger.info(f"Unzipping {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

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
    try:
        download_url(url_country, zip_filename_country)
    except Exception:
        logger.warning(f"Cannot download FewsNet data for {iso2_code}, {date}")
    try:
        unzip(zip_filename_country, os.path.join(output_dir, f"{iso2_code}{date}"))
        os.remove(zip_filename_country)
    except Exception:
        logger.warning(
            f"File {zip_filename_country} is not a zip file, probably indicates FewsNet data for {iso2_code}, {date} doesn't exist. Removing the file.")
        os.remove(zip_filename_country)


    url_region = f"{FEWSNET_BASE_URL_REGION}{regioncode}/{region}{date}.zip"
    zip_filename_region = os.path.join(
        output_dir, f"{region}{date}.zip"
    )
    try:
        download_url(url_region, zip_filename_region)
    except Exception:
        logger.warning(f"Cannot download FewsNet data for {region}, {date}")
    try:
        unzip(zip_filename_region, os.path.join(output_dir, f"{region}{date}"))
        os.remove(zip_filename_region)
    except Exception:
        logger.warning(
            f"File {zip_filename_region} is not a zip file, probably indicates FewsNet data for {region}, {date} doesn't exist. Removing the file.")
        os.remove(zip_filename_region)
