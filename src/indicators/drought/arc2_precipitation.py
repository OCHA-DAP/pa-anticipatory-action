import os
from pathlib import Path

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
PUBLIC_DATA_DIR = "public"
RAW_DATA_DIR = "raw"
PROCESSED_DATA_DIR = "processed"


def get_request_url(start_date, end_date, range_y, range_x):
    """
    Generate the url to get the ARC2 data from IRI.
    """
    # "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/.Africa/.DAILY/
    # .ARC2/.daily/.est_prcp/T/%281%20Jan%202000%29%2830%20Mar%202021%29RANGEEDGES/X
    # /%2832E%29%2836E%29RANGEEDGES/Y/%2820S%29%285S%29RANGEEDGES/data.nc"

    return


def get_raw_filepath(iso3: str):
    # return public / iso3 / raw / arc2
    return


def get_processed_filepath(iso3: str):
    # return public / iso3 / processed / arc2
    return


def download_data(url: str, raw_filepath: str):
    """
    Download data from IRI servers and save raw .nc file on
    gdrive at the location returned from get_raw_filepath
    File name should follow something like arc2_daily_precip_iso3_start_end.nc
    """
    # Download the data from IRI's site
    # URL generated as in the get_request_url

    # Data should be saved to raw_filepath

    # FROM TINKA'S SCRIPT
    # strange things happen when just overwriting the file,
    # so delete it first if it already exists
    # if os.path.exists(arc2_filepath):
    #     os.remove(arc2_filepath)

    # #have to authenticate by using a cookie
    # cookies = {
    #     '__dlauth_id': os.getenv("IRI_AUTH"),
    # }

    # # logger.info("Downloading arc2 NetCDF file. This might take some time")
    # response = requests.get(arc2_mwi_url, cookies=cookies, verify=False)

    # with open(arc2_filepath, "wb") as fd:
    #     for chunk in response.iter_content(chunk_size=128):
    #         fd.write(chunk)
    return


def process_data(processed_filepath, raw_filepath, crs, clip_bounds):
    """
    Clip the data, set the CRS, compute 14-day rolling sum
    """
