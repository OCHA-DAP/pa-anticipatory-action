"""Code to download and process WRSI data

The WRSI data is made publicly available through
the USGS website. The data is dekadal and each
dekadal dataset is shared through zip files.
https://edcftp.cr.usgs.gov/project/fews/dekadal/

These include two separate zones for West Africa:

- WA: the largest region, covering from the coast at
    5 degrees to around 15 degrees latitude, ranging
    from the western coast of Senegal to the western
    border of Sudan. This is specifically for croplands:
    https://earlywarning.usgs.gov/fews/product/56
- W1: smaller, ranges from around 11 degrees latitude
    to around 17, and from the eastern border of Chad
    to Senegal. This is specifically for rangelands:
    https://earlywarning.usgs.gov/fews/product/57

Both are published using EPSG:7764. There is overlap between the
two datasets where they don't agree, even though both are
measuring the WRSI with respect to millet, due to their
calculations for rangeland and cropland. Since most
countries in the Sahel overlap with both datasets, both
should be considered and we should get clarification from
experts on the differences between the two, how they relate,
and how best to use them.
"""

import logging
import os
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Literal, Union, get_args
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

logger = logging.getLogger(__name__)

_base_url = "https://edcftp.cr.usgs.gov/project/fews/dekadal/africa_west/"

_url_zip_filename = "w{year:02}{dekad}{region}.zip"

_base_file_name = "w{year:02}{dekad}{type}.tif"


def _get_url_path(year, dekad, region, historical=False):
    url_zip_filename = _url_zip_filename.format(
        year=year % 100, dekad=dekad, region=region
    )
    historical = "historical/" if historical else ""
    return _base_url + historical + url_zip_filename


def _download_wrsi(year: int, dekad: int, region: str, file_dir: Path):
    try:
        url = _get_url_path(year=year, dekad=dekad, region=region)
        resp = urlopen(url)
    except HTTPError:
        try:
            url = _get_url_path(
                year=year, dekad=dekad, region=region, historical=True
            )
            resp = urlopen(url)
        except HTTPError:
            logger.error(
                f"No data available for region {region} for "
                f"dekad {dekad} of {year}, skipping."
            )
            return

    zf = ZipFile(BytesIO(resp.read()))
    for type in ["do", "eo"]:
        file_name = _base_file_name.format(
            year=year % 100, dekad=dekad, type=type
        )
        save_path = os.path.join(file_dir, file_name)
        if os.path.exists(save_path):
            os.remove(save_path)

        zf.extract(file_name, file_dir)


_raw_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "raw", "general", "wrsi", "west_africa"
)


def _get_raw_dir(region):
    return Path(os.path.join(_raw_path, region))


def _get_year_dekad(date_obj: Union[date, str]):
    if isinstance(date_obj, str):
        date_obj = date.fromisoformat(date_obj)
    year = date_obj.year
    dekad = (date_obj.timetuple().tm_yday // 10) + 1
    return year, dekad


RegionArgument = Literal["cropland", "rangeland"]
_valid_region = get_args(RegionArgument)


def download_wrsi(
    region: RegionArgument,
    start_date: Union[date, str, None] = None,
    end_date: Union[date, str, None] = None,
):
    if region not in _valid_region:
        raise ValueError("`region` must be one of 'cropland' or 'rangeland'.")
    else:
        url_region = "wa" if region == "cropland" else "w1"

    if start_date is None:
        start_year, start_dekad = 2001, 13
        if region == "rangeland":
            start_year += 1
    else:
        start_year, start_dekad = _get_year_dekad(start_date)

    end_date = date.today() if end_date is None else end_date
    end_year, end_dekad = _get_year_dekad(end_date)
    end_dekad -= 1

    i_year = start_year
    i_dekad = min(start_dekad, 33)

    file_dir = _get_raw_dir(region)
    file_dir.mkdir(parents=True, exist_ok=True)

    while not (
        i_year > end_year or (i_year == end_year and i_dekad > i_dekad)
    ):
        _download_wrsi(
            year=i_year, dekad=i_dekad, region=url_region, file_dir=file_dir
        )
        i_dekad += 1
        if i_dekad > 33:
            i_year += 1
            i_dekad = 13


_processed_filename = "biomasse_{admin}_dekad_{start_dekad}.csv"

_processed_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "processed", "general", "biomasse"
)


def _get_processed_path(admin, start_dekad):
    return os.path.join(
        _processed_path,
        _processed_filename.format(admin=admin, start_dekad=start_dekad),
    )
