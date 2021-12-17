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

Both are published using ESRI:102022. There is overlap between the
two datasets where they don't agree, even though both are
measuring the WRSI with respect to millet, due to their
calculations for rangeland and cropland. Since many
countries in the Sahel overlap with both datasets, both
should be considered and we should be careful in learning
to apply and use both products where applications.

I have also just found the CHIRPS WRSI data on the USGS website.
These are available at:

https://edcftp.cr.usgs.gov/project/fews/africa/west/dekadal/wrsi-chirps-etos
"""

# TODO: investigate having multiple dimensions for year/dekad vs. time
import datetime
import logging
import os
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Union, get_args
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://edcftp.cr.usgs.gov/project/fews/"
    "africa/west/dekadal/wrsi-chirps-etos/{region}/"
)

_URL_ZIP_FILENAME = "w{year:04}{dekad}{region}.zip"

_BASE_FILENAME = "w{year:04}{dekad}{type}.tif"

_RAW_DIR = Path(
    os.getenv("AA_DATA_DIR"), "public", "raw", "glb", "wrsi", "west_africa"
)

_PROCESSED_FILENAME = "wrsi_{region}_{type}.nc"

_PROCESSED_DIR = Path(
    os.getenv("AA_DATA_DIR"), "public", "processed", "glb", "wrsi"
)

# defined here: https://gis.stackexchange.com/questions/
# 177447/defining-the-correct-aea-proj4string-for-
# fewsnet-rainfall-data-southern-africa
_WRSI_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=-19.0 +lat_2=21.0 +lat_0=1.0 "
    "+lon_0=20 +x_0=0 +y_0=0 +ellps=clrk66 +units=m +no_defs"
)

RegionArgument = Literal["cropland", "rangeland"]
_VALID_REGION = get_args(RegionArgument)


def download_wrsi(
    region: RegionArgument,
    start_date: Union[date, str, List[int], None] = None,
    end_date: Union[date, str, List[int], None] = None,
    clobber: bool = False,
):
    """Download USGS WRSI data

    Downloads USGS WRSI data from their website.
    Currently only downloads data for west African
    croplands and rangelands, where data is available
    from 2001, for all dekads between 13 and 33 of
    each year.
    Parameters
    ----------
    region: RegionArgument
        One of 'cropland' or 'rangeland'.
    start_date: Union[date, str, List[int], None]
        Starting date for data downloading passed in as
        a ``datetime.date``, ISO8601 date string, or
        ``list`` of ``int`` for year and dekad. If ``None``,
        defaults to ``[2001, 13]``.
    end_date: Union[date, str, List[int], None]
        Ending date for data downloading passed in as
        a ``datetime.date``, ISO8601 date string, or
        ``list`` of ``int`` for year and dekad. If ``None``,
        defaults to ``datetime.date.today()``.
    clobber: bool (optional)
        If ``True``, rewrite existing files. Otherwise,
        only download files not alreayd available.

    Returns
    -------
        None
    """
    if region not in _VALID_REGION:
        raise ValueError("`region` must be one of 'cropland' or 'rangeland'.")

    if start_date is None:
        start_year, start_dekad = 2001, 13
    elif isinstance(start_date, list):
        start_year, start_dekad = start_date[0], start_date[1]
    else:
        start_year, start_dekad = _date_to_dekad(start_date)

    if isinstance(end_date, list):
        end_year, end_dekad = end_date[0], end_date[1]
    else:
        if end_date is None:
            end_date = date.today()
        end_year, end_dekad = _date_to_dekad(end_date)

    i_year = max(start_year, 2001)
    i_dekad = max(min(start_dekad, 33), 13)
    raw_dir = _get_raw_dir(region)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # existing file dates
    dts = [
        _date_to_dekad(_fp_date(filename))
        for filename in raw_dir.glob("*.tif")
    ]
    while not (
        i_year > end_year or (i_year == end_year and i_dekad > i_dekad)
    ):
        if clobber or [i_year, i_dekad] not in dts:
            _download_wrsi_dekad(
                year=i_year, dekad=i_dekad, region=region, raw_dir=raw_dir
            )
        i_dekad += 1
        if i_dekad > 33:
            i_year += 1
            i_dekad = 13


def process_wrsi(region: RegionArgument, type: str):
    """Process WRSI data

    Processes WRSI data for specific
    region and type. Data is currently
    only available for `cropland` and
    `rangeland` areas in West Africa.
    The available products are the
    current, extended, and anomaly
    WRSI.

    Parameters
    ----------
    region: RegionArgument
        One of 'cropland' or 'rangeland'.
    type: str
        One of 'current', 'extended', or
        'anomaly'.

    Returns
    -------
    xr.DataArray
    """
    if region not in _VALID_REGION:
        raise ValueError("`region` must be one of 'cropland' or 'rangeland'.")

    raw_dir = _get_raw_dir(region)

    types = {"current": "do", "extended": "eo", "anomaly": "er"}

    if type in types.keys():
        fp_type = types[type]
    else:
        raise ValueError(
            "`type` must be one of 'current', 'extended', or 'anomaly'."
        )

    processed_path = _get_processed_path(region, type)

    arrays = [
        _load_raw(raw_dir, filename)
        for filename in raw_dir.glob(f"*{fp_type}.tif")
    ]

    arrays_merged = xr.concat(arrays, "time").sortby("time")

    # convert to dataset
    da = arrays_merged.squeeze().to_dataset(name="wrsi")

    # saving file
    processed_path.unlink(missing_ok=True)

    da.to_netcdf(processed_path)

    return da


def filter_wrsi(da: xr.DataArray, dekad: int):
    """Filter WRSI array to dekad

    Filters WRSI data array to
    specific dekad.

    Parameters
    ----------
    da: xr.DataArray
        WRSI xarray DataArray
    dekad: int
        Dekad.

    Returns
    -------
    xr.DataArray
    """
    dt = _dekad_to_date(2001, dekad)
    filter_month = dt.month
    filter_day = dt.day
    return da.where(
        (da.time.dt.day == filter_day) & (da.time.dt.month == filter_month),
        drop=True,
    )


def wrsi_percent_below(da: xr.DataArray, thresholds: List[int]):
    """Calculate percent of area below threshold

    Calculates WRSI % below or equal to threshold.

    Parameters
    ----------
    da: xr.DataArray
        WRSI xarray DataArray
    thresholds:
        List of integer thresholds.


    Returns
    -------

    """
    dfs = [
        _wrsi_percent_below_thresh(da, threshold) for threshold in thresholds
    ]
    return pd.concat(dfs)


def load_wrsi(region: RegionArgument, type: str):
    """Load WRSI data

    Loads WRSI raster data for
    specified region and type.

    Parameters
    ----------
    region: RegionArgument
        One of 'cropland' or 'rangeland'.
    type: str
        One of 'current', 'extended', or
        'anomaly'.

    Returns
    -------

    """
    processed_path = _get_processed_path(region, type)
    a = rioxarray.open_rasterio(processed_path)
    a.rio.write_crs(_WRSI_CRS, inplace=True)
    a = a.rio.reproject("EPSG:4326")
    return a


def _get_url_path(year, dekad, region):
    if region == "cropland":
        url_region, file_region = "west", "wa"
    else:
        url_region, file_region = "west1", "w1"

    url_zip_filename = _URL_ZIP_FILENAME.format(
        year=year, dekad=dekad, region=file_region
    )

    base_url = _BASE_URL.format(region=url_region)
    return base_url + url_zip_filename


def _download_wrsi_dekad(year: int, dekad: int, region: str, raw_dir: Path):
    try:
        url = _get_url_path(year=year, dekad=dekad, region=region)
        resp = urlopen(url)
    except HTTPError:
        logger.message(
            f"No data available for region {region} for "
            f"dekad {dekad} of {year}, skipping."
        )
        return

    with ZipFile(BytesIO(resp.content)) as zf:
        # do is current, eo is extended, er is anomaly.
        for type in ["do", "eo", "er"]:
            file_name = _BASE_FILENAME.format(
                year=year, dekad=dekad, type=type
            )
            zf.extract(file_name, raw_dir)


def _get_raw_dir(region):
    return _RAW_DIR / region


def _date_to_dekad(date_obj: Union[date, str]):
    if isinstance(date_obj, str):
        date_obj = date.fromisoformat(date_obj)
    year = date_obj.year
    dekad = (date_obj.day // 10) + ((date_obj.month - 1) * 3) + 1
    return year, dekad


def _dekad_to_date(year: int, dekad: int):
    """Compute date from year and dekad

    From a year and dekad, computes the
    start date for that time period. This
    is based on the USGS (and relatively
    common) dekadal definition of the
    1st and 2nd dekad of a month being
    the first 10 day periods, and the 3rd
    dekad being the remaining days within
    that month.
    """
    month = ((dekad - 1) // 3) + 1
    day = 10 * ((dekad - 1) % 3) + 1
    return datetime.datetime(year, month, day)


def _fp_date(fp):
    """Get date from filepath"""
    return _dekad_to_date(int(fp.stem[1:5]), int(fp.stem[5:7]))


def _get_processed_path(region, type):
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return _PROCESSED_DIR / _PROCESSED_FILENAME.format(
        region=region, type=type
    )


def _load_raw(raw_dir, fp):
    arr = rioxarray.open_rasterio(raw_dir / fp, masked=True)
    dt = [_fp_date(fp)]

    arr = arr.expand_dims(time=dt)
    # round x dimension due to issues with
    # concat and tiny numerical differences
    arr = arr.assign_coords({"x": arr.x.astype("float32")})
    return arr


def _wrsi_percent_below_thresh(da, threshold):
    da = xr.where(da <= threshold, 1, 0).mean(dim=["x", "y"])
    time = da.time
    df = da.to_dataframe(name="wrsi_percent_area").reset_index(drop=True)
    df["time"] = time
    df["threshold"] = threshold
    return df
