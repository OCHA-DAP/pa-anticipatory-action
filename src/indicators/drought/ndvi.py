"""Code to download and process USGS NDVI data

USGS processed and smoothed NDVI data is downloadable
and available through the USGS webiste
with methods described for west-africa at
https://earlywarning.usgs.gov/fews/product/451 and for east-africa
at https://earlywarning.usgs.gov/fews/product/448
An example of the actual data from the USGS file explorer for west-africa:
https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/africa/west/dekadal/emodis/ndvi_c6/temporallysmoothedndvi/downloads/monthly/

The products include temporally smoothed NDVI, median anomaly,
difference from the previous year, and median anomaly
presented as a percentile. Currently the code only has the implementation
of the percent of median because it was developed for Chad,
where it's similar to the % anomaly products we were comparing to
for Biomasse and WRSI. However, this code can be extended easily in
the future to download and process other NDVI data.

Data by USGS is published quickly after the dekad.
After about 1 month this data is updated with temporal smoothing
 and error correction for cloud cover.
"""

# TODO: add progress bar
import logging
import os
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import List, Union
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import geopandas.geoseries
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/"
    "web/africa/{region}/dekadal/emodis/ndvi_c6/percentofmedian/"
    "downloads/dekadal/{base_file_name}.zip"
)

_BASE_FILENAME = "{region_code}{year:02}{dekad:02}pct"

_RAW_DIR = Path(os.getenv("AA_DATA_DIR"), "public", "raw", "glb", "ndvi")

_REGION_ABBR_MAPPING = {"west": "wa", "east": "ea"}


def download_ndvi(
    # TODO: in toolbox have region as mandatory argument or included in config
    region: str = "west",
    start_date: Union[date, str, List[int], None] = None,
    end_date: Union[date, str, List[int], None] = None,
    clobber: bool = False,
) -> None:
    """Download NDVI data

    Downlaods NDVI data from the start to the end date for the given
    region in Africa. Region "west" and "east" have been tested


    Parameters
    ----------
    region: str
        The region in Africa to download the data for.
        Currently tested with "west" and "east"
    start_date: Union[date, str, List[int], None]
        Start date for download.
    end_date: Union[date, str, List[int], None]
        End date for download.
    clobber: bool
        If ``True`` redownload old data.

    Returns
    -------
    None
    """
    if start_date is None:
        start_year, start_dekad = 2002, 19
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

    dts = [
        _fp_date(filename.stem)
        for filename in _RAW_DIR.glob(f"{_REGION_ABBR_MAPPING[region]}*.tif")
    ]
    for year in range(start_year, end_year):
        for dekad in range(1, 37):
            if year == start_year and dekad < start_dekad:
                continue
            if year == end_year and dekad > end_dekad:
                continue
            if clobber or [year, dekad] not in dts:
                print(year)
                _download_ndvi_dekad(region=region, year=year, dekad=dekad)


def load_raw_dekad_ndvi(region: str, year: int, dekad: int):
    _, file_name = _get_paths(region=region, year=year, dekad=dekad)
    file_path = Path(_RAW_DIR, file_name)
    ds = xr.load_dataset(file_path)
    return ds.squeeze().drop("band").band_data


def process_ndvi(
    iso3: str,
    geometries: geopandas.geoseries.GeoSeries,
    thresholds: List[int],
    region: str = "west",
) -> pd.DataFrame:
    """Process NDVI data for specific area

    NDVI data is clipped to the provided
    ``geometries``, usually a geopandas
    dataframes ``geometry`` feature, and
    percent of area below each ``threshold``
    is calculated.

    Parameters
    ----------
    iso3: str
        ISO3 code for the country file.
    geometries: geopandas.geoseries.GeoSeries
        Geometry to clip raster file to.
    thresholds: List[int]
        List of thresholds to calculate percent
        of area below.
    region: str
        The region in Africa to download the data for.
        Currently tested with "west" and "east"

    Returns
    -------
    pd.DataFrame
    """
    iso3 = iso3.lower()
    processed_path = _get_processed_path(iso3)

    data = []
    for filename in _RAW_DIR.glob(f"{_REGION_ABBR_MAPPING[region]}*.tif"):
        da = xr.open_rasterio(_RAW_DIR / filename)
        da = da.rio.clip(geometries, drop=True, from_disk=True)
        da_date = _fp_date(filename.stem)
        for threshold in thresholds:
            area_pct = (
                xr.where(da <= threshold, 1, 0).mean(dim=["x", "y"]).values[0]
                * 100
            )
            data.append([da_date, iso3, threshold, area_pct])

    df = pd.DataFrame(
        data, columns=["date", "iso3", "anomaly_thresholds", "percent_area"]
    )
    df.sort_values(by="date", inplace=True)

    # saving file
    df.to_csv(processed_path)

    return df


def load_processed_ndvi(iso3: str) -> pd.DataFrame:
    """Load processed NDVI data

    Loads processed NDVI data for
    the specific ISO3 string
    passed in.

    Parameters
    ----------
    iso3: str
        ISO3 code for the country file.

    Returns
    -------
    pd.DataFrame
    """
    processed_path = _get_processed_path(iso3.lower())
    return pd.read_csv(processed_path)


def _get_paths(region: str, year: int, dekad: int):
    base_file_name = _BASE_FILENAME.format(
        region_code=_REGION_ABBR_MAPPING[region], year=year % 100, dekad=dekad
    )
    url_path = _BASE_URL.format(region=region, base_file_name=base_file_name)
    return url_path, f"{base_file_name}.tif"


def _download_ndvi_dekad(region: str, year: int, dekad: int):
    """Download NDVI for specific dekad"""
    url, file_name = _get_paths(region=region, year=year, dekad=dekad)
    try:
        resp = urlopen(url)
    except HTTPError:
        logger.error(
            f"No NDVI data available for "
            f"dekad {dekad} of {year}, skipping."
        )
        return

    zf = ZipFile(BytesIO(resp.read()))
    save_path = Path(_RAW_DIR, file_name)
    save_path.unlink(missing_ok=True)

    zf.extract(file_name, _RAW_DIR)


def _date_to_dekad(date_obj: Union[date, str]):
    """Compute dekad from date

    Dekad computed from date. This
    is based on the USGS (and relatively
    common) dekadal definition of the
    1st and 2nd dekad of a month being
    the first 10 day periods, and the 3rd
    dekad being the remaining days within
    that month.
    """
    if isinstance(date_obj, str):
        date_obj = date.fromisoformat(date_obj)
    year = date_obj.year
    dekad = (date_obj.day // 10) + ((date_obj.month - 1) * 3) + 1
    return year, dekad


def _dekad_to_date(year: int, dekad: int):
    """Compute date from dekad and year

    Date computed from dekad and year in
    datetime object, corresponding to
    first day of the dekad. This
    is based on the USGS (and relatively
    common) dekadal definition of the
    1st and 2nd dekad of a month being
    the first 10 day periods, and the 3rd
    dekad being the remaining days within
    that month.
    """
    month = ((dekad - 1) // 3) + 1
    day = 10 * ((dekad - 1) % 3) + 1
    return datetime(year, month, day)


def _fp_date(fp):
    return _dekad_to_date(2000 + int(fp[2:4]), int(fp[4:6]))


def _get_processed_path(iso3: str) -> Path:
    _processed_dir = Path(
        os.getenv("AA_DATA_DIR"), "public", "processed", iso3.lower(), "ndvi"
    )
    _processed_dir.mkdir(parents=True, exist_ok=True)
    return _processed_dir / f"{iso3.lower()}_ndvi_anomaly.csv"
