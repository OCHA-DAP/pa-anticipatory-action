"""Code to download and process NDVI data

NDVI data is downloadable and available through the USGS
with a description available at
https://earlywarning.usgs.gov/fews/product/451 and the
actual data from the USGS file explorer:
https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/africa/west/dekadal/emodis/ndvi_c6/temporallysmoothedndvi/downloads/monthly/

The products include temporally smoothed NDVI, median anomaly,
difference from the previous year, and median anomaly
presented as a percentile. For the current exploration,
will just use percent of median because it's similar
to the % anomaly products we are comparing to for Biomasse
and WRSI in Chad. However, can expand this code more easily in
the future to download and process other NDVI data.

Data is published with about 1 month delay from the end of the
dekad.
"""
import logging
import os
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import List, Union
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/"
    "web/africa/west/dekadal/emodis/ndvi_c6/percentofmedian/downloads/dekadal/"
)

_BASE_FILENAME = "wa{year:02}{dekad:02}pct"

_RAW_DIR = Path(os.getenv("AA_DATA_DIR"), "public", "raw", "glb", "ndvi")


def _get_paths(year, dekad):
    base_file_name = _BASE_FILENAME.format(year=year % 100, dekad=dekad)
    url_zip_filename = base_file_name + ".zip"
    url_path = _BASE_URL + url_zip_filename
    file_name = base_file_name + ".tif"
    return url_path, file_name


def _download_ndvi(year: int, dekad: int):
    url, file_name = _get_paths(year, dekad)
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
    if isinstance(date_obj, str):
        date_obj = date.fromisoformat(date_obj)
    year = date_obj.year
    dekad = (
        (date_obj.timetuple().tm_mday // 10) + ((date_obj.month - 1) * 3) + 1
    )
    return year, dekad


def _dekad_to_date(year: int, dekad: int):
    month = ((dekad - 1) // 3) + 1
    day = 10 * ((dekad - 1) % 3) + 1
    return datetime(year, month, day)


def _fp_date(fp):
    return _dekad_to_date(2000 + int(fp[2:4]), int(fp[4:6]))


def download_ndvi(
    start_date: Union[date, str, List[int], None] = None,
    end_date: Union[date, str, List[int], None] = None,
    redownload: bool = False,
):
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

    i_year = start_year
    i_dekad = start_dekad

    if not redownload:
        dts = [_fp_date(filename.stem) for filename in _RAW_DIR.glob("*.tif")]
        if dts:
            min_year, min_dekad = _date_to_dekad(min(dts))
            max_year, max_dekad = _date_to_dekad(max(dts))
            if (start_year < min_year) or (
                (min_year == start_year) and start_dekad < min_dekad
            ):
                if min_dekad == 1:
                    min_dekad = 36
                    min_year -= 1
                else:
                    min_dekad -= 1
                download_ndvi(
                    start_date=[start_year, start_dekad],
                    end_date=[min_year, min_dekad],
                    redownload=True,
                )
            if (end_year > max_year) or (
                (max_year == end_year) and end_dekad > max_dekad
            ):
                if max_dekad == 36:
                    max_dekad = 1
                    max_year += 1
                else:
                    max_dekad += 1
                download_ndvi(
                    start_date=[max_year, max_dekad],
                    end_date=[end_year, end_dekad],
                    redownload=True,
                )
    else:
        while not (
            i_year > end_year or (i_year == end_year and i_dekad > i_dekad)
        ):
            _download_ndvi(year=i_year, dekad=i_dekad)
            i_dekad += 1
            if i_dekad > 36:
                i_year += 1
                i_dekad = 1


def _get_processed_path(iso3):
    _processed_dir = Path(
        Path(os.getenv("AA_DATA_DIR"), "public", "processed", iso3, "ndvi")
    )
    _processed_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        _processed_dir,
        "ndvi_anomaly_{iso3}.nc".format(iso3=iso3),
    )


def process_ndvi(iso3, geometries):
    processed_path = _get_processed_path(iso3)

    def _load_raw(fp):
        arr = xr.open_rasterio(Path(_RAW_DIR, fp))
        dt = [_fp_date(fp.stem)]

        arr = arr.expand_dims(time=dt)
        # much more time costly to merge all and then clip
        arr = arr.rio.clip(geometries, drop=True, from_disk=True)
        # round x dimension due to issues with
        # concat and tiny numerical differences
        # arr = arr.assign_coords({"x": np.round(arr.x.values, 7)})
        return arr

    arrays = [_load_raw(filename) for filename in _RAW_DIR.glob("*.tif")]

    arrays_merged = xr.concat(arrays, "time").sortby("time")

    # change missing values from 0 to nan
    arrays_merged = arrays_merged.where(arrays_merged.values < 255)
    arrays_merged.attrs["_FillValue"] = np.NaN

    # convert to dataset
    ad = arrays_merged.to_dataset("band")
    ad = ad.rename({1: "ndvi"})

    # saving file
    processed_path.unlink(missing_ok=True)

    ad.to_netcdf(processed_path)

    return ad


def process_ndvi_es(iso3, geometries):
    processed_path = _get_processed_path(iso3)

    def _load_raw(fp):
        arr = xr.open_rasterio(Path(_RAW_DIR, fp))
        dt = [_fp_date(fp.stem)]

        arr = arr.expand_dims(time=dt)
        # much more time costly to merge all and then clip
        arr = arr.rio.clip(geometries, drop=True)
        # round x dimension due to issues with
        # concat and tiny numerical differences
        # arr = arr.assign_coords({"x": np.round(arr.x.values, 7)})
        return arr

    arrays = [_load_raw(filename) for filename in _RAW_DIR.glob("*.tif")]

    arrays_merged = xr.concat(arrays, "time").sortby("time")

    # change missing values from 0 to nan
    arrays_merged = arrays_merged.where(arrays_merged.values < 255)
    arrays_merged.attrs["_FillValue"] = np.NaN

    # convert to dataset
    ad = arrays_merged.to_dataset("band")
    ad = ad.rename({1: "ndvi"})

    # saving file
    processed_path.unlink(missing_ok=True)

    ad.to_netcdf(processed_path)

    return ad
