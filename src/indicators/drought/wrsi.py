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
calculations for rangeland and cropland. Since most
countries in the Sahel overlap with both datasets, both
should be considered and we should get clarification from
experts on the differences between the two, how they relate,
and how best to use them.

I have also just found the CHIRPS WRSI data on the USGS website.
These are available at:

https://edcftp.cr.usgs.gov/project/fews/africa/west/dekadal/wrsi-chirps-etos
"""
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

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

_base_url = (
    "https://edcftp.cr.usgs.gov/project/fews/"
    "africa/west/dekadal/wrsi-chirps-etos/{region}/"
)

_url_zip_filename = "w{year:04}{dekad}{region}.zip"

_base_file_name = "w{year:04}{dekad}{type}.tif"


def _get_url_path(year, dekad, region):
    if region == "cropland":
        url_region, file_region = "west", "wa"
    else:
        url_region, file_region = "west1", "w1"

    url_zip_filename = _url_zip_filename.format(
        year=year, dekad=dekad, region=file_region
    )

    base_url = _base_url.format(region=url_region)
    return base_url + url_zip_filename


def _download_wrsi(year: int, dekad: int, region: str, raw_dir: Path):
    try:
        url = _get_url_path(year=year, dekad=dekad, region=region)
        resp = urlopen(url)
    except HTTPError:
        logger.error(
            f"No data available for region {region} for "
            f"dekad {dekad} of {year}, skipping."
        )
        return

    zf = ZipFile(BytesIO(resp.read()))
    for type in ["do", "eo", "er"]:
        file_name = _base_file_name.format(year=year, dekad=dekad, type=type)
        save_path = os.path.join(raw_dir, file_name)
        if os.path.exists(save_path):
            os.remove(save_path)

        zf.extract(file_name, raw_dir)


_raw_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "raw", "general", "wrsi", "west_africa"
)


def _get_raw_dir(region):
    return Path(os.path.join(_raw_path, region))


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
    return datetime.datetime(year, month, day)


def _fp_date(fp):
    return _dekad_to_date(int(fp[1:5]), int(fp[5:7]))


RegionArgument = Literal["cropland", "rangeland"]
_valid_region = get_args(RegionArgument)


def download_wrsi(
    region: RegionArgument,
    start_date: Union[date, str, List[int], None] = None,
    end_date: Union[date, str, List[int], None] = None,
    redownload: bool = False,
):
    if region not in _valid_region:
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

    i_year = start_year
    i_dekad = min(start_dekad, 33)
    raw_dir = _get_raw_dir(region)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not redownload:
        dts = [
            _fp_date(filename)
            for filename in os.listdir(raw_dir)
            if filename.endswith(".tif")
        ]
        if dts:
            min_year, min_dekad = _date_to_dekad(min(dts))
            max_year, max_dekad = _date_to_dekad(max(dts))
            if (start_year < min_year) or (
                (min_year == start_year) and start_dekad < min_dekad
            ):
                if min_dekad == 1:
                    min_dekad = 33
                    min_year -= 1
                else:
                    min_dekad -= 1
                download_wrsi(
                    region=region,
                    start_date=[start_year, start_dekad],
                    end_date=[min_year, min_dekad],
                    redownload=True,
                )
            if (end_year > max_year) or (
                (max_year == end_year) and end_dekad > max_dekad
            ):
                if max_dekad == 33:
                    max_dekad = 1
                    max_year += 1
                else:
                    max_dekad += 1
                download_wrsi(
                    region=region,
                    start_date=[max_year, max_dekad],
                    end_date=[end_year, end_dekad],
                    redownload=True,
                )
    else:
        while not (
            i_year > end_year or (i_year == end_year and i_dekad > i_dekad)
        ):
            _download_wrsi(
                year=i_year, dekad=i_dekad, region=region, raw_dir=raw_dir
            )
            i_dekad += 1
            if i_dekad > 33:
                i_year += 1
                i_dekad = 13


_processed_filename = "wrsi_{region}_{type}.nc"

_processed_dir = Path(
    os.path.join(
        os.getenv("AA_DATA_DIR"), "public", "processed", "general", "wrsi"
    )
)


def _get_processed_path(region, type):
    _processed_dir.mkdir(parents=True, exist_ok=True)
    return os.path.join(
        _processed_dir,
        _processed_filename.format(region=region, type=type),
    )


# defined here: https://gis.stackexchange.com/questions/
# 177447/defining-the-correct-aea-proj4string-for-
# fewsnet-rainfall-data-southern-africa
_wrsi_crs = CRS.from_proj4(
    "+proj=aea +lat_1=-19.0 +lat_2=21.0 +lat_0=1.0 "
    "+lon_0=20 +x_0=0 +y_0=0 +ellps=clrk66 +units=m +no_defs"
)


def process_wrsi(region, type):
    if region not in _valid_region:
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

    def _load_raw(fp):
        arr = xr.open_rasterio(os.path.join(raw_dir, fp))
        dt = [_fp_date(fp)]

        arr = arr.expand_dims(time=dt)
        # round x dimension due to issues with
        # concat and tiny numerical differences
        arr = arr.assign_coords({"x": np.round(arr.x.values, 7)})
        return arr

    arrays = [
        _load_raw(filename)
        for filename in os.listdir(raw_dir)
        if filename.endswith(fp_type + ".tif")
    ]

    arrays_merged = xr.concat(arrays, "time").sortby("time")

    # change missing values from 0 to nan
    arrays_merged = arrays_merged.where(arrays_merged.values > 0)
    arrays_merged.attrs["_FillValue"] = np.NaN

    # convert to dataset
    ad = arrays_merged.to_dataset("band")
    ad = ad.rename({1: "wrsi"})

    # saving file
    if os.path.exists(processed_path):
        os.remove(processed_path)

    ad.to_netcdf(processed_path)

    return ad


def load_wrsi(region, type):
    processed_path = _get_processed_path(region, type)
    a = xr.open_dataset(processed_path).to_array("wrsi")
    a = a.squeeze(drop=True)
    a.rio.write_crs(_wrsi_crs, inplace=True)
    a = a.rio.reproject("EPSG:4326")
    return a


def filter_wrsi(da, dekad):
    dt = _dekad_to_date(2001, dekad)
    filter_month = dt.month
    filter_day = dt.day
    return da.where(
        (da.time.dt.day == filter_day) & (da.time.dt.month == filter_month),
        drop=True,
    )
