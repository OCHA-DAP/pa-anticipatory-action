"""Download and process Biomasse data

The Biomasse data is produced by Action Against Hunger, Spain,
and made available through the website http://geosahel.info/.
Documentation of the methodology is made available through
https://sigsahel.info/wp-content/uploads/2020/12/ACF_BioHydroGenerator_User_Guide.pdf.
The Biomasse analysis uses Copernicus Global Land Service dry matter
production (DMP) data: https://land.copernicus.eu/global/products/dmp.
It is a dekadal product and is updated around 1 week to
10 days after the end of the period.

While the calculated cumulative biomasse and anomaly data
is presented at a yearly level on the GeoSahel portal, it
is not available in dekadal form. The only dekadal form data
is for DMP values downsampled to the ADMIN0-2 levels across
the region of interested. The code below accesses this dekadal
data and then recalculates cumulative biomasse and anomaly
using that. Since DMP measurements are productivity values, the
calculation of total biomasse in a year is taken by averaging
the DMP over the entire period and multiplying by 365.25.
The default year starts in the 10th dekad (1st of April) until
the last dekad of March, to correspond with the West African
monsoon season.
"""

import os
from typing import List, Literal, Union, get_args

import numpy as np
import pandas as pd
import rioxarray

from src.utils_general.utils import download_ftp

_base_url = (
    "http://80.69.76.253:8080/geoserver"
    "/Biomass/wfs?&REQUEST="
    "GetFeature&SERVICE=wfs&VERSION=1.1.0"
    "&TYPENAME=WA_DMP_{admin}_ef_v0&"
    "outputformat=CSV&srsName=EPSG:4326"
)

_raw_filename = "WA_DMP_{admin}_ef_v0.csv"

_raw_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "raw", "general", "biomasse"
)


def _get_raw_path(admin):
    return os.path.join(_raw_path, _raw_filename.format(admin=admin))


_processed_filename = "biomasse_{admin}_dekad_{start_dekad}.csv"

_processed_path = os.path.join(
    os.getenv("AA_DATA_DIR"), "public", "processed", "general", "biomasse"
)


def _get_processed_path(admin, start_dekad):
    return os.path.join(
        _processed_path,
        _processed_filename.format(admin=admin, start_dekad=start_dekad),
    )


AdminArgument = Literal["ADM0", "ADM1", "ADM2"]
_valid_admin = get_args(AdminArgument)


def _check_admin(admin):
    if admin not in _valid_admin:
        raise ValueError("`admin` must be one of 'ADM0', 'ADM1', or 'ADM2'.")


def download_dmp(admin: AdminArgument = "ADM2"):
    """Download raw DMP data

    Raw DMP data is downloaded for specified
    administrative area. The data is downloaded
    in CSV format and can be processed using
    ``process_dmp()``.
    """
    _check_admin(admin)
    url = _base_url.format(admin=admin)
    raw_path = _get_raw_path(admin)
    download_ftp(url=url, save_path=raw_path)
    return None


def load_dmp(admin: AdminArgument = "ADM2", redownload: bool = False):
    _check_admin(admin)
    raw_path = _get_raw_path(admin)
    if not os.path.exists(raw_path):
        download_dmp(admin)
    elif redownload:
        os.remove(raw_path)
        download_dmp(admin)

    df = pd.read_csv(raw_path, na_values=["-9998.8"])
    df.dropna(axis="columns", how="all", inplace=True)
    return df


def process_dmp(
    admin: AdminArgument = "ADM2",
    redownload: bool = False,
    start_dekad: int = 10,
):
    """Process DMP raw data

    DMP raw data is received in a wide format
    dataset that needs processing. This pivots
    the data from wide to long format with
    DMP dekadal observations for all years, while
    also pivoting the mean values to allow for
    calculation of biomasse anomaly.
    """
    df = load_dmp(admin, redownload)

    # process mean and DMP separately since mean values
    # are dekadal and DMP are year/dekadal
    # keep ID column for working with multipolygon admin areas
    df_mean = df.filter(regex="(^admin|^DMP_MEA|^ID)")
    df_dmp = df.filter(regex="(^admin|^DMP_[0-9]+|^ID)")
    admin_cols = [col for col in df_mean.columns if col.startswith("admin")]

    # groupby to average out for the few cases where admin areas
    # are not contiguous
    df_mean_long = (
        pd.wide_to_long(
            df_mean,
            i=admin_cols + ["IDBIOHYDRO"],
            j="dekad",
            stubnames="DMP_MEA",
            sep="_",
        )
        .reset_index()
        .drop(labels="IDBIOHYDRO", axis=1)
        .groupby(admin_cols + ["dekad"])
        .mean()
        .reset_index()
    )

    # calculate anomaly for mean separate from
    # observed to ensure unique dekadal values
    df_mean_long["season_index"] = np.where(
        df_mean_long["dekad"] >= start_dekad,
        df_mean_long["dekad"] - start_dekad,
        df_mean_long["dekad"] + (36 - start_dekad),
    )
    df_mean_long.sort_values(by=admin_cols + ["season_index"], inplace=True)
    df_mean_long["biomasse_mean"] = df_mean_long.groupby(by=admin_cols)[
        "DMP_MEA"
    ].apply(lambda x: x.cumsum() * 365.25 / 36)

    df_dmp_long = (
        pd.wide_to_long(
            df_dmp,
            i=admin_cols + ["IDBIOHYDRO"],
            j="time",
            stubnames="DMP",
            sep="_",
        )
        .reset_index()
        .drop(labels="IDBIOHYDRO", axis=1)
        .groupby(admin_cols + ["time"])
        .sum()
        .reset_index()
    )

    # convert time into year and dekad
    df_dmp_long[["year", "dekad"]] = (
        df_dmp_long["time"].apply(str).str.extract("([0-9]{4})([0-9]{2})")
    )
    df_dmp_long = df_dmp_long.astype({"year": int, "dekad": int})

    # drop values from 1999 that are not from a complete season
    df_dmp_long = df_dmp_long[
        ~((df_dmp_long["year"] == 1999) & (df_dmp_long["dekad"] < start_dekad))
    ]

    # join mean and observed values
    df_merged = pd.merge(
        left=df_dmp_long, right=df_mean_long, on=admin_cols + ["dekad"]
    )

    # calculate biomasse (cumulative sum of DMP across the season
    # starting with the start_dekad and ending at start_dekad - 1
    df_merged.sort_values(
        by=admin_cols + ["year", "dekad"], inplace=True, ignore_index=True
    )
    df_merged["season"] = df_merged["year"] + (
        df_merged["dekad"] >= start_dekad
    )
    df_merged["biomasse"] = df_merged.groupby(by=admin_cols + ["season"])[
        "DMP"
    ].apply(lambda x: x.cumsum() * 365.25 / 36)
    df_merged["biomasse_anomaly"] = (
        100 * df_merged["biomasse"] / df_merged["biomasse_mean"]
    )

    # re-arrange columns
    df_merged = df_merged[
        admin_cols
        + [
            "year",
            "dekad",
            "DMP_MEA",
            "DMP",
            "biomasse_mean",
            "biomasse",
            "biomasse_anomaly",
        ]
    ]

    # save file to processed filepath
    processed_path = _get_processed_path(admin, start_dekad)
    df_merged.to_csv(processed_path, index=False)

    return df_merged


def load_biomasse_data(
    admin: AdminArgument = "ADM2",
    start_dekad: int = 10,
    reprocess: bool = False,
    redownload: bool = False,
):
    """Load biomasse processed data

    Load the data that has been processed at the
    specified ADM2 level and starting dekad.

    Args:
        admin: Admin level, one of 'ADM0', 'ADM1',
            or 'ADM2'.
        start_dekad: Starting dekad for annual
            calculations.
        reprocess: Boolean, whether to reprocess.
        redownload: Boolean, whether to reprocess.
        file_descriptor: If not None, file descriptor
            to access non-admin file.

    Returns:
        Pandas data frame.
    """
    if reprocess:
        return process_dmp(
            admin=admin, start_dekad=start_dekad, redownload=redownload
        )
    else:
        _check_admin(admin)
        processed_path = _get_processed_path(admin, start_dekad)
        return pd.read_csv(processed_path)


def aggregate_biomasse(
    admin_pcodes: List[str],
    admin: AdminArgument = "ADM2",
    start_dekad: int = 10,
    file_descriptor: Union[None, str] = None,
):
    """Aggregate biomasse data to set of areas

    While the generic processed administrative
    biomasse data is useful, sometimes we want
    to aggregate the biomasse values to a set
    of administrative areas. This should be done
    by summing up the mean biomasse for each dekad
    for those areas and the reported biomasse for
    each year and dekad, then recalculating the
    anomaly.
    """
    _check_admin(admin)
    df = load_biomasse_data(admin=admin, start_dekad=start_dekad)
    admin_col = admin.replace("ADM", "admin") + "Pcod"
    df_subset = df[df[admin_col].isin(admin_pcodes)]
    # just keep to year 2000 for unique value per dekad
    # when calculating biomasse mean aggregated
    df_mean = df_subset[df["year"] == 2000]
    df_mean = (
        df_mean[["dekad", "biomasse_mean"]]
        .groupby("dekad")
        .sum()
        .reset_index()
    )
    df_dekad = (
        df_subset[["year", "dekad", "biomasse"]]
        .groupby(["year", "dekad"])
        .sum()
        .reset_index()
    )
    df_merged = pd.merge(df_dekad, df_mean, on=["dekad"])
    df_merged["biomasse_anomaly"] = (
        100 * df_merged["biomasse"] / df_merged["biomasse_mean"]
    )

    # sorting data again year then dekad
    df_merged.sort_values(by=["year", "dekad"], inplace=True)

    if file_descriptor is not None:
        processed_filepath = _get_processed_path(
            admin=file_descriptor, start_dekad=start_dekad
        )
        df_merged.to_csv(processed_filepath, index=False)

    return df_merged


def load_aggregated_biomasse_data(file_descriptor: str, start_dekad: int = 10):
    processed_filepath = _get_processed_path(
        admin=file_descriptor, start_dekad=start_dekad
    )
    return pd.read_csv(processed_filepath)


def load_biomasse_mean(mask: bool = True):
    """Used to mask WRSI"""
    bm_fp = os.path.join(_raw_path, "BiomassValueMean.tif")
    da = rioxarray.open_rasterio(bm_fp)
    if mask:
        da = da.where(da.values <= 0, drop=True)
    return da
