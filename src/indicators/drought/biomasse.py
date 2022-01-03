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
using that. Since DMP measurements are productivity values,
measuring the daily dry matter productivity across the dekad of
interest, the calculation of total biomasse in a year is taken
by averaging the DMP over the entire period and multiplying by
365.25. The default year starts in the 10th dekad (1st of
April) until the last dekad of March, to correspond with the
West African monsoon season.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, get_args

import numpy as np
import pandas as pd
import rioxarray

from src.utils_general.utils import download_ftp

_BASE_URL = (
    "http://80.69.76.253:8080/geoserver"
    "/Biomass/wfs?&REQUEST="
    "GetFeature&SERVICE=wfs&VERSION=1.1.0"
    "&TYPENAME=WA_DMP_{admin}_ef_v0&"
    "outputformat=CSV&srsName=EPSG:4326"
)

_RAW_FILENAME = "WA_DMP_{admin}_ef_v0.csv"

_RAW_PATH = Path(os.getenv("AA_DATA_DIR"), "public", "raw", "glb", "biomasse")

AdminArgument = Literal["ADM0", "ADM1", "ADM2"]
_VALID_ADMIN = get_args(AdminArgument)

_PROCESSED_FILENAME = "biomasse_{iso3}_{admin}_dekad_{start_dekad}.csv"


def download_dmp(admin_level: AdminArgument = "ADM2") -> None:
    """Download raw DMP data

    Raw DMP data is downloaded for specified
    administrative level, either ADM0, ADM1,
    or ADM2. This data is downloaded
    for all countries in West Africa covered by
    Biomasse, using those specific countries'
    administrative boundaries, and is downloaded
    for all available years. The data is downloaded
    in CSV format and can be processed using
    ``calculate_biomasse()``. Since it is provided
    in a single file, the existing file is
    always overwritten.

    Parameters
    ----------
    admin_level: AdminArgument
        Admin area to load DMP for, one of
        'ADM0', 'ADM1', or 'ADM2'.

    Returns
    -------
    None
    """
    _check_admin(admin_level)
    url = _BASE_URL.format(admin=admin_level)
    raw_path = _get_raw_path(admin_level)
    download_ftp(url=url, save_path=raw_path)


def load_dmp(admin: AdminArgument = "ADM2") -> pd.DataFrame:
    """Load raw DMP data

    Parameters
    ----------
    admin: AdminArgument
        Admin area to load DMP for, one of
        'ADM0', 'ADM1', or 'ADM2'.

    Returns
    -------
    pd.DataFrame
    """
    _check_admin(admin)
    raw_path = _get_raw_path(admin)
    if not raw_path.is_file():
        raise OSError(
            "Raw DMP data not available, run `download_dmp()` first."
        )
    # na values set by Biomasse as -9998.8
    df = pd.read_csv(raw_path, na_values=["-9998.8"])
    df.dropna(axis="columns", how="all", inplace=True)
    return df


def calculate_biomasse(
    admin_level: AdminArgument = "ADM2",
    start_dekad: int = 10,
) -> pd.DataFrame:
    """Calculate Biomasse from DMP raw data

    DMP raw data is received in a wide format
    dataset that needs processing. This pivots
    the data from wide to long format with
    DMP dekadal observations for all years, while
    also pivoting the mean values to allow for
    calculation of biomasse anomaly. Outputs
    dataframe of Biomasse values.

    Parameters
    ----------
    admin_level: AdminArgument
        Admin area to load DMP for, one of
        'ADM0', 'ADM1', or 'ADM2'.
    start_dekad: int
        Starting dekad of the season to use
        in calculations. Season will start
        in starting dekad and end in the
        previous dekad.
    Returns
    -------
    pd.DataFrame
    """
    df = load_dmp(admin_level)

    # process mean and DMP separately since mean values
    # are dekadal and DMP are year/dekadal
    # keep ID column for working with multipolygon admin areas
    id_col = "IDBIOHYDRO"
    df_mean = df.filter(regex=f"(^admin|^DMP_MEA|^{id_col}|^AREA)")
    df_dmp = df.filter(regex=f"(^admin|^DMP_[0-9]+|^{id_col})")
    admin_cols = [col for col in df_mean.columns if col.startswith("admin")]

    # groupby to average out for the few cases where admin areas
    # are not contiguous, because noncontiguous polygons
    # appear as separate rows in the data frame
    # happens in Cote d'Ivoire and Liberia
    df_mean_long = (
        pd.wide_to_long(
            df_mean,
            i=admin_cols + [id_col],
            j="dekad",
            stubnames="DMP_MEA",
            sep="_",
        )
        .reset_index()
        .drop(labels=id_col, axis=1)
        .groupby(admin_cols + ["dekad"])
        .apply(
            lambda x: pd.Series(
                {"DMP_MEA": np.average(x["DMP_MEA"], weights=x["AREA"])}
            )
        )
        .reset_index()
    )

    # calculate anomaly for mean separate from
    # observed to ensure unique dekadal values
    # based on what we removed in the above code
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
            i=admin_cols + [id_col],
            j="time",
            stubnames="DMP",
            sep="_",
        )
        .reset_index()
        .drop(labels=id_col, axis=1)
        .groupby(admin_cols + ["time"])
        .apply(
            lambda x: pd.Series(
                {"DMP": np.average(x["DMP"], weights=x["AREA"])}
            )
        )
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
    df_merged["season"] = (
        df_merged["year"] + (df_merged["dekad"] >= start_dekad) - 1
    )
    # if the season isn't starting from 1
    # create clear season definition of year1-year2
    if start_dekad > 1:
        df_merged["season_end"] = (df["season"] + 1).astype("str")
        df_merged["season"] = df_merged["season"].astype("str")
        df_merged["season"] = df_merged[["season", "season_end"]].agg(
            "-".join, axis=1
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
            "season",
            "DMP_MEA",
            "DMP",
            "biomasse_mean",
            "biomasse",
            "biomasse_anomaly",
        ]
    ]

    # save file to processed filepath
    processed_path = _get_processed_path(
        admin=admin_level, start_dekad=start_dekad, iso3=None
    )
    df_merged.to_csv(processed_path, index=False)

    return df_merged


def load_biomasse_data(
    admin: AdminArgument = "ADM2",
    iso3: Optional[str] = None,
    start_dekad: int = 10,
):
    """Load biomasse processed data

    Load the data that has been processed at the
    specified ADM2 level and starting dekad.

    Parameters
    ----------
    admin: AdminArgument
        Admin level, one of 'ADM0', 'ADM1',
        or 'ADM2'.
    iso3: Optional[str]
        ISO3 string. If present, saves data
        within that countries folder for
        download.
    start_dekad: int
        Starting dekad for annual
        calculations.

    Returns
    -------
    pd.DataFrame
    """
    _check_admin(admin)
    processed_path = _get_processed_path(
        admin=admin, start_dekad=start_dekad, iso3=iso3
    )
    return pd.read_csv(processed_path)


def aggregate_biomasse(
    admin_pcodes: List[str],
    iso3: Optional[str],
    admin: AdminArgument = "ADM2",
    start_dekad: int = 10,
) -> pd.DataFrame:
    """Aggregate Biomasse data to set of areas

    While the generic processed administrative
    Biomasse data is useful, sometimes we want
    to aggregate the Biomasse values to a set
    of administrative areas. This should be done
    by summing up the mean biomasse for each dekad
    for those areas and the reported biomasse for
    each year and dekad, then recalculating the
    anomaly.

    Output is saved only if ``iso3`` is passed as
    an argument.

    Parameters
    ----------
    admin_pcodes: List[str]
        List of administrative pcodes to aggregate to.
    iso3: Optional[str]
        ISO3 string or ``None``. If present, saves data
        within that country's folder for download.
    admin: AdminArgument
        Admin type to subset with, one of 'ADM0', 'ADM1',
        or 'ADM2'.
    start_dekad: int
        Starting dekad for annual
        calculations.

    """
    _check_admin(admin)
    df = load_biomasse_data(admin=admin, start_dekad=start_dekad)
    admin_col = admin.replace("ADM", "admin") + "Pcod"
    df_subset = df[df[admin_col].isin(admin_pcodes)]
    # just keep to year 2000 for unique value per dekad
    # when calculating biomasse mean aggregated
    # since mean is the same each year
    df_mean = df_subset[df.year.isin([2000])]
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

    if iso3 is not None:
        processed_filepath = _get_processed_path(
            admin=admin, start_dekad=start_dekad, iso3=iso3
        )
        df_merged.to_csv(processed_filepath, index=False)

    return df_merged


def load_aggregated_biomasse_data(
    iso3: str, admin: AdminArgument, start_dekad: int = 10
) -> pd.DataFrame:
    """Load aggregated Biomasse data

    Parameters
    ----------
    iso3: str
        ISO3 string to download data from.
    admin: AdminArgument
        Admin type to load, one of 'ADM0', 'ADM1',
        or 'ADM2'.
    start_dekad: int
        Starting dekad for filename.
    Returns
    -------
    pd.DataFrame
    """
    processed_filepath = _get_processed_path(
        admin=admin, start_dekad=start_dekad, iso3=iso3
    )
    return pd.read_csv(processed_filepath)


def load_biomasse_mean(mask: bool = True):
    """Used to mask WRSI

    Static file loaded from
    http://www.geosahel.info/MetaDownload/BiomassValueMean.tif
    so didn't bother including code to download.
    """
    bm_fp = _RAW_PATH / "BiomassValueMean.tif"
    da = rioxarray.open_rasterio(bm_fp)
    return da
    if mask:
        da = da.where(da.values <= 0, drop=False)
    return da


def _get_raw_path(admin):
    return _RAW_PATH / _RAW_FILENAME.format(admin=admin)


def _get_processed_path(
    admin: AdminArgument, start_dekad: int, iso3: Optional[str] = None
):
    if iso3 is None:
        iso3 = "glb"

    _processed_path = Path(
        os.getenv("AA_DATA_DIR"), "public", "processed", iso3, "biomasse"
    )

    return Path(
        _processed_path,
        _PROCESSED_FILENAME.format(
            iso3=iso3, admin=admin, start_dekad=start_dekad
        ),
    )


def _check_admin(admin):
    if admin not in _VALID_ADMIN:
        raise ValueError("`admin` must be one of 'ADM0', 'ADM1', or 'ADM2'.")
