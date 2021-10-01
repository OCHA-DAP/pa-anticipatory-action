"""
Computes the trigger status of the predictive trigger on
dry spells in Malawi, version1
Downloads and processes ecmwf seasonal forecast from CDS
and then computes the trigger status per admin
"""
import logging
import os
import sys
from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# TODO: remove this after making top-level
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.indicators.drought.ecmwf_seasonal.processing import (
    compute_stats_per_admin,
    get_stats_filepath,
)
from src.utils_general.area import AreaFromShape

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

COUNTRY_ISO3 = "mwi"
# version number of the trigger
# this script is written for v1
VERSION = 1
USE_INCORRECT_AREA_COORDS = False
CONFIG = Config()
PARAMETERS = CONFIG.parameters(COUNTRY_ISO3)

COUNTRY_DATA_RAW_DIR = (
    Path(CONFIG.DATA_DIR) / CONFIG.PUBLIC_DIR / CONFIG.RAW_DIR / COUNTRY_ISO3
)

COUNTRY_DATA_PROCESSED_DIR = (
    Path(CONFIG.DATA_DIR)
    / CONFIG.PUBLIC_DIR
    / CONFIG.PROCESSED_DIR
    / COUNTRY_ISO3
)


def get_output_path(
    iso3: str, version: int, target_year: str, target_month: str
):
    directory = (
        COUNTRY_DATA_PROCESSED_DIR
        / CONFIG.DRY_SPELLS_DIR
        / f"v{version}"
        / CONFIG.TRIGGER_METRICS_DIR
        / "predictive_trigger"
    )
    filename = f"{iso3}_predictive_trigger_{target_year}{target_month}.csv"
    return directory / filename


def retrieve_forecast(
    iso3: str,
    gdf_bound: gpd.GeoSeries,
    target_date: datetime,
    adm_level: int,
    pcode_col: str,
    leadtimes: List[int],
    add_col: List[str] = None,
):
    """
    Download and process the raw forecasts,
    and aggregate to the given adm_level
    :param iso3: country iso3 code
    :param gdf_bound: containing the bounds of the area that
    should be downloaded
    :param target_date: date the forecasts should predict (year-month)
    :param adm_level: admin level to aggregate to
    :param pcode_col: name of column that contains pcode in gdf_bound
    :param leadtimes: list of leadtimes to get data for
    :param add_col: additional columns in gdf_bound that should be added to the
    output of compute_stats_admin
    """
    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast(
        use_incorrect_area_coords=USE_INCORRECT_AREA_COORDS
    )
    # add buffer
    # not in correct crs for it to do properly
    # but not important in this case as we just want some extra area
    area = AreaFromShape(gdf_bound.buffer(3))

    year_start = (target_date - relativedelta(months=6)).year

    months = [
        (target_date - relativedelta(months=lt)).month for lt in leadtimes
    ]

    # this will download the months for year_start and target_date.year
    # so some of that data might not be needed, but that is okay
    ecmwf_forecast.download(
        country_iso3=iso3,
        area=area,
        year_min=year_start,
        year_max=target_date.year,
        months=months,
    )

    ecmwf_forecast.process(country_iso3=iso3)
    compute_stats_per_admin(
        iso3=iso3,
        interpolate=False,
        date_list=[target_date.strftime("%Y-%m-%d")],
        adm_level=adm_level,
        pcode_col=pcode_col,
        add_col=add_col,
        # do not use cache as new leadtimes can be added
        use_cache=False,
    )


def compute_trigger(
    iso3: str,
    target_date: datetime,
    min_prob: float,
    precip_cap: int,
    download: bool = True,
    interpolate_raster=False,
    leadtimes: List[int] = None,
    pcodes: List[str] = None,
    adm_level: int = 1,
    stats_col: str = "mean_ADM1_PCODE",
    pcode_col: str = "ADM1_PCODE",
    adm_name_col: str = "ADM1_EN",
    date_col: str = "date",
    leadtime_col: str = "leadtime",
):
    """
    Compute the trigger metric and a binary true/false if trigger is met
    The logic assumes that the trigger is defined as being met if there is
    >= min_prob probability of <= precip_cap monthly preciptiation (mm)
    :param iso3: country iso3 code
    :param target_date:  date the forecasts should predict (year-month)
    :param min_prob: minimum probability of the forecast for the trigger
    to be met. should be between 0 and 1
    :param precip_cap: max precipitation of the forecast for the trigger
    to be met. Defined as monthly precipitation in milimeters
    :param download: if True, download and process new data
    :param interpolate_raster: if True, interpolate the original raster
    to a higher resolution
    :param leadtimes: list of leadtimes to compute the trigger for
    :param pcodes: list of pcodes to compute the trigger for
    :param adm_level: admin level to aggregate to
    :param stats_col: column in the stats file that contains
    the statistic that should be used for the trigger
    :param pcode_col: column in the shapefile that contains the
    pcode
    :param adm_name_col: column in the shapefile that contains
    the admin name
    :param date_col: column in the stats file that contains the date
    :param leadtime_col: column in the stats file that contains the leadtime
    :return:
    """

    adm_bound_path = (
        Path(COUNTRY_DATA_RAW_DIR)
        / CONFIG.SHAPEFILE_DIR
        / PARAMETERS[f"path_admin{adm_level}_shp"]
    )
    gdf_adm = gpd.read_file(adm_bound_path)

    if download:
        retrieve_forecast(
            iso3,
            gdf_bound=gdf_adm,
            target_date=target_date,
            adm_level=adm_level,
            pcode_col=pcode_col,
            leadtimes=range(0, 7) if leadtimes is None else leadtimes,
            add_col=[adm_name_col],
        )

    stats_filename = get_stats_filepath(
        iso3,
        CONFIG,
        target_date,
        interpolate=interpolate_raster,
        adm_level=adm_level,
    )
    df_stats = pd.read_csv(stats_filename, parse_dates=[date_col])
    # for earlier dates, the model included less members -->
    # values for those members are nan --> remove those rows
    df_stats = df_stats[df_stats[stats_col].notna()]

    if pcodes is not None:
        df_stats = df_stats[df_stats[pcode_col].isin(pcodes)]
    if leadtimes is not None:
        df_stats = df_stats[df_stats[leadtime_col].isin(leadtimes)]

    # compute the value for which x% of members forecasts
    # the precipitation to be below or equal to that value
    df_stats_quant = df_stats.groupby(
        [date_col, pcode_col, adm_name_col, leadtime_col], as_index=False
    ).quantile(min_prob)
    df_stats_quant["date_month"] = df_stats_quant.date.dt.to_period("M")
    df_stats_quant[f"below_{precip_cap}"] = np.where(
        df_stats_quant[stats_col] <= precip_cap, 1, 0
    )
    df_stats_quant["trigger_met"] = np.where(
        df_stats_quant[f"below_{precip_cap}"] == 1, True, False
    )
    logging.debug(df_stats_quant)
    df_stats_quant = df_stats_quant[
        [
            "date_month",
            pcode_col,
            adm_name_col,
            leadtime_col,
            stats_col,
            f"below_{precip_cap}",
            "trigger_met",
        ]
    ]

    output_path = get_output_path(
        iso3, VERSION, target_date.year, target_date.month
    )
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    df_stats_quant.to_csv(output_path, index=False)

    return df_stats_quant


def main():
    compute_trigger(
        COUNTRY_ISO3,
        target_date=pd.to_datetime("2020-01-01"),
        min_prob=0.5,
        precip_cap=210,
        download=False,
        interpolate_raster=False,
    )


if __name__ == "__main__":
    main()
