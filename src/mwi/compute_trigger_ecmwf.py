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


def get_ouput_path(
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
    ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast()
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
    aggr_meth: str = "mean_ADM1_PCODE",
    pcode_col: str = "ADM1_PCODE",
    adm_name_col: str = "ADM1_EN",
    date_col: str = "date",
    leadtime_col: str = "leadtime",
):

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
    df_stats = df_stats[df_stats[aggr_meth].notna()]

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
        df_stats_quant[aggr_meth] <= precip_cap, 1, 0
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
            aggr_meth,
            f"below_{precip_cap}",
            "trigger_met",
        ]
    ]

    output_path = get_ouput_path(
        iso3, VERSION, target_date.year, target_date.month
    )
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    df_stats_quant.to_csv(output_path, index=False)

    return df_stats_quant


#
# def plot_ensemble(
#     forec_year,
#     forec_month,
#     aggr_meth="mean_cell",
# ):
#     stats_filename = (
#         ECMWF_PROCESSED_DIR
#         / f"mwi_seasonal-monthly-single-levels_v5_"
#           f"{forec_year}_{forec_month}_test.csv"
#     )
#     df_stats = pd.read_csv(stats_filename, parse_dates=[date_col])
#     # for earlier dates, the model included less members --> values
#     # for those members are nan --> remove those rows
#     df_stats = df_stats[df_stats[aggr_meth].notna()]
#     plt.plot()
#
#     # Slice time and get mean of ensemble members for simple plotting
#     start = "2020-01-01"
#
#     rf_list_slice = da_for_clip.sel(
#         time=start,
#         latitude=da_for_clip.latitude.values[3],
#         longitude=da_for_clip.longitude.values[2],
#     )
#
#     rf_list_slice.dropna("leadtime").plot.line(
#         label="Historical", c="grey", hue="number", add_legend=False
#     )
#     rf_list_slice.dropna("leadtime").mean(dim="number").plot.line(
#         label="Historical", c="red", hue="number", add_legend=False
#     )
#     plt.show()


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
