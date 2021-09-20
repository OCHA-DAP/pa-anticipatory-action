# notes:
# we are now first merging all grib files to one nc
# and using that nc to compute admin stats
# it might be neater and more efficient to convert
# each grib file to a separate nc instead

import logging
import os
import sys
from typing import List
from datetime import datetime


# TODO: remove this after making top-level
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

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
CONFIG = Config()
PARAMETERS = CONFIG.parameters(COUNTRY_ISO3)

COUNTRY_DATA_RAW_DIR = (
    Path(CONFIG.DATA_DIR) / CONFIG.PUBLIC_DIR / CONFIG.RAW_DIR / COUNTRY_ISO3
)


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

    year_start = target_date.year + (
        (target_date.month - max(leadtimes)) // 12
    )

    months = [(target_date.month - lt) % 12 for lt in leadtimes]

    # this will download the months for year_start and target_date.year
    # so some duplication, but that is okay
    ecmwf_forecast.download(
        country_iso3=iso3,
        area=area,
        year_min=year_start,
        year_max=target_date.year,
        months=months,
    )
    # print(os.path.getmtime(ecmwf_seasonal.get_raw)
    # # #todo: would be nice to only do process if new data is downloaded
    # ecmwf_forecast.process(country_iso3=iso3)
    # do not use cache as new leadtimes can be added
    compute_stats_per_admin(
        iso3=iso3,
        interpolate=False,
        date_list=[target_date.strftime("%Y-%m-%d")],
        adm_level=adm_level,
        pcode_col=pcode_col,
        add_col=add_col,
        use_cache=False,
    )


def compute_trigger(
    iso3,
    target_date: datetime,
    min_prob,
    precip_cap,
    interpolate_raster=False,
    leadtimes: List[int] = None,
    pcodes: List[str] = None,
    adm_level=1,
    aggr_meth="mean_ADM1_PCODE",
    pcode_col="ADM1_PCODE",
    adm_name_col="ADM1_EN",
    date_col="date",
    leadtime_col="leadtime",
    # todo: check more general debug flag
    debug=False,
):

    adm_bound_path = (
        Path(COUNTRY_DATA_RAW_DIR)
        / CONFIG.SHAPEFILE_DIR
        / PARAMETERS[f"path_admin{adm_level}_shp"]
    )
    gdf_adm = gpd.read_file(adm_bound_path)

    retrieve_forecast(
        iso3,
        gdf_bound=gdf_adm,
        target_date=target_date,
        adm_level=adm_level,
        pcode_col=pcode_col,
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
    if not debug:
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
    print(df_stats_quant)
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
    interpolate_raster = False
    # todo: check if better method than pd.to_datetime
    compute_trigger(
        COUNTRY_ISO3,
        pd.to_datetime("2020-01-01"),
        0.5,
        210,
        interpolate_raster=interpolate_raster,
    )


if __name__ == "__main__":
    main()
