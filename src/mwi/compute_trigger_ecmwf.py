"""
Computes the trigger status of the predictive trigger on
dry spells in Malawi, version1
Downloads and processes ecmwf seasonal forecast from CDS
and then computes the trigger status per admin
"""
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# TODO: remove this after making top-level
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.indicators.drought.ecmwf_seasonal.processing import (
    compute_stats_per_admin,
    get_ecmwf_forecast_by_leadtime,
    get_stats_filepath,
)
from src.utils_general.area import AreaFromShape

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

COUNTRY_ISO3 = "mwi"
# version number of the trigger
# this script is written for v1
VERSION = 1
# when True, the area coordinates for which to retrieve the forecast
# are not rounded to integers
# since the original forecast is produced for integer coordinates,
# in this case the data is automatically interpolated by CDS
# for now we are sticking to using unrounded coords as that was
# the method used during the development of the trigger
# however for the future, we recommend using the rounded coordinates
USE_UNROUNDED_AREA_COORDS = True

# Define parameters for plotting a map
# question: not sure if it is best to define them as global constants
# or give them as input to compute_metrics()
# reason I defined them here is because the number of args for
# compute_metrics() is already massive
SLICE_LON = slice(32, 37)
SLICE_LAT = slice(-9, -19)
# bins are left-inclusive, i.e. if value is 150,
# it will fall in the 150-210.1 bin, not the 100-150
# therefore use 210.1 instead of 210 as boundary
BINS = [0, 50, 100, 150, 210.1, 250, 300, 350]

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


def get_output_path_metrics(
    iso3: str, version: int, target_year: str, target_month: str
):
    directory = (
        COUNTRY_DATA_PROCESSED_DIR
        / CONFIG.DRY_SPELLS_DIR
        / f"v{version}"
        / CONFIG.TRIGGER_METRICS_DIR
        / "predictive_trigger"
    )
    filename = f"{iso3}_predictive_trigger_{target_year}{target_month}"
    if USE_UNROUNDED_AREA_COORDS:
        filename += "_unrounded-coords"
    filename += ".csv"
    return directory / filename


def _get_output_path_map(
    iso3: str,
    version: int,
    target_year: str,
    target_month: str,
    leadtime: str,
):
    directory = (
        COUNTRY_DATA_PROCESSED_DIR
        / CONFIG.PLOT_DIR
        / CONFIG.DRY_SPELLS_DIR
        / f"v{version}"
        / CONFIG.TRIGGER_METRICS_DIR
        / "predictive_trigger"
    )
    filename = (
        f"{iso3}_predictive_trigger_map_"
        f"{target_year}{target_month}_lt{leadtime}"
    )
    if USE_UNROUNDED_AREA_COORDS:
        filename += "_unrounded-coords"
    filename += ".png"
    output_path = directory / filename
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    return output_path


def retrieve_forecast(
    iso3: str,
    gdf_bound: gpd.GeoSeries,
    target_date: date,
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
        use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS
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
    new_data = ecmwf_forecast.download(
        country_iso3=iso3,
        area=area,
        year_min=year_start,
        year_max=target_date.year,
        months=months,
    )
    if new_data:
        # this takes a few minutes, so only recompute
        # if new data has been downloaded
        ecmwf_forecast.process(country_iso3=iso3)
        compute_stats_per_admin(
            iso3=iso3,
            interpolate=False,
            date_list=[target_date.strftime("%Y-%m-%d")],
            adm_level=adm_level,
            pcode_col=pcode_col,
            add_col=add_col,
            use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS,
            # do not use cache as new leadtimes can be added
            use_cache=False,
        )


def create_map(
    iso3: str,
    target_date: date,
    prob: float,
    round_precip_int: bool = True,
    leadtimes: List[int] = None,
    gdf_adm: gpd.GeoDataFrame = None,
    slice_lon: slice = None,
    slice_lat: slice = None,
    bins: List[float] = None,
    figsize: tuple = (6, 10),
):
    """
    Produce a map showing the forecasted values.

    Parameters
    ----------
    iso3: str
        iso3 code of the country of interest
    target_date: date
        date the forecasts should predict (year-month)
    prob: float
        minimum probability of the forecast for the trigger
        to be met. should be between 0 and 1
    round_precip_int : bool
        If True, round the values in the dataarray to the closest
        integer before plotting
    leadtimes : List[int]
        list of leadtimes to produce a map for
    gdf_adm : gpd.GeoDataFrame
        A geodataframe containing the polygons for which the boundaries
        should be plotted on the map. If None, only plot the raster data
    slice_lon : slice
        Only plot longitudes within the slice. If None, plot all longitudes
        in the dataarray
    slice_lat : slice
        Only plot latitudes within the slice. If None, plot all latitudes
        in the dataarray
    bins : List[float]
        Generate discrete colorbins in the plot separated by the boundaries
        in `bins`. These boundaries are left-inclusive .
        e.g. if bins=[0,50,100] a value of 50 will be placed
        in the [50,100] bin.
        If None, the bins will be automatically generated
    figsize : tuple
        Size of the figure
    """
    da_for = get_ecmwf_forecast_by_leadtime(
        iso3, use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS
    )
    da_for_date = da_for.sel(time=target_date.strftime("%Y-%m-%d"))
    if round_precip_int:
        # round values to closest integer
        da_for_date = da_for_date.round(0)
    # get the threshold for which prob % ensemble members
    # predict smaller or equal to that amount
    da_for_date_quant = da_for_date.quantile(prob, dim="number")

    if slice_lon is not None:
        da_for_date_quant = da_for_date_quant.sel(longitude=slice_lon)
    if slice_lat is not None:
        da_for_date_quant = da_for_date_quant.sel(latitude=slice_lat)

    if leadtimes is None:
        leadtimes = da_for_date_quant.leadtime.values
    for lt in leadtimes:
        da_for_date_quant_lt = da_for_date_quant.sel(leadtime=lt)
        if da_for_date_quant_lt.isnull().values.all():
            logger.info(
                f"No data available for {target_date} with leadtime {lt} "
                f"to produce a map. Skipping to next leadtime"
            )
            continue
        g = da_for_date_quant_lt.plot(
            figsize=figsize,
            levels=bins,
            # extend the colorbar with a pointy tip to indicate
            # any values above the highest value
            # doing this to make sure the colorscheme aligns
            # given the same set of bins,
            # regardless of the max value in the data
            # could also make this an input var,
            # but want to limit the number of input vars
            extend="max",
            cmap="RdBu",
            cbar_kwargs={
                "label": "forecasted precipitation (mm)",
                "shrink": 0.6,
                "pad": 0.1,
            },
        )
        # question: should we include "with {prob}% probability"
        # in the title or is it too much?
        published_month = (
            target_date + relativedelta(months=-(lt - 1))
        ).strftime("%Y-%m")
        g.axes.set_title(
            f"Forecasted monthly precipitation with {int(prob*100)}% "
            "probability for {target_date.strftime('%Y-%m')},\n Published on "
            f"{published_month}-13",
            size=10,
        )
        g.axes.set_xlabel("longitude")
        g.axes.set_ylabel("latitude")
        g.axes.spines["right"].set_visible(False)
        g.axes.spines["top"].set_visible(False)
        if gdf_adm is not None:
            gdf_adm.boundary.plot(ax=g.axes, color="#888888", linewidth=2)

        plt_path = _get_output_path_map(
            iso3=iso3,
            version=VERSION,
            target_year=target_date.year,
            target_month=target_date.month,
            leadtime=lt,
        )
        plt.savefig(plt_path, facecolor="white", bbox_inches="tight")


def compute_trigger(
    iso3: str,
    target_date: date,
    prob: float,
    precip_cap: int,
    download: bool,
    interpolate_raster: bool,
    leadtimes: List[int] = None,
    pcodes: List[str] = None,
    adm_level: int = 1,
    stats_col: str = "mean_ADM1_PCODE",
    pcode_col: str = "ADM1_PCODE",
    adm_name_col: str = "ADM1_EN",
    date_col: str = "date",
    leadtime_col: str = "leadtime",
    round_precip_int: bool = True,
):
    """
    Compute the trigger metric and a binary true/false if trigger is met
    The logic assumes that the trigger is defined as being met if there is
    `prob` probability of <= precip_cap monthly preciptiation (mm)
    :param iso3: country iso3 code
    :param target_date:  date the forecasts should predict (year-month)
    :param prob: minimum probability of the forecast for the trigger
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
        use_unrounded_area_coords=USE_UNROUNDED_AREA_COORDS,
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
    ).quantile(prob)
    if round_precip_int:
        # round float to closest integer
        df_stats_quant[stats_col] = round(df_stats_quant[stats_col], 0)
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

    output_path = get_output_path_metrics(
        iso3, VERSION, target_date.year, target_date.month
    )
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    df_stats_quant.to_csv(output_path, index=False)
    logger.info(f"The trigger output has been saved to {output_path}")

    # question: should this function be called within
    # compute_trigger() or from main()?
    create_map(
        iso3=iso3,
        target_date=target_date,
        leadtimes=leadtimes,
        prob=prob,
        gdf_adm=gdf_adm,
        slice_lon=SLICE_LON,
        slice_lat=SLICE_LAT,
        bins=BINS,
    )

    return df_stats_quant


def main():
    compute_trigger(
        COUNTRY_ISO3,
        target_date=date(year=2022, month=1, day=1),
        prob=0.5,
        precip_cap=210,
        download=True,
        interpolate_raster=False,
    )


if __name__ == "__main__":
    main()
