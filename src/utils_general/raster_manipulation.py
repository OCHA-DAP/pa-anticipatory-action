import logging
import geopandas as gpd
import pandas as pd
import xarray as xr
from typing import List
import numpy as np
from rasterstats import zonal_stats
import rioxarray  # noqa: F401

logger = logging.getLogger(__name__)


def invert_latlon(ds):
    """
    This function checks for inversion of latitude and longitude
    and changes them if needed
    Some datasets come with flipped coordinates.
    For latitude this means that the coordinates start with the
    smallest number while for longitude they are flipped when starting
    with the largest number.
    Some functions, such as zonal_stats, produce wrong results when
    these coordinates are flipped.
    The dataset given as input (ds) should have its lat coordinates
    named "lat" and lon coordinates "lon"
    Function largely copied from
    https://github.com/perrygeo/python-rasterstats/issues/218
    Args:
        ds (xarray dataset): dataset containing the variables
        and coordinates

    Returns:
        da (xarray dataset): dataset containing the variables
        and flipped coordinates
    """
    lat_start = ds.lat[0].item()
    lat_end = ds.lat[ds.dims["lat"] - 1].item()
    lon_start = ds.lon[0].item()
    lon_end = ds.lon[ds.dims["lon"] - 1].item()
    if lat_start < lat_end:
        lat_status = "north down"
    else:
        lat_status = "north up"
    if lon_start < lon_end:
        lon_status = "lon normal"
    else:
        lon_status = "lon inverted"

    # Flip the raster as necessary (based on the flags)
    if lat_status == "north down":
        logger.info(
            "Dataset was north down, latitude coordinates have been flipped"
        )
        ds = ds.reindex(lat=ds["lat"][::-1])
    # TODO: implement longitude inversion
    if lon_status == "inverted":
        logger.error("Inverted longitude still needs to be implemented..")
        # ds=np.flip(ds.squeeze('F'),axis=1)
    return ds


# TODO: understand when to change longitude range for rasterstats
# to work and when not!!!
# For IRI seasonal forecast, outputs are wrong when not changing to
# -180 180,
# while for IRI CAMS observational terciles outputs are wrong WHEN
# changing to -180 180 instead of 0 360...
def change_longitude_range(ds):
    """
    If longitude ranges from 0 to 360,
    change it to range from -180 to 180.
    Assumes the name of the longitude coordinates is "lon"
    Args:
        ds (xarray dataset): dataset that should be transformed

    Returns:
        ds_lon (xarray dataset): dataset with transformed longitude
        coordinates
    """
    ds_lon = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds_lon


def fix_calendar(ds, timevar="F"):
    """
    Some datasets come with a wrong calendar attribute that isn't
    recognized by xarray
    So map this attribute that can be read by xarray
    Args:
        ds (xarray dataset): dataset of interest
        timevar (str): variable that contains the time in ds

    Returns:
        ds (xarray dataset): modified dataset
    """
    if "calendar" in ds[timevar].attrs.keys():
        if ds[timevar].attrs["calendar"] == "360":
            ds[timevar].attrs["calendar"] = "360_day"
    elif "units" in ds[timevar].attrs.keys():
        if "months since" in ds[timevar].attrs["units"]:
            ds[timevar].attrs["calendar"] = "360_day"
    return ds


def compute_raster_statistics(
    gdf: gpd.GeoDataFrame,
    raster_array: xr.DataArray,
    stats_list: List[str] = None,
    percentile_list: List[float] = None,
    all_touched: bool = False,
):
    """
    Compute statistics of the raster_array per geographical region
    defined in the boundary_path file. The raster_array has to be 2D
    :param gdf: geodataframe containing a row per area for which
    the stats are computed
    :param raster_array: DataArray containing the raster data.
    Needs to have a CRS.
    Should not be a DataSet but DataArray and should be 2D
    :param stats_list: list with function names indicating
    which stats to compute
    :param percentile_list: list with floats ranging from 0 to 100
    indicating which percentiles to compute
    :param all_touched: if False, only cells with their centre within the
    region will be included when computing the stat.
    If True all cells touching the region will be included.
    :return: dataframe containing the computed statistics
    """

    if stats_list is None:
        stats_list = ["mean", "std", "min", "max", "sum", "count"]

    if percentile_list is not None:
        percentile_list_str = " ".join(
            [f"percentile_{str(p)}" for p in percentile_list]
        )
        stats_list.append(percentile_list_str)

    df_zonal_stats = pd.DataFrame(
        zonal_stats(
            vectors=gdf,
            raster=raster_array.values,
            affine=raster_array.rio.transform(),
            stats=stats_list,
            nodata=np.nan,
            all_touched=all_touched,
        )
    )

    return df_zonal_stats
