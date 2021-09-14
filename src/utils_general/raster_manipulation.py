import logging
from typing import List

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

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
    bound_col: str,
    raster_array: xr.DataArray,
    lon_coord: str = "x",
    lat_coord: str = "y",
    stats_list: List[str] = None,
    percentile_list: List[int] = None,
    all_touched: bool = False,
    geom_col: str = "geometry",
):
    """
    Compute statistics of the raster_array per geographical region
    defined in the gdf
    the area covered by the gdf should be a subset of that
    covered by raster_array
    :param gdf: geodataframe containing a row per area for which
    the stats are computed
    :param bound_col: name of the column containing the region names
    :param raster_array: DataArray containing the raster data.
    Needs to have a CRS.
    Should not be a DataSet but DataArray
    :param lon_coord: name of longitude dimension in raster_array
    :param lat_coord: name of latitude dimension in raster_array
    :param stats_list: list with function names indicating
    which stats to compute
    :param percentile_list: list with integers ranging from 0 to 100
    indicating which percentiles to compute
    :param all_touched: if False, only cells with their centre within the
    region will be included when computing the stat.
    If True all cells touching the region will be included.
    :param geom_col: name of the column in boundary_path
    containing the polygon geometry
    :return: dataframe containing the computed statistics
    """
    df_list = []

    if stats_list is None:
        stats_list = ["mean", "std", "min", "max", "sum", "count"]

    for bound_id in gdf[bound_col].unique():
        gdf_adm = gdf[gdf[bound_col] == bound_id]

        da_clip = raster_array.rio.set_spatial_dims(
            x_dim=lon_coord, y_dim=lat_coord
        ).rio.clip(gdf_adm[geom_col], all_touched=all_touched)

        grid_stat_all = []
        for stat in stats_list:
            # count automatically ignores NaNs
            # therefore skipna can also not be given as an argument
            # implemented count cause needed for computing percentages
            kwargs = {}
            if stat != "count":
                kwargs["skipna"] = True
            # makes sum return NaN instead of 0 if array
            # only contains NaNs
            if stat == "sum":
                kwargs["min_count"] = 1
            grid_stat = getattr(da_clip, stat)(
                dim=[lon_coord, lat_coord], **kwargs
            ).rename(f"{stat}_{bound_col}")
            grid_stat_all.append(grid_stat)

        if percentile_list is not None:
            grid_quant = [
                da_clip.quantile(quant / 100, dim=[lon_coord, lat_coord])
                .drop("quantile")
                .rename(f"{quant}quant_{bound_col}")
                for quant in percentile_list
            ]
            grid_stat_all.extend(grid_quant)

        # if dims is 0, it throws an error when merging
        # and then converting to a df
        # this occurs when the input da is 2D
        if not grid_stat_all[0].dims:
            df_adm = pd.DataFrame(
                {da_stat.name: [da_stat.values] for da_stat in grid_stat_all}
            )
        else:
            zonal_stats_xr = xr.merge(grid_stat_all)
            df_adm = (
                zonal_stats_xr.to_dataframe()
                .drop("spatial_ref", axis=1)
                .reset_index()
            )
        df_adm[bound_col] = bound_id
        df_list.append(df_adm)

    df_zonal_stats = pd.concat(df_list).reset_index(drop=True)
    return df_zonal_stats
