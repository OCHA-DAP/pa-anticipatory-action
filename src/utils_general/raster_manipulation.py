import logging
import geopandas as gpd
import pandas as pd

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
    boundary_path,
    bound_col,
    raster_array,
    lon_coord="lon",
    lat_coord="lat",
    all_touched=False,
    geom_col="geometry",
    # raster_transform,
    # threshold,
    band=1,
    nodata=-9999,
    upscale_factor=None,
):
    """
    Compute statistics of the raster_array per geographical region
    defined in the boundary_path file
    Currently several methods are implemented, namely the maximum
    and mean per region, and the percentage of the area with a value
    larger than threshold.
    For all three methods, two variations are implemented:
    one where all raster cells touching a region are counted,
    and one where only the raster cells that have their center within
    the region are counted.
    Args:
        boundary_path (str): path to the shapefile
        raster_array (numpy array): array containing the raster data
        raster_transform (numpy array): array containing the
        transformation of the raster data, this is related to the CRS
        threshold (float): minimum probability of a raster cell to count
        that cell as meeting the criterium
        upscale_factor: currently not implemented

    Returns:
        df (Geodataframe): dataframe containing the computed statistics
    """
    df_list = []
    gdf = gpd.read_file(boundary_path)

    for a in gdf[bound_col].unique():
        gdf_adm = gdf[gdf[bound_col] == a]

        da_clip = raster_array.rio.set_spatial_dims(
            x_dim=lon_coord, y_dim=lat_coord
        ).rio.clip(gdf_adm[geom_col], all_touched=all_touched)
        # how to best do different stats?
        grid_mean = da_clip.mean(
            dim=[lon_coord, lat_coord], skipna=True
        ).rename("mean_adm")
        # call(da_clip, "mean")()
        # grid_max = raster_clip.max(dim=[lon_coord, lat_coord]).rename(
        #     {var_name: "max_cell"}
        # )
        # grid_quant90 = raster_clip.quantile(
        #     0.9, dim=[lon_coord, lat_coord]
        # ).rename({var_name: "10quant_cell"})
        # grid_percth40 = (
        #     raster_clip.where(raster_clip.prob >= 40).count(
        #         dim=[lon_coord, lat_coord]
        #     )
        #     / raster_clip.count(dim=[lon_coord, lat_coord])
        #     * 100
        # )
        # zonal_stats_xr = xr.merge(
        #     [
        #         grid_mean,
        #         grid_min,
        #         grid_max,
        #         grid_std,
        #         grid_quant90,
        #         grid_percth40,
        #         grid_dom,
        #     ]
        # )

        df_adm = grid_mean.to_dataframe().reset_index()
        df_adm[bound_col] = a
        df_list.append(df_adm)

    df_zonal_stats = pd.concat(df_list)
    return df_zonal_stats
