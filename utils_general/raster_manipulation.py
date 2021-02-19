import requests
import os
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import pandas as pd

logger = logging.getLogger(__name__)

def invert_latlon(ds):
    """
    This function checks for inversion of latitude and longitude and changes them if needed
    Some datasets come with flipped coordinates.
    For latitude this means that the coordinates start with the smallest number while for longitude they are flipped when starting with the largest number.
    Some functions, such as zonal_stats, produce wrong results when these coordinates are flipped.
    The dataset given as input (ds) should have its lat coordinates named "lat" and lon coordinates "lon"
    Function largely copied from https://github.com/perrygeo/python-rasterstats/issues/218
    Args:
        ds (xarray dataset): dataset containing the variables and coordinates

    Returns:
        da (xarray dataset): dataset containing the variables and flipped coordinates
    """
    lat_start=ds.lat[0].item()
    lat_end=ds.lat[ds.dims["lat"]-1].item()
    lon_start=ds.lon[0].item()
    lon_end=ds.lon[ds.dims["lon"]-1].item()
    if lat_start<lat_end:
        lat_status='north down'
    else:
        lat_status='north up'
    if lon_start<lon_end:
        lon_status='lon normal'
    else:
        lon_status='lon inverted'

    # Flip the raster as necessary (based on the flags)
    if lat_status=='north down':
        logger.info("Dataset was north down, latitude coordinates have been flipped")
        ds=ds.reindex(lat=ds["lat"][::-1])
    #TODO: implement longitude inversion
    if lon_status=='inverted':
        logger.error("Inverted longitude still needs to be implemented..")
        # ds=np.flip(ds.squeeze('F'),axis=1)
    return ds

#TODO: understand when to change longitude range for rasterstats to work and when not!!! For IRI seasonal forecast, outputs are wrong when not changing to -180 180, while for IRI CAMS observational terciles outputs are wrong WHEN changing to -180 180 instead of 0 360...
def change_longitude_range(ds):
    """
    If longitude ranges from 0 to 360, change it to range from -180 to 180.
    Assumes the name of the longitude coordinates is "lon"
    Args:
        ds (xarray dataset): dataset that should be transformed

    Returns:
        ds_lon (xarray dataset): dataset with transformed longitude coordinates
    """
    ds_lon = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    return ds_lon

def fix_calendar(ds, timevar='F'):
    """
    Some datasets come with a wrong calendar attribute that isn't recognized by xarray
    So map this attribute that can be read by xarray
    Args:
        ds (xarray dataset): dataset of interest
        timevar (str): variable that contains the time in ds

    Returns:
        ds (xarray dataset): modified dataset
    """
    if "calendar" in ds[timevar].attrs.keys():
        if ds[timevar].attrs['calendar'] == '360':
            ds[timevar].attrs['calendar'] = '360_day'
    elif "units" in ds[timevar].attrs.keys():
        if "months since" in ds[timevar].attrs['units']:
            ds[timevar].attrs['calendar'] = '360_day'
    return ds

def compute_raster_statistics(boundary_path, raster_array, raster_transform, threshold, band=1, nodata=-9999, upscale_factor=None):
    """
    Compute statistics of the raster_array per geographical region defined in the boundary_path file
    Currently several methods are implemented, namely the maximum and mean per region, and the percentage of the area with a value larger than threshold.
    For all three methods, two variations are implemented: one where all raster cells touching a region are counted, and one where only the raster cells that have their center within the region are counted.
    Args:
        boundary_path (str): path to the shapefile
        raster_array (numpy array): array containing the raster data
        raster_transform (numpy array): array containing the transformation of the raster data, this is related to the CRS
        threshold (float): minimum probability of a raster cell to count that cell as meeting the criterium
        upscale_factor: currently not implemented

    Returns:
        df (Geodataframe): dataframe containing the computed statistics
    """
    df = gpd.read_file(boundary_path)
    #TODO: decide if we want to upsample and if yes, implement
    # if upscale_factor:
    #     forecast_array, transform = resample_raster(raster_path, upscale_factor)
    # else:

    # extract statistics for each polygon. all_touched=True includes all cells that touch a polygon, with all_touched=False only those with the center inside the polygon are counted.
    df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=band, nodata=nodata))["max"]
    df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=band, nodata=nodata))["max"]

    df["avg_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=band, nodata=nodata))[
        "mean"]
    df["avg_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=band, nodata=nodata))[
        "mean"]

    # calculate the percentage of the area within an geographical area that has a value larger than threshold
    forecast_binary = np.where(raster_array >= threshold, 1, 0)
    bin_zonal = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'], band=band, nodata=nodata))
    df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
    bin_zonal_touched = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, all_touched=True, stats=['count', 'sum'], band=band, nodata=nodata))
    df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100

    return df