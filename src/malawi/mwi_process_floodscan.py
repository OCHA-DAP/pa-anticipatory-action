import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.enums import Resampling
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl
import seaborn as sns
import glob

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.config import Config
from src.utils_general.utils import download_ftp, download_url
from src.utils_general.raster_manipulation import (
    fix_calendar,
    invert_latlon,
    change_longitude_range,
)
from src.utils_general.plotting import plot_raster_boundaries_clip

DATA_PRIVATE_DIR = os.path.join(os.environ["AA_DATA_PRIVATE_DIR"])
country = "malawi"
config = Config()
parameters = config.parameters(country)


country_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)
country_data_raw_dir = os.path.join(config.DATA_DIR, config.RAW_DIR, country)

adm1_bound_path = os.path.join(
    country_data_raw_dir, config.SHAPEFILE_DIR, parameters["path_admin1_shp"]
)

df_bound = gpd.read_file(adm1_bound_path)

floodscan_dir = os.path.join(DATA_PRIVATE_DIR, "floodscan-africa-1998-2020")
floodscan_path = os.path.join(
    floodscan_dir, "aer_sfed_area_300s_19980112-20201231_v05r01.nc"
)
country_floodscan_dir = os.path.join(
    DATA_PRIVATE_DIR, "processed", country, "floodscan"
)
country_floodscan_path = os.path.join(
    country_floodscan_dir, f"{country}_floodscan_1998_2020.nc"
)


def main():
    # #Only needed once to clip data to country, takes some time
    # also actually the stats can be computed without clipping
    # ds=xr.load_dataset(floodscan_path)
    # ds_clip = ds.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    # ds_clip.to_netcdf(country_floodscan_path)
    # load the data
    ds = xr.load_dataset(country_floodscan_path)
    # compute statistics at adm1 level
    # this takes some time
    df_month_total_adm1 = alldates_statistics_total(
        ds,
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        .rio.write_crs("EPSG:4326")
        .rio.transform(),
        adm1_bound_path,
    )
    # save to file
    df_month_total_adm1.drop("geometry", axis=1).to_csv(
        os.path.join(
            country_floodscan_dir, f"{country}_floodscan_statistics_admin1.csv"
        ),
        index=False,
    )


def alldates_statistics_total(ds, raster_transform, adm_path, data_var="SFED_AREA"):
    # compute statistics on level in adm_path for all dates in ds
    df_list = []
    for date in ds.time.values:
        df = gpd.read_file(adm_path)
        ds_date = ds.sel(time=date)

        df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
            zonal_stats(
                vectors=df,
                raster=ds_date[data_var].values,
                affine=raster_transform,
                nodata=np.nan,
            )
        )[["mean", "max", "min"]]

        percentile_list = [10, 20, 30, 40, 50, 60, 70, 80]
        zonal_stats_percentile_dict = zonal_stats(
            vectors=df,
            raster=ds_date[data_var].values,
            affine=raster_transform,
            nodata=np.nan,
            stats=" ".join([f"percentile_{str(p)}" for p in percentile_list]),
        )[0]
        for p in percentile_list:
            df[[f"percentile_{str(p)}" for p in percentile_list]] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date[data_var].values,
                    affine=raster_transform,
                    nodata=np.nan,
                    stats=" ".join([f"percentile_{str(p)}" for p in percentile_list]),
                )
            )[[f"percentile_{str(p)}" for p in percentile_list]]

        df["date"] = pd.to_datetime(date)  # .strftime("%Y-%m-%d"))

        df_list.append(df)
    df_hist = pd.concat(df_list)
    df_hist = df_hist.sort_values(by="date")

    #     df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    #     df_hist['date_month']=df_hist.date.dt.to_period("M")

    return df_hist


if __name__ == "__main__":
    # main()
    print(parameters)
