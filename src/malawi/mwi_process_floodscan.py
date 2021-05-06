import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import xarray as xr
import rioxarray

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.config import Config

COUNTRY = "malawi"
ADM = 1
config = Config()
parameters = config.parameters(COUNTRY)

COUNTRY_DIR = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, COUNTRY)
COUNTRY_DATA_RAW_DIR = os.path.join(config.DATA_DIR, config.RAW_DIR, COUNTRY)

# ADM_BOUND_PATH = os.path.join(
#    COUNTRY_DATA_RAW_DIR, config.SHAPEFILE_DIR, parameters[f"path_admin{str(ADM)}_shp"]
# )
ADM_BOUND_PATH = os.path.join(
    config.DATA_DIR, "processed/malawi/Shapefiles/mwi_adm2_flooding_sel.shp"
)

FLOODSCAN_DIR = os.path.join(
    config.DATA_PRIVATE_DIR, "raw", "floodscan-africa-1998-2020"
)
FLOODSCAN_PATH = os.path.join(
    FLOODSCAN_DIR, "aer_sfed_area_300s_19980112_20201231_v05r01.nc"
)
COUNTRY_FLOODSCAN_DIR = os.path.join(
    config.DATA_PRIVATE_DIR, "processed", COUNTRY, "floodscan"
)
# country_floodscan_path = os.path.join(country_floodscan_dir,f"{country}_floodscan_1998_2020.nc")


def main():
    # #Only needed once to clip data to country, takes some time
    # also actually the stats can be computed without clipping
    # ds=xr.load_dataset(floodscan_path)
    # ds_clip = ds.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    # ds_clip.to_netcdf(country_floodscan_path)
    # load the data
    ds = xr.open_dataset(FLOODSCAN_PATH)
    ds_sel = ds.sel(time="2020-01")
    # compute statistics at adm1 level
    # this takes some time
    df_month_total_adm1 = alldates_statistics_total(
        ds,
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        .rio.write_crs("EPSG:4326")
        .rio.transform(),
        ADM_BOUND_PATH,
    )
    # save to file
    df_month_total_adm1.drop("geometry", axis=1).to_csv(
        os.path.join(
            COUNTRY_FLOODSCAN_DIR, f"{COUNTRY}_floodscan_statistics_admin2.csv"
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
        df[[f"percentile_{str(p)}" for p in percentile_list]] = pd.DataFrame(
            zonal_stats(
                vectors=df,
                raster=ds_date[data_var].values,
                affine=raster_transform,
                nodata=np.nan,
                stats=" ".join([f"percentile_{str(p)}" for p in percentile_list]),
            )
        )[[f"percentile_{str(p)}" for p in percentile_list]]

        df["date"] = pd.to_datetime(date)

        df_list.append(df)
    df_hist = pd.concat(df_list)
    df_hist = df_hist.sort_values(by="date")

    return df_hist


if __name__ == "__main__":
    main()
