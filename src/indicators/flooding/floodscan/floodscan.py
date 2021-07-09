import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import xarray as xr

# not using rioxarray directly but using .rio
# so let flake8 ignore it
import rioxarray  # noqa
import logging


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config

config = Config()

DATA_DIR = Path(config.DATA_DIR)
PRIVATE_DATA_DIR = config.PRIVATE_DIR
PUBLIC_DATA_DIR = config.PUBLIC_DIR
RAW_DATA_DIR = config.RAW_DIR
GLOBAL_DIR = "glb"
SHAPEFILE_DIR = config.SHAPEFILE_DIR
PROCESSED_DATA_DIR = config.PROCESSED_DIR
FLOODSCAN_DIR = Path("floodscan")
FLOODSCAN_FILENAME = (
    "floodscan_africa_sfed_area_300s_19980112_20201231_v05r01.nc"
)

DEFAULT_ADMIN_LEVEL = 1

logger = logging.getLogger(__name__)


class Floodscan:
    """Create an instance of a Floodscan object, from which you can process the
    raw data and read the data."""

    def read_raw_dataset(self):
        filepath = self._get_raw_filepath()
        # would be better to do with load_dataset, but since dataset is
        # huge this takes up too much memory..
        with xr.open_dataset(filepath) as ds:
            return ds
        # return xr.open_dataset(filepath)

    def process(
        self,
        country_name: str,
        adm_level: int = DEFAULT_ADMIN_LEVEL,
    ):
        """
        Load data, call function to compute statistics per admin, and
        save the results to a csv Args: country_name: name of the
        country of interest adm_level: admin level to compute the
        statistics on
        """
        config = Config()
        parameters = config.parameters(country_name)
        country_iso3 = parameters["iso3_code"]
        adm_boundaries_path = os.path.join(
            DATA_DIR,
            PUBLIC_DATA_DIR,
            RAW_DATA_DIR,
            country_iso3,
            config.SHAPEFILE_DIR,
            parameters[f"path_admin{adm_level}_shp"],
        )
        ds = self.read_raw_dataset()
        # get the affine transformation of the dataset. looks
        # complicated, but haven't found better way to do it
        coords_transform = (
            ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326")
            .rio.transform()
        )
        # this takes a few hours to compute
        df = self.compute_stats_per_area(
            ds,
            coords_transform,
            adm_boundaries_path,
            parameters[f"shp_adm{adm_level}c"],
        )
        self._write_to_processed_file(country_iso3, adm_level, df)

    def compute_stats_per_area(
        self,
        ds,
        raster_transform,
        adm_path,
        adm_col,
        data_var="SFED_AREA",
        percentile_list=[2, 4, 6, 8, 10, 20],
    ):
        """
        Compute statistics on the raster cells per admin area Args: ds:
        the xarray dataset with values per raster cell raster_transform:
        the affine transformation of ds adm_path: the path to the admin
        boundaries shp file adm_col: the name of the column containing
        the admin name data_var: the variable of interest in ds
        percentile_list: list of thresholds to compute the value x% of
        the cells is below at

        Returns: df_hist: dataframe with the statistics per admin
        """
        # compute statistics on level in adm_path for all dates in ds
        df_list = []
        for date in ds.time.values:
            df = gpd.read_file(adm_path)[[adm_col, "geometry"]]
            ds_date = ds.sel(time=date)

            df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date[data_var].values,
                    affine=raster_transform,
                    nodata=np.nan,
                )
            )[["mean", "max", "min"]]
            # TODO: the percentiles seem to always return 0, even if
            # setting the p to 0.00001. Don't understand why yet..
            df[
                [f"percentile_{str(p)}" for p in percentile_list]
            ] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date[data_var].values,
                    affine=raster_transform,
                    nodata=np.nan,
                    stats=" ".join(
                        [f"percentile_{str(p)}" for p in percentile_list]
                    ),
                )
            )[
                [f"percentile_{str(p)}" for p in percentile_list]
            ]

            df["date"] = pd.to_datetime(date)

            df_list.append(df)
        df_hist = pd.concat(df_list)
        df_hist = df_hist.sort_values(by="date")
        # drop the geometry column, else csv becomes huge
        df_hist = df_hist.drop("geometry", axis=1)

        return df_hist

    def read_processed_dataset(
        self,
        country_iso3: str,
        adm_level: int,
    ):
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3,
            adm_level=adm_level,
        )
        return pd.read_csv(filepath, index_col=False)

    def _write_to_processed_file(
        self,
        country_iso3: str,
        adm_level: int,
        df: pd.DataFrame,
    ) -> Path:
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3,
            adm_level=adm_level,
        )
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        filepath.unlink(missing_ok=True)
        logger.info(f"Writing to {filepath}")
        df.to_csv(filepath)
        return filepath

    def _get_raw_filepath(
        self,
    ):
        directory = (
            DATA_DIR
            / PRIVATE_DATA_DIR
            / RAW_DATA_DIR
            / GLOBAL_DIR
            / FLOODSCAN_DIR
        )

        return directory / Path(FLOODSCAN_FILENAME)

    def _get_processed_filepath(
        self,
        country_iso3: str,
        adm_level: int,
    ) -> Path:
        filename = f"{country_iso3.lower()}_floodscan_stats_adm{adm_level}.csv"
        return (
            DATA_DIR
            / PRIVATE_DATA_DIR
            / PROCESSED_DATA_DIR
            / country_iso3
            / FLOODSCAN_DIR
            / filename
        )
