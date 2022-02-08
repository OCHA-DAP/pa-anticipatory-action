"""
Process Floodscan data.

From their user guide:
FloodScan’s primary products are
best at depicting large scale,
inland river flooding when landscapes are unfrozen.
Flooding in smaller floodplains and within 5–10 km of coastlines
is usually not depicted unless it is part of a
larger flood event.

They have two products: the SFED and MFED.
MFED is more sensitive than MFED but thereby also has a higher chance of
false positives.
We currently only have access to SFED which
seems most applicable for our purposes.
Standard flood extent depiction (SFED):
SFED is designed to prioritize low false positive
rates for large scale flooding with algorithmic consistency
over long time scales and large
regions. SFED processing includes dynamic 2- to 3-day weighted
averaging along with
other spatiotemporal methods that minimize false
positives and noise. As a result, the
SFED algorithm mode makes relatively conservative
estimates of maximum flood extent,
flood frequency, and flood duration.

We also have a variation of the SFD,
namely the no detection threshold SFED (NDT-SFED).
Like SFED but produced without flooded
fraction thresholding.
The algorithm downscales flooded fraction to make its
flood depiction products (e.g., 90-m scale).
However, the algorithm applies a minimum
detectable flooded fraction (MDFF) threshold prior to
downscaling to create the SFED
products. In the NDT-SFED this threshold is not applied,
and thus it is more sensitive
than the NDT-SFED but with the drawback of more false positives.

Lastly there is the LWMASK_AREA which is used to mask areas
FloodScan considers to be persistent
open water. As far as I understand,
this mask is already applied when computing the SFED values
and thus doesn't have to be substracted anymore from the SFED data.

The original data is binary. This data is at 3 arcseconds.
We have the data at 300 arcseconds.
It is unclear how the downsampling is done, but it seems as
it is the fraction of the cells
 that are flooded (i.e. are True in the original data). It is thus float data.
"""

import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterstats import zonal_stats

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.config import Config

config = Config()

DATA_DIR = Path(config.DATA_DIR)
PRIVATE_DATA_DIR = config.DATA_PRIVATE_DIR
PUBLIC_DATA_DIR = config.DATA_DIR
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
        custom_path: str = None,
        custom_id_col: str = "",
        custom_name: str = "",
        start_date: str = "1998-01-12",
        end_date: str = "2020-12-31",
    ):
        """
        Load data, call function to compute statistics per admin,
        and save the results to a csv
        Args:
            country_name: name of the country of interest
            adm_level: admin level to compute the statistics on
            custom_path: file path to the custom area shapefile
            custom_id_col: the name of the id column for features
            in the custom area shapefile
            custom_name: the name of the custom area (for the output file)
            start_date: to filter by start date
            end_date: to filter by end date
        """
        config = Config()
        parameters = config.parameters(country_name)
        country_iso3 = parameters["iso3_code"]

        ds = self.read_raw_dataset().sel(time=slice(start_date, end_date))

        # get the affine transformation of the dataset.
        # looks complicated, but haven't found better way to do it
        coords_transform = (
            ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326")
            .rio.transform()
        )

        if custom_path:
            boundaries_path = custom_path

            # this takes a few hours to compute
            df = self.compute_stats_per_area(
                ds,
                coords_transform,
                boundaries_path,
                custom_id_col,
            )
            self._write_to_processed_file(
                country_iso3, adm_level, df, custom_name
            )

        else:
            boundaries_path = os.path.join(
                DATA_DIR,
                RAW_DATA_DIR,
                country_name,
                config.SHAPEFILE_DIR,
                parameters[f"path_admin{adm_level}_shp"],
            )

            # this takes a few hours to compute
            df = self.compute_stats_per_area(
                ds,
                coords_transform,
                boundaries_path,
                parameters[f"shp_adm{adm_level}c"],
            )
            self._write_to_processed_file(
                country_iso3, adm_level, df, custom_name
            )

    def compute_stats_per_area(
        self,
        ds,
        raster_transform,
        bound_path,
        id_col,
        data_var="SFED_AREA",
        percentile_list=None,
    ):
        """
        Compute statistics on the raster cells per admin area
        Args:
            ds: the xarray dataset with values per raster cell
            raster_transform: the affine transformation of ds
            bound_path: the path to the boundaries shp file
            id_col: the name of the column containing the
            boundary IDs
            data_var: the variable of interest in ds
            percentile_list: list of thresholds to compute
            the value x% of the cells is below at

        Returns:
            df_hist: dataframe with the statistics per admin
        """
        if not percentile_list:
            percentile_list = [2, 4, 6, 8, 10, 20]

        # compute statistics on level in adm_path for all dates in ds
        df_list = []
        for date in ds.time.values:
            df = gpd.read_file(bound_path)[[id_col, "geometry"]]
            ds_date = ds.sel(time=date)

            df[["mean_cell", "max_cell", "min_cell"]] = pd.DataFrame(
                zonal_stats(
                    vectors=df,
                    raster=ds_date[data_var].values,
                    affine=raster_transform,
                    nodata=np.nan,
                )
            )[["mean", "max", "min"]]
            # TODO: the percentiles seem to always return 0,
            #  even if setting the p to 0.00001.
            #  Don't understand why yet..
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
        custom_name: str = None,
    ) -> Path:
        if custom_name:
            filepath = self._get_processed_filepath(
                country_iso3=country_iso3,
                adm_level=adm_level,
                custom_name=custom_name,
            )
        else:
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
        self, country_iso3: str, adm_level: int, custom_name: str = None
    ) -> Path:
        if custom_name:
            filename = (
                f"{country_iso3.lower()}_floodscan_stats_{custom_name}.csv"
            )
        else:
            filename = (
                f"{country_iso3.lower()}_floodscan_stats_adm{adm_level}.csv"
            )
        return (
            DATA_DIR
            / PRIVATE_DATA_DIR
            / PROCESSED_DATA_DIR
            / country_iso3
            / FLOODSCAN_DIR
            / filename
        )
