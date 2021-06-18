"""
Download raster data from ECMWF and extracts time series of rainfall in selected locations
"""
import logging
from typing import Dict

import xarray as xr

from src.indicators.flooding.cds.area import Area, Station
from src.indicators.flooding.cds import cds


DATA_DIRECTORY = "ecmwf"
CDS_VARIABLE_NAME = "total_precipitation"
XARRAY_VARIABLE_NAME = "tp"

logger = logging.getLogger(__name__)


class EcmwfEra5(cds.Cds):
    def __init__(self):
        super().__init__(
            data_directory=DATA_DIRECTORY,
            cds_variable_name=CDS_VARIABLE_NAME,
            xarray_variable_name=XARRAY_VARIABLE_NAME,
            year_min=1979,
            year_max=2020,
            cds_name="reanalysis-era5-single-levels",
            dataset=["reanalysis", "ensemble_members"],
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        year_min: int = None,
        year_max: int = None,
    ):
        year_min = self.year_min if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(f"Downloading ECMWF rainfall for years {year_min} - {year_max}")
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            super()._download(
                country_iso3=country_iso3, area=area, year=year, toggle_hours=True
            )

    def process(
        self,
        country_iso3: str,
        stations: Dict[str, Station],
    ):
        # Get list of files to open
        logger.info(f"Processing ECMWF rainfall")
        filepath_list = [
            self._get_raw_filepath(
                country_iso3=country_iso3,
                year=year,
            )
            for year in range(self.year_min, self.year_max + 1)
        ]
        # Read in the dataset
        # TODO: Read in the ensemble (numberOfPoints 136)
        # TODO: numberOfPoints is unique to each Area and won't work for a new country
        logger.info(f"Reading in {len(filepath_list)} files")
        with xr.open_mfdataset(
            filepath_list,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"numberOfPoints": 136}},
        ) as ds:
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = self.get_station_dataset(
                stations=stations, ds=ds, coord_names=["time"]
            )
        # Write out the new dataset to a file
        self._write_to_processed_file(
            country_iso3=country_iso3,
            ds=ds_new,
        )
