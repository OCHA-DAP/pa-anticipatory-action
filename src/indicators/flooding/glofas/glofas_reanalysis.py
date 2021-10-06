import logging
from typing import Dict

import xarray as xr

from utils_general.area import Area, Station

from .glofas import DEFAULT_VERSION, Glofas, _get_station_dataset

logger = logging.getLogger(__name__)


class GlofasReanalysis(Glofas):
    def __init__(self, **kwargs):
        super().__init__(
            year_min=1979,
            year_max=2020,
            cds_name="cems-glofas-historical",
            dataset=["consolidated_reanalysis"],
            dataset_variable_name="dataset",
            system_version_minor={2: 1, 3: 1},
            date_variable_prefix="h",
            **kwargs,
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        version: int = DEFAULT_VERSION,
        year_min: int = None,
        year_max: int = None,
    ):
        year_min = self.year_min if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Downloading GloFAS reanalysis v{version} for years {year_min} -"
            f" {year_max}"
        )
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            super()._download(
                country_iso3=country_iso3,
                area=area,
                year=year,
                version=version,
            )

    def process(
        self,
        country_iso3: str,
        stations: Dict[str, Station],
        version: int = DEFAULT_VERSION,
    ):
        # Get list of files to open
        logger.info(f"Processing GloFAS Reanalysis v{version}")
        filepath_list = [
            self._get_raw_filepath(
                country_iso3=country_iso3,
                version=version,
                year=year,
            )
            for year in range(self.year_min, self.year_max + 1)
        ]
        # Read in the dataset
        logger.info(f"Reading in {len(filepath_list)} files")

        with xr.open_mfdataset(
            filepath_list, engine="cfgrib", backend_kwargs={"indexpath": ""}
        ) as ds:
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = _get_station_dataset(
                stations=stations, ds=ds, coord_names=["time"]
            )
        # Write out the new dataset to a file
        return self._write_to_processed_file(
            country_iso3=country_iso3,
            version=version,
            ds=ds_new,
        )
