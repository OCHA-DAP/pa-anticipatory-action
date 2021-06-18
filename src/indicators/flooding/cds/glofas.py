"""
Download raster data from GLOFAS and extracts time series of water discharge in selected locations
"""
import logging
from typing import Dict, List

import xarray as xr

from src.indicators.flooding.cds.area import Area, Station
from src.indicators.flooding.cds import cds

DATA_DIRECTORY = "glofas"
CDS_VARIABLE_NAME = "river_discharge_in_the_last_24_hours"
XARRAY_VARIABLE_NAME = "dis24"

DEFAULT_VERSION = 3
HYDROLOGICAL_MODELS = {2: "htessel_lisflood", 3: "lisflood"}

logger = logging.getLogger(__name__)


class Glofas(cds.Cds):
    def __init__(self, *args, **kwargs):
        """
        :param system_version_minor: The minor version of the GloFAS model. Depends on the major version,
         so is given as a dictionary with the format {major_version: minor_version}
        """
        self.system_version_minor = kwargs.pop("system_version_minor")
        super().__init__(
            data_directory=DATA_DIRECTORY,
            cds_variable_name=CDS_VARIABLE_NAME,
            xarray_variable_name=XARRAY_VARIABLE_NAME,
            *args,
            **kwargs,
        )

    def _get_query(self, *args, **kwargs):
        version = kwargs.pop("version")
        query = super()._get_query(*args, **kwargs)
        query[
            "system_version"
        ] = f"version_{version}_{self.system_version_minor[version]}"
        query["hydrological_model"] = HYDROLOGICAL_MODELS[version]
        return query


class GlofasReanalysis(Glofas):
    def __init__(self):
        super().__init__(
            year_min=1979,
            year_max=2020,
            cds_name="cems-glofas-historical",
            dataset=["consolidated_reanalysis"],
            dataset_variable_name="dataset",
            system_version_minor={2: 1, 3: 1},
            date_variable_prefix="h",
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
            f"Downloading GloFAS reanalysis v{version} for years {year_min} - {year_max}"
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
            ds_new = self.get_station_dataset(
                stations=stations, ds=ds, coord_names=["time"]
            )
        # Write out the new dataset to a file
        self._write_to_processed_file(
            country_iso3=country_iso3,
            ds=ds_new,
            version=version,
        )


class GlofasForecast(Glofas):
    def __init__(self):
        super().__init__(
            year_min={2: 2019, 3: 2020},
            year_max=2020,
            cds_name="cems-glofas-forecast",
            dataset=["control_forecast", "ensemble_perturbed_forecasts"],
            system_version_minor={2: 1, 3: 1},
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
        year_min: int = None,
        year_max: int = None,
    ):
        year_min = self.year_min[version] if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Downloading GloFAS forecast v{version} for years {year_min} - {year_max} and lead time {leadtimes}"
        )
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            for leadtime in leadtimes:
                super()._download(
                    country_iso3=country_iso3,
                    area=area,
                    year=year,
                    leadtime=leadtime,
                    version=version,
                )

    def process(
        self,
        country_iso3: str,
        stations: Dict[str, Station],
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
    ):
        logger.info(f"Processing GloFAS Forecast v{version}")
        for leadtime in leadtimes:
            logger.info(f"For lead time {leadtime}")
            # Get list of files to open
            filepath_list = [
                self._get_raw_filepath(
                    country_iso3=country_iso3,
                    version=version,
                    year=year,
                    leadtime=leadtime,
                )
                for year in range(self.year_min[version], self.year_max + 1)
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_control_and_perturbed_datasets(filepath_list)
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = self.get_station_dataset(
                stations=stations, ds=ds, coord_names=["number", "time"]
            )
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_iso3=country_iso3,
                ds=ds_new,
                leadtime=leadtime,
                version=version,
            )


class GlofasReforecast(Glofas):
    def __init__(self):
        super().__init__(
            year_min=1999,
            year_max=2018,
            cds_name="cems-glofas-reforecast",
            dataset=["control_reforecast", "ensemble_perturbed_reforecasts"],
            system_version_minor={2: 2, 3: 1},
            date_variable_prefix="h",
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
        split_by_month: bool = False,
        year_min: int = None,
        year_max: int = None,
    ):
        year_min = self.year_min if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Downloading GloFAS reforecast v{version} for years {year_min} - {year_max} and lead time {leadtimes}"
        )
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            month_range = range(1, 13) if split_by_month else [None]
            for month in month_range:
                for leadtime in leadtimes:
                    super()._download(
                        country_iso3=country_iso3,
                        area=area,
                        version=version,
                        year=year,
                        month=month,
                        leadtime=leadtime,
                    )

    def process(
        self,
        country_iso3: str,
        stations: Dict[str, Station],
        leadtimes: List[int],
        version: int = DEFAULT_VERSION,
        split_by_month: bool = False,
    ):
        logger.info(f"Processing GloFAS Reforecast v{version}")
        for leadtime in leadtimes:
            logger.info(f"For lead time {leadtime}")
            # Get list of files to open
            month_range = range(1, 13) if split_by_month else [None]
            filepath_list = [
                self._get_raw_filepath(
                    country_iso3=country_iso3,
                    version=version,
                    year=year,
                    month=month,
                    leadtime=leadtime,
                )
                for year in range(self.year_min, self.year_max + 1)
                for month in month_range
            ]
            # Read in both the control and ensemble perturbed forecast and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_control_and_perturbed_datasets(
                filepath_list=filepath_list
            )
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            ds_new = self.get_station_dataset(
                stations=stations, ds=ds, coord_names=["number", "time"]
            )
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_iso3=country_iso3,
                ds=ds_new,
                leadtime=leadtime,
                version=version,
            )
