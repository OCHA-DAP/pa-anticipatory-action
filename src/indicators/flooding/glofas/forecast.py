import logging
from typing import Dict, List

from src.indicators.flooding.glofas import glofas
from src.utils_general.area import Area, Station

logger = logging.getLogger(__name__)


class GlofasForecastBase(glofas.Glofas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _download(
        self,
        is_reforecast: bool,
        country_iso3: str,
        area: Area,
        leadtimes: List[int],
        version: int = glofas.DEFAULT_VERSION,
        split_by_month: bool = False,
        split_by_leadtimes: bool = False,
        year_min: int = None,
        year_max: int = None,
    ):
        forecast_type = "reforecast" if is_reforecast else "forecast"
        year_min = self.year_min[version] if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Downloading GloFAS {forecast_type} v{version} for years"
            f" {year_min} - {year_max} and lead time {leadtimes}"
        )
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            month_range = range(1, 13) if split_by_month else [None]
            for month in month_range:
                leadtime_range = (
                    leadtimes if split_by_leadtimes else [leadtimes]
                )
                for leadtime in leadtime_range:
                    super()._download(
                        country_iso3=country_iso3,
                        area=area,
                        version=version,
                        year=year,
                        month=month,
                        leadtime=leadtime,
                    )

    def _process(
        self,
        is_reforecast: bool,
        country_iso3: str,
        stations: Dict[str, Station],
        leadtimes: List[int],
        version: int = glofas.DEFAULT_VERSION,
        split_by_month: bool = False,
        split_by_leadtimes: bool = False,
        year_min: int = None,
        year_max: int = None,
    ):
        forecast_type = "reforecast" if is_reforecast else "forecast"
        year_min = self.year_min[version] if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Processing GloFAS {forecast_type} v{version} for years"
            f" {year_min} - {year_max} and lead time {leadtimes}"
        )
        month_range = range(1, 13) if split_by_month else [None]
        leadtime_range = leadtimes if split_by_leadtimes else [leadtimes]
        for leadtime in leadtime_range:
            logger.info(f"For lead time {leadtime}")
            # Get list of files to open
            filepath_list = [
                self._get_raw_filepath(
                    country_iso3=country_iso3,
                    version=version,
                    year=year,
                    month=month,
                    leadtime=leadtime,
                )
                for year in range(year_min, year_max + 1)
                for month in month_range
            ]
            # Read in both the control and ensemble perturbed forecast
            # and combine
            logger.info(f"Reading in {len(filepath_list)} files")
            ds = self._read_in_ensemble_and_perturbed_datasets(
                filepath_list=filepath_list
            )
            # Create a new dataset with just the station pixels
            logger.info("Looping through stations, this takes some time")
            coord_names = ["number", "time"]
            if not split_by_leadtimes:
                coord_names += ["step"]
            ds_new = glofas.get_station_dataset(
                stations=stations,
                ds=ds,
                coord_names=coord_names,
            )
            # Write out the new dataset to a file
            self._write_to_processed_file(
                country_iso3=country_iso3,
                version=version,
                ds=ds_new,
                leadtime=leadtime,
            )


class GlofasForecast(GlofasForecastBase):
    def __init__(self, **kwargs):
        super().__init__(
            year_min={2: 2019, 3: 2020},
            year_max=2020,
            cds_name="cems-glofas-forecast",
            dataset=["control_forecast", "ensemble_perturbed_forecasts"],
            system_version_minor={2: 1, 3: 1},
            dataset_variable_name="product_type",
            **kwargs,
        )

    def download(self, *args, **kwargs):
        super()._download(is_reforecast=False, *args, **kwargs)

    def process(self, *args, **kwargs):
        super()._process(is_reforecast=False, *args, **kwargs)


class GlofasReforecast(GlofasForecastBase):
    def __init__(self, **kwargs):
        super().__init__(
            year_min={2: 1999, 3: 1999},
            year_max=2018,
            cds_name="cems-glofas-reforecast",
            dataset=["control_reforecast", "ensemble_perturbed_reforecasts"],
            dataset_variable_name="product_type",
            system_version_minor={2: 2, 3: 1},
            date_variable_prefix="h",
            **kwargs,
        )

    def download(self, *args, **kwargs):
        super()._download(is_reforecast=True, *args, **kwargs)

    def process(self, *args, **kwargs):
        super()._process(is_reforecast=True, *args, **kwargs)
