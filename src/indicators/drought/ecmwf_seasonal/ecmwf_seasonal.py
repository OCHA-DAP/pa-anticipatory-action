"""Download raster data from ECMWF's seasonal forecast for selected areas and
combines all dates into one dataframe."""
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List

import cdsapi
import xarray as xr

from src.utils_general.area import Area

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) / "public"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ECMWF_SEASONAL_DIR = Path("ecmwf")
CDSAPI_CLIENT = cdsapi.Client()
DEFAULT_VERSION = 5
# monthly forecasts are produced with 1 to 6 months leadtime now always
# downloading all the lead times, might want to have one file per
# leadtime in the future
DEFAULT_LEADTIMES = list(range(1, 7))

logger = logging.getLogger(__name__)


class EcmwfSeasonal:
    def __init__(
        self,
        year_min: int,
        year_max: int,
        cds_name: str,
        dataset: List[str],
        dataset_variable_name: str,
        use_unrounded_area_coords: bool = False,
    ):
        """Create an instance of a EcmwfSeasonal object, from which you
        can download and process raw data, and read in the processed
        data.

        :param year_min: The earliest year that the dataset is
        available.
        :param year_max: The most recent that the dataset is
        available
        :param cds_name: The name of the dataset in CDS
        :param dataset: The sub-datasets that you would like to download (as a
        list of strings)
        :param dataset_variable_name: The variable name
        with which to pass the above datasets in the CDS query
        :param use_unrounded_area_coords: Generally not meant to be used,
        needed for backward compatibility with some historical data.
        If True, no rounding to the coordinates will be done which results in
        unroundedly shifted data
        """
        self.year_min = year_min
        self.year_max = date.today().year if year_max is None else year_max
        self.cds_name = cds_name
        self.dataset = dataset
        self.dataset_variable_name = dataset_variable_name
        self.use_unrounded_area_coords = use_unrounded_area_coords

    def _download(
        self,
        country_iso3: str,
        area: Area,
        version: int,
        year: int,
        leadtimes: List[int],
        month: int = None,
        use_cache: bool = True,
    ):
        filepath = self._get_raw_filepath(
            country_iso3=country_iso3,
            version=version,
            year=year,
            month=month,
        )
        # If caching is on and file already exists, don't download again
        if use_cache and filepath.exists():
            logger.debug(
                f"{filepath} already exists and cache is set to True, skipping"
            )
            return False
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Querying for {filepath}...")
        query_str = self._get_query(
            area=area,
            version=version,
            year=year,
            month=month,
            leadtimes=leadtimes,
        )
        logger.debug(query_str)
        CDSAPI_CLIENT.retrieve(
            name=self.cds_name,
            request=self._get_query(
                area=area,
                version=version,
                year=year,
                month=month,
                leadtimes=leadtimes,
            ),
            target=filepath,
        )
        logger.debug(f"...successfully downloaded {filepath}")
        # Wait 2 seconds between requests or else API hangs
        time.sleep(2)
        return True

    def _get_raw_filepath(
        self,
        country_iso3: str,
        version: int,
        year: int,
        month: int = None,
    ):
        directory = (
            RAW_DATA_DIR / country_iso3 / ECMWF_SEASONAL_DIR / self.cds_name
        )
        if self.use_unrounded_area_coords:
            directory = directory / "unrounded-coords"
        filename = f"{country_iso3}_{self.cds_name}_v{version}"
        if self.use_unrounded_area_coords:
            filename += "_unrounded-coords"
        filename += f"_{year}"
        if month is not None:
            filename += f"-{str(month).zfill(2)}"
        filename += ".grib"
        return directory / Path(filename)

    def _get_query(
        self,
        area: Area,
        version: int,
        year: int,
        leadtimes: List[int],
        month: int = None,
    ) -> dict:
        query = {
            "variable": "total_precipitation",
            "originating_centre": "ecmwf",
            "system": version,
            "format": "grib",
            self.dataset_variable_name: self.dataset,
            "year": str(year),
            "month": [str(x).zfill(2) for x in range(1, 13)]
            if month is None
            else str(month).zfill(2),
            "leadtime_month": [str(x) for x in leadtimes],
            "area": area.list_for_api(
                round_val=None if self.use_unrounded_area_coords else 1,
                offset_val=None if self.use_unrounded_area_coords else 0,
            ),
        }
        logger.debug(f"Query: {query}")
        return query

    @staticmethod
    def _read_in_monthly_mean_dataset(
        filepath_list: List[Path],
        leadtimes: List[int],
    ):
        """Read in the dataset for each date and combine them."""

        def _preprocess_monthly_mean_dataset(ds, leadtimes):
            # step is in timedelta (in nanoseconds), where the timedelta
            # is the end of the valid time of the forecast since the
            # nanoseconds depends on the length of the month, convert
            # this to the leadtime in months instead to be able to
            # compare across months other option could be to convert it
            # to the forecasted time, but gets difficult to concat all
            # different publication dates afterwards
            ds["step"] = leadtimes
            # ds["step"] = ds["time"] + ds["step"] time is the
            # publication month of the forecast, add this to the
            # dimensions to be able to merge different times
            ds = ds.expand_dims("time")

            return ds

        with xr.open_mfdataset(
            filepath_list,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            preprocess=lambda d: _preprocess_monthly_mean_dataset(
                d, leadtimes
            ),
        ) as ds:
            return ds

    def _write_to_processed_file(
        self,
        country_iso3: str,
        version: int,
        ds: xr.Dataset,
    ) -> Path:
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3,
            version=version,
        )
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to {filepath}")
        ds.to_netcdf(filepath)
        return filepath

    def _get_processed_filepath(self, country_iso3: str, version: int) -> Path:
        directory = (
            PROCESSED_DATA_DIR
            / country_iso3
            / ECMWF_SEASONAL_DIR
            / self.cds_name
        )
        if self.use_unrounded_area_coords:
            directory = directory / "unrounded-coords"
        filename = f"{country_iso3}_{self.cds_name}_v{version}"
        if self.use_unrounded_area_coords:
            filename += "_unrounded-coords"
        filename += ".nc"
        return directory / filename

    def read_processed_dataset(
        self,
        country_iso3: str,
        version: int = DEFAULT_VERSION,
    ):
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3, version=version
        )
        return xr.load_dataset(filepath)


class EcmwfSeasonalForecast(EcmwfSeasonal):
    def __init__(self, **kwargs):
        super().__init__(
            year_min=1993,
            year_max=None,
            cds_name="seasonal-monthly-single-levels",
            dataset=["monthly_mean"],
            dataset_variable_name="product_type",
            **kwargs,
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        leadtimes: List[int] = None,
        version: int = DEFAULT_VERSION,
        months: List[int] = None,
        year_min: int = None,
        year_max: int = None,
    ):
        year_min = self.year_min if year_min is None else year_min
        year_max = self.year_max if year_max is None else year_max
        logger.info(
            f"Downloading ECMWF seasonal forecast v{version} for years"
            f" {year_min} - {year_max}"
        )
        new_data_all = False
        current_date = datetime.now(timezone.utc)
        if leadtimes is None:
            leadtimes = DEFAULT_LEADTIMES
        for year in range(year_min, year_max + 1):
            logger.info(f"...{year}")
            if year > current_date.year:
                logger.info(
                    f"Cannot download data published in {year}, because it "
                    f"is in the future. Skipping year {year}."
                )
                continue
            else:
                if months is None:
                    if year < current_date.year:
                        months_year = range(1, 13)
                    elif year == current_date.year:
                        # forecast becomes available on the 13th of the month
                        # at 12 GMT
                        max_month = (
                            current_date.month
                            if current_date.day > 13
                            or (
                                current_date.day == 13
                                and current_date.hour >= 12
                            )
                            else current_date.month - 1
                        )
                        months_year = range(1, max_month + 1)

                else:
                    if year < current_date.year:
                        months_year = months
                    elif year == current_date.year:
                        # forecast becomes available on the 13th of the month
                        # at 12 GMT
                        max_month = (
                            current_date.month
                            if current_date.day > 13
                            or (
                                current_date.day == 13
                                and current_date.hour >= 12
                            )
                            else current_date.month - 1
                        )
                        months_year = [m for m in months if m <= max_month]

            for month in months_year:
                new_data = super()._download(
                    country_iso3=country_iso3,
                    area=area,
                    year=year,
                    month=month,
                    version=version,
                    leadtimes=leadtimes,
                )
                if not new_data_all and new_data:
                    new_data_all = True
        return new_data_all

    def process(
        self,
        country_iso3: str,
        leadtimes: List[int] = None,
        version: int = DEFAULT_VERSION,
    ):
        logger.info(f"Processing ECMWF Forecast v{version}")
        # Get list of files to open
        filepath_list = [
            self._get_raw_filepath(
                country_iso3=country_iso3,
                version=version,
                year=year,
                month=month,
            )
            for year in range(self.year_min, self.year_max + 1)
            for month in range(1, 13)
        ]
        # only include files that exist, e.g. if year_max=current year
        # then there might not be forecasts for all months
        filepath_list = [f for f in filepath_list if os.path.isfile(f)]

        if leadtimes is None:
            leadtimes = DEFAULT_LEADTIMES
        # Read in all forecasts and combine into one file
        logger.info(f"Reading in {len(filepath_list)} files")
        ds = self._read_in_monthly_mean_dataset(filepath_list, leadtimes)

        # Write out the new dataset to a file
        self._write_to_processed_file(
            country_iso3=country_iso3,
            ds=ds,
            version=version,
        )
