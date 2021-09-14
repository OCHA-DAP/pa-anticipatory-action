"""Download raster data from ECMWF's seasonal forecast for selected areas and
combines all dates into one dataframe."""
from pathlib import Path
import logging
import time
from datetime import datetime, timezone
import os
from typing import List
import xarray as xr
import cdsapi

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
    ):
        """Create an instance of a EcmwfSeasonal object, from which you
        can download and process raw data, and read in the processed
        data.

        :param year_min: The earliest year that the dataset is
        available. :param year_max: The most recent that the dataset is
        available :param cds_name: The name of the dataset in CDS :param
        dataset: The sub-datasets that you would like to download (as a
        list of strings) :param dataset_variable_name: The variable name
        with which to pass the above datasets in the CDS query
        """
        self.year_min = year_min
        self.year_max = year_max
        self.cds_name = cds_name
        self.dataset = dataset
        self.dataset_variable_name = dataset_variable_name

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
            return filepath
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
        return filepath

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
        filename = f"{country_iso3}_{self.cds_name}_v{version}_{year}"
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
            "area": area.list_for_api(round_val=1, offset_val=0),
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
        filename = f"{country_iso3}_{self.cds_name}_v{version}"
        filename += ".nc"
        return (
            PROCESSED_DATA_DIR / country_iso3 / ECMWF_SEASONAL_DIR / filename
        )

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
    def __init__(self):
        super().__init__(
            year_min=2000,
            year_max=2021,
            cds_name="seasonal-monthly-single-levels",
            dataset=["monthly_mean"],
            dataset_variable_name="product_type",
        )

    def download(
        self,
        country_iso3: str,
        area: Area,
        leadtimes: List[int] = None,
        version: int = DEFAULT_VERSION,
        split_by_month: bool = True,
    ):
        logger.info(
            f"Downloading ECMWF seasonal forecast v{version} for years"
            f" {self.year_min} - {self.year_max}"
        )
        current_date = datetime.now(timezone.utc)
        month_range = range(1, 13) if split_by_month else [None]
        if leadtimes is None:
            leadtimes = DEFAULT_LEADTIMES
        for year in range(self.year_min, self.year_max + 1):
            logger.info(f"...{year}")
            if split_by_month:
                if year < current_date.year:
                    month_range = range(1, 13)
                elif year == current_date.year:
                    # forecast becomes available on the 13th of the month
                    max_month = (
                        current_date.month
                        if current_date.day > 13
                        or (current_date.day == 13 and current_date.hour >= 12)
                        else current_date.month - 1
                    )
                    month_range = range(1, max_month + 1)
                elif year > current_date.year:
                    logger.info(
                        f"Cannot download data for {year}, because it is in"
                        " the future"
                    )
            else:
                month_range = [None]

            for month in month_range:
                super()._download(
                    country_iso3=country_iso3,
                    area=area,
                    year=year,
                    month=month,
                    version=version,
                    leadtimes=leadtimes,
                )

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
