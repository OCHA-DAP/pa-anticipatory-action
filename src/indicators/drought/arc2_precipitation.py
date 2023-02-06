import logging
import os
from datetime import date, datetime
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import shutil
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xarray as xr
from fiona.errors import DriverError
from rasterio.errors import RasterioIOError

from src.utils_general.raster_manipulation import compute_raster_statistics

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
PUBLIC_DATA_DIR = "public"
RAW_DATA_DIR = "raw"
PROCESSED_DATA_DIR = "processed"
ARC2_DIR = "arc2"
ARC2_CRS = "EPSG:4326"
T_COL = "T"

# TODO: Convert documentation to Sphinx

# TODO: Transfer to pa-aa-toolbox

# TODO: Once in pa-aa-toolbox, use the new COD and other
# file functionality to change default processing for
# aggregating and allow linking the dry spell
# identification with re-downloading

# TODO: Add CHIRPS class and sort out dual inheritance

# TODO: Better logging

# TODO: Write tests

# TODO: Change the function order

# TODO: make class variables private where applicable
# when porting to toolbox


class ARC2:
    """Summary of class

    TODO: fill, but this class will be replaced by a pipeline class
    and much of the code integrated into different aspects of the
    toolbox format, so not bothering to fill out now

    Attributes:
        country_iso3: ISO3 string.
        date_min: Minimum date to load data from, either string
            in ISO 8601 format, e.g. '2021-03-20' or `datetime.date` object.
        date_max: Maximum date to load data from, either string
            in ISO 8601 format, e.g. '2021-04-20' or `datetime.date` object.
        range_x: Tuple of strings specifying longitude range
            for download, e.g. ('32E', '36E'). Must be one or two numbers
            with the letter indicating the half of the globe.
        range_y: Tuple of strings specifying latitude range
            for download, e.g. ('20S', '5S'). Must be one or two numbers
            with the letter indicating the half of the globe.
    """

    def __init__(
        self,
        country_iso3: str,
        date_min: Union[str, date],
        date_max: Union[str, date],
        range_x: Tuple[str, str],
        range_y: Tuple[str, str],
    ):
        self.country_iso3 = country_iso3

        if not isinstance(date_min, date):
            date_min = date.fromisoformat(date_min)
        self.date_min = date_min

        if not isinstance(date_max, date):
            date_max = date.fromisoformat(date_max)
        self.date_max = date_max

        self.range_x = range_x
        self.range_y = range_y

        # store the dates for latest and earliest
        # available data
        raw_fp = self._get_raw_filepath(True)
        if raw_fp.exists():
            self._update_available_dates()
        else:
            self.latest_available_date = None
            self.earliest_available_date = None

    def load_raw_data(
        self, raw_filepath: Union[Path, None] = None, convert_date: bool = True
    ) -> pd.DataFrame:
        """
        Convenience function to load raw raster data, squeeze
        it and write its CRS. The function always accesses
        the main file if a filepath is not provided.

        :param raw_filepath: Path to raw file to load. If `None`,
            loads main file.
        :param convert_date: Convert date into datetime index.
            If planning to save the raw data later, don't convert
            because xarray doesn't know how to parse the date.
        """
        if raw_filepath is None:
            raw_filepath = self._get_raw_filepath(main=True)

        # load raw main data
        try:
            with xr.open_dataarray(raw_filepath) as ds:
                da = ds.load()
                da = da.squeeze().rio.write_crs(ARC2_CRS)

        except RasterioIOError:
            raise OSError(
                "Raw raster file %s does not exist, "
                "first download using `download()`.",
                raw_filepath.name,
            )

        # explicitly remove missing values
        da.values[da.values == -999] = np.NaN

        # convert to standard date
        if convert_date:
            t_index = da.indexes[T_COL]
            if not isinstance(t_index, pd.DatetimeIndex):
                t_index = t_index.to_datetimeindex()
            da[T_COL] = t_index.date

        return da

    # TODO: redownload option for toolbox in case data gets
    # updated.

    # TODO: check for dates first for clarity

    # TODO: reconsider replace_missing during refactor as part
    # of shift to using clobber like the rest

    def download(
        self,
        main: bool = False,
        replace_missing: bool = False,
        start_replace_date: Optional[str] = None,
    ):
        """
        Download ARC2 data for all dates between `self.date_min`
        and `self.date_max`. If `main`, then all data
        is downloaded directly from the servers and set as the
        main file. If not `main`, then data
        is only downloaded for dates not already
        available in the main file, and then merged
        into the main file.

        :param main: Boolean on whether to set download as
            main if `True`, or only download missing data
            and merge to main if `False`.
        :param replace_missing: Only relevant if not `main`.
            If `True`, looks for missing values through entire
            dataset and ensures they are re-downloaded.
        :param start_replace_date: Only relevant if
            ``replace_missing``. If an ISO 8601 string is
            passed, then missing values are only looked for on
            or after that date.
        """
        if main:
            self._download_date_ranges(self.date_min, self.date_max, main)
        else:
            self._check_main_exists()
            # load main data and find all dates covered
            # compare to min/max and then download missing
            da_main = self.load_raw_data()

            # drop missing data if requested
            if replace_missing:
                if start_replace_date:
                    replace_date = datetime.fromisoformat(start_replace_date)
                else:
                    replace_date = datetime.combine(
                        self.date_min, datetime.min.time()
                    )

                t_subset = da_main.indexes[T_COL] < replace_date
                val_subset = np.max(
                    np.max(np.isnan(da_main.values), array=1), array=1
                )

                # keeps rows if date is less than specified or
                # value is different than -999 (missing)
                da_main = da_main.loc[t_subset | val_subset, :, :]

            # subtracting 12 hours to ensure they match with dates
            # generated from pd.date_range()
            loaded_dates = da_main.indexes[T_COL] - pd.to_timedelta("12:00:00")
            full_dates = pd.date_range(self.date_min, self.date_max)
            needed_dates = full_dates.difference(loaded_dates)

            if len(needed_dates) > 0:
                date_ranges = self._group_date_ranges(needed_dates)
                for dates in date_ranges:
                    self._download_date_ranges(
                        date_min_dl=dates[0],
                        date_max_dl=dates[1],
                        main=False,
                    )

                self._sort_raw_data()
            else:
                logger.info("No additional data needs downloading.")
        self._update_available_dates()

    def _download_date_ranges(
        self, date_min_dl: date, date_max_dl: date, main: bool = False
    ):

        """
        Download data from IRI servers for a specific date range
        and longitude/latitude bounds. If `main`, the file
        is fully loaded as the main file. Otherwise, it is
        temporarily loaded and merged into the main file before
        removal.

        :param date_min_dl: `datetime.date` specifying first date
            to download from.
        :param date_max_dl: `datetime.date` specifying last date
            to download from.
        :param main: Boolean specifying whether download is main
            file. If `True`, existing main is removed and full
            file is saved. If `False`, download is temporarily saved
            and then merged to main and deleted.
        """

        url = (
            f"https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/"
            f".Africa/.DAILY/.ARC2/.daily/.est_prcp/T/"
            f"%28{date_min_dl.day}%20{date_min_dl:%b%%20%Y}%29"
            f"%28{date_max_dl.day}%20{date_max_dl:%b%%20%Y}%29RANGEEDGES/"
            f"X/%28{self.range_x[0]}%29%28{self.range_x[1]}%29RANGEEDGES/"
            f"Y/%28{self.range_y[0]}%29%28{self.range_y[1]}%29RANGEEDGES/"
            f"data.nc"
        )

        raw_filepath = self._get_raw_filepath(main=main)

        if main:
            logger.info(
                "Removing existing main raw ARC2 file before re-downloading."
            )
            raw_filepath.unlink(missing_ok=True)

        logger.info(
            f"Downloading ARC2 data from {self.date_min:%d %b %Y} to "
            f"{self.date_max:%d %b %Y} "
            f"covering longitudes {self.range_x[0]} to {self.range_x[1]} "
            f"and latitudes {self.range_y[0]} to {self.range_y[1]}."
        )

        # TODO: explore specific errors generated by request
        try:
            response = requests.get(url, verify=False)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        if response.status_code == 404:
            logger.error("No data available from %s.", url)
            return

        # create folders if necessary and write file
        raw_filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(raw_filepath, "wb") as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)

        # merge to main and delete temporary files
        if not main:
            main_filepath = self._get_raw_filepath(main=True)
            temp_filepath = self._get_raw_filepath(main=True, temp=True)

            logger.info(
                f"Merging ARC2 data from {self.date_min:%d %b %Y} "
                f"to {self.date_max:%d %b %Y} "
                f"into main file: {main_filepath.name}."
            )
            # copy main over to copy and then unlink
            shutil.copy(main_filepath, temp_filepath)
            main_filepath.unlink()

            main_da = xr.open_dataarray(temp_filepath)
            raw_da = xr.open_dataarray(raw_filepath)

            main_merge = xr.concat([main_da, raw_da], dim=T_COL)

            # Ensuring fill value encoding is properly set in new
            # merged dataset
            main_merge.encoding["_FillValue"] = -999
           
            main_merge.to_netcdf(main_filepath)
            raw_filepath.unlink()
            temp_filepath.unlink()


    def _get_directory(self, dir: Union[Path, str]) -> Path:
        """
        Return ARC2 directory, either in public, processed
        or raw for a specific ISO3.

        :param dir: Path or string of folder name in public for passing.
        """
        directory = Path(
            DATA_DIR / PUBLIC_DATA_DIR / dir / self.country_iso3 / ARC2_DIR
        )
        return directory

    def _get_raw_filepath(self, main: bool, temp: bool = False) -> Path:
        """
        Return filepath to raw ARC2 data for specific x
        and y bounds. All data stored within a single
        main file. If `main`, the filepath is not
        specified for a specific date range and designated
        with 'main.nc' at the end. Otherwise, end of
        filepath designated with `self.date_min` and
        `self.date_max`.

        :param main: Whether or not the filepath is
            for the main file.
        :param temp: Whether or not the filepath is
            for a temp file used for saving.
        """
        directory = self._get_directory(RAW_DATA_DIR)

        if main:
            if not temp:
                end = "main"
            else:
                end = "main_copy.nc"
        else:
            end = f"{self.date_min:%d_%b_%Y}_{self.date_max:%d_%b_%Y}"

        filename = (
            f"arc2_daily_precip_{self.country_iso3}_"
            f"{self.range_x[0]}_{self.range_x[1]}_"
            f"{self.range_y[0]}_{self.range_y[1]}_{end}.nc"
        )
        return directory / filename

    def _check_main_exists(self):
        """Check if main file exists

        Used to check if main file exists
        """
        fp = self._get_raw_filepath(main=True)
        if not fp.exists():
            raise OSError(
                "Main file does not exist. " "First run `download(main=True)`."
            )

    def _group_date_ranges(self, dates) -> list:
        """
        Group range of dates into list of consecutive dates to pass to
        `self._download_date_ranges()` for calling to the API. For each
        group, only the min and max dates are returned rather than the
        full list of dates in the group.

        :param dates: Date range generated by `pd.date_range()`
        """
        date_ranges = []

        for _, g in groupby(
            enumerate(dates),
            key=lambda x: x[0] - (x[1] - datetime(1970, 1, 1)).days,
        ):
            group = map(itemgetter(1), g)
            group = list(map(pd.Timestamp, group))
            if len(group) == 1:
                group.append(group[0])
            date_ranges.append((group[0], group[-1]))

        return date_ranges

    def _sort_raw_data(self):
        """
        Sort main file by time coordinates to ensure
        correct ordering.
        """
        main_filepath = self._get_raw_filepath(main=True)
        with xr.open_dataarray(main_filepath) as ds:
            main = ds.load()

        main.encoding["_FillValue"] = -999
        main.sortby(T_COL).to_netcdf(main_filepath)

    def _update_available_dates(self):
        """
        Update available dates, earliest and latest,
        based on dates in the raw data NetCDF file.
        """
        da = self.load_raw_data()
        self.latest_available_date = max(da.indexes[T_COL])
        self.earliest_available_date = min(da.indexes[T_COL])


class DrySpells(ARC2):
    """
    Dry spells

    TODO: fill and link to ARC2
    params using Sphinx so no
    need for duplication

    monitoring_start: Minimum date to load data from, either string
        in ISO 8601 format, e.g. '2021-03-20' or `datetime.date` object.
        Passed to `ARC2.date_min` as `monitoring_start - rolling_window`
        days to ensure sufficient data for calculating rolling sum
        available.
    monitoring_end: Maximum date to load data from, either string
        in ISO 8601 format, e.g. '2021-04-20' or `datetime.date` object.
    :param agg_method: One of 'centroid',  'touching', or 'approximate_mask'.
        If 'approximate_mask', the data is upsampled 4x before aggregating
        using the 'centroid' method.
    :param rolling_window: Number of days for rolling sum of precipitation.
    :param rainfall_mm: Maximum cumulative precipitation during window to
        classify as dry spell.
    :param polygon_path: Path to polygon file for clipping and aggregating
        raster data.
    :param bound_col: Column in polygon file to aggregate raster to.
    """

    def __init__(
        self,
        country_iso3: str,
        polygon_path: Union[Path, str],
        bound_col: str,
        monitoring_start: Union[str, date],
        monitoring_end: Union[str, date],
        range_x: Tuple[str, str],
        range_y: Tuple[str, str],
        agg_method: str = "centroid",
        rolling_window: int = 14,
        rainfall_mm: int = 2,
    ):
        if not isinstance(monitoring_start, date):
            monitoring_start = date.fromisoformat(monitoring_start)

        if not isinstance(monitoring_end, date):
            monitoring_end = date.fromisoformat(monitoring_end)

        self.monitoring_start = monitoring_start
        self.monitoring_end = monitoring_end
        self.agg_method = agg_method
        self.rolling_window = rolling_window
        self.rainfall_mm = rainfall_mm
        self.polygon_path = polygon_path
        self.bound_col = bound_col

        super().__init__(
            country_iso3=country_iso3,
            date_min=monitoring_start
            - pd.to_timedelta(self.rolling_window - 1, unit="d"),
            date_max=monitoring_end,
            range_x=range_x,
            range_y=range_y,
        )

    # TODO: Consider dropping reprocessing because not necessary

    # TODO: Remove redownload when refactoring and just check that dates
    # are available.

    def aggregate_data(
        self, reprocess: bool = False, redownload: bool = False
    ) -> pd.DataFrame:
        """
        Get mean aggregation by admin boundary for the downloaded ARC2 data.
        Outputs a csv with daily aggregated statistics. If data already
        aggregated between `self.date_min` and `self.date_max`, returns
        pre-aggregated data, otherwise aggregates additional data and joins
        to aggregated main file.

        Where data is missing, it is infilled using linear interpolation for
        the day before and after the missing date. This is done because missing
        data as observed is more geographically correlated than temporally, so
        data for interpolation is typically available on previous and following
        days but often unavailable across large areas on the same day.

        :param reprocess: Boolean, if `True` reprocesses all raster data.
            Otherwise, only processes dates that have not already been
            aggregated.
        :param redownload: Boolean, if `True`, calls `download()`
            without replacing missing data or the main file,
            only downloading for new dates not already downloaded.
        """
        self._check_main_exists()

        if redownload:
            self.download(main=False, replace_missing=False)

        aggregated_filepath = self._get_aggregated_filepath()
        aggregated_filepath.parent.mkdir(parents=True, exist_ok=True)

        da = self.load_raw_data()

        # load polygon data
        try:
            gdf = gpd.read_file(self.polygon_path)
        except DriverError:
            raise OSError(
                "Clip file %s does not exist.", Path(self.polygon_path).name
            )

        # only process data for dates that have not already been aggregated
        exist_stats = None
        if aggregated_filepath.exists() and not reprocess:
            exist_time = pd.read_csv(
                aggregated_filepath,
                parse_dates=[T_COL],
                usecols=[T_COL],
                index_col=[T_COL],
            ).index.date

            da = da.sel(T=~da[T_COL].isin(exist_time))
            exist_stats = self.load_aggregated_data()

            if da.shape[0] == 0:
                logger.info(
                    "No additional dates to process for %s.",
                    aggregated_filepath,
                )
                return exist_stats

        all_touched = self.agg_method == "touching"
        if self.agg_method == "approximate_mask":
            x_res, y_res = da.rio.resolution()
            da = da.rio.reproject(
                da.rio.crs, resolution=(x_res / 4, y_res / 4)
            ).rename({"x":"X", "y":"Y"})

        df_zonal_stats = compute_raster_statistics(
            gdf=gdf,
            bound_col=self.bound_col,
            lon_coord="X",
            lat_coord="Y",
            raster_array=da,
            all_touched=all_touched,
            stats_list=["mean"],
        )

        # join up existing data if necessary
        if exist_stats is not None:
            # explicitly convert to datetime64 series before joining
            # and add infilled to ensure what's happening
            df_zonal_stats[T_COL] = pd.to_datetime(df_zonal_stats[T_COL])
            df_zonal_stats = df_zonal_stats.append(
                exist_stats, ignore_index=True
            )
            df_zonal_stats.sort_values(by=[T_COL], inplace=True)

        # infill missing data with interpolation
        data_col = f"mean_{self.bound_col}"

        df_zonal_stats["infilled"] = np.where(
            df_zonal_stats[data_col].isna(), True, False
        )

        # TODO: look into limiting NA infilling only if there's
        # enough data value
        df_zonal_stats[data_col] = df_zonal_stats.groupby(self.bound_col)[
            data_col
        ].transform(lambda x: x.interpolate())

        df_zonal_stats.to_csv(aggregated_filepath, index=False)
        return df_zonal_stats

    def load_aggregated_data(self, filter: bool = False) -> pd.DataFrame:
        """
        Load admin aggregated data. If `filter`, then
        just `filter` the data frame to the period for monitoring.
        """
        aggregated_fp = self._get_aggregated_filepath()
        df = pd.read_csv(aggregated_fp, parse_dates=[T_COL])

        if filter:
            df = df[
                (df[T_COL].dt.date >= self.date_min)
                & (df[T_COL].dt.date <= self.date_max)
            ]
        return df

    # TODO: look into whether we want to include the reprocess code
    # when bringing into the toolbox
    def calculate_rolling_sum(self) -> pd.DataFrame:
        """
        Calculates rolling sum from the latest aggregated
        data, based on the DrySpell objects window of
        observations and aggregation method.
        """
        df = self.load_aggregated_data()
        rollsum_df = self._calculate_rolling_sum(df)
        rollsum_fp = self._get_rolling_sum_filepath()
        rollsum_df.to_csv(rollsum_fp, index=False)
        return rollsum_df

    def load_rolling_sum_data(
        self, filter_dates: bool = False
    ) -> pd.DataFrame:
        """Load rolling sum data"""
        rollsum_fp = self._get_rolling_sum_filepath()
        df = pd.read_csv(rollsum_fp, parse_dates=[T_COL])
        if filter_dates:
            df = df[
                (df[T_COL].dt.date >= self.date_min)
                & (df[T_COL].dt.date <= self.date_max)
            ]
        return df

    def identify_dry_spells(self, reprocess: bool = False) -> pd.DataFrame:
        """
        Identifies dry spells based on latest rolling
        sum values.

        :param reprocess: Boolean, if `True` reprocesses
        aggregated data by calculating rolling sums.
        """
        if reprocess:
            self.calculate_rolling_sum()

        rollsum_fp = self._get_rolling_sum_filepath()
        df = pd.read_csv(rollsum_fp, parse_dates=[T_COL])

        adm_col = self.bound_col
        rs_col = f"rolling_sum_{self.rolling_window}_days"

        # Identify all dry spells and unique consecutive groups
        df["ds"] = df[rs_col] <= self.rainfall_mm
        df["dsg"] = (df.groupby(adm_col)["ds"].diff() != 0).cumsum()

        # Generate data frame of dry spells
        ds_df = (
            df[df["ds"]]
            .groupby("dsg")
            .agg(
                adm_col=(adm_col, "unique"),
                ds_confirmation=(T_COL, "min"),
                ds_last_date=(T_COL, "max"),
            )
            .reset_index(drop=True)
            .assign(
                adm_col=lambda x: x.adm_col.str[0],
                ds_first_date=lambda x: x.ds_confirmation
                - pd.to_timedelta(self.rolling_window - 1, unit="d"),
                ds_duration=lambda x: (
                    x.ds_last_date - x.ds_first_date
                ).dt.days
                + 1,
            )
            .rename(columns={"adm_col": adm_col})
        )

        # Calculate rainfall within each dry spell
        precip_df = self.load_aggregated_data()
        precip_col = f"mean_{self.bound_col}"

        total_df = pd.merge(
            left=ds_df, right=precip_df, on=adm_col, how="outer"
        )

        # Filter dates to just those within the dry spell and sum
        total_df = total_df[total_df[T_COL] >= total_df["ds_first_date"]]
        total_df = total_df[total_df[T_COL] <= total_df["ds_last_date"]]
        total_df = (
            total_df.groupby([adm_col, "ds_confirmation"])[precip_col]
            .agg("sum")
            .reset_index()
        )

        # Add dry spell rainfall to main data frame
        total_df.rename(columns={precip_col: "ds_rainfall"}, inplace=True)

        ds_df = pd.merge(
            left=ds_df,
            right=total_df,
            on=[adm_col, "ds_confirmation"],
            how="left",
        )

        # Re-arrange dry spell data frame
        cols = [
            adm_col,
            "ds_first_date",
            "ds_confirmation",
            "ds_last_date",
            "ds_duration",
            "ds_rainfall",
        ]
        ds_df = ds_df[cols]

        ds_fp = self._get_dry_spell_filepath()
        ds_df.to_csv(ds_fp, index=False)

        return ds_df

    def load_dry_spell_data(self, filter: bool = True) -> pd.DataFrame:
        """
        Load dry spells classified through method.
        """
        fp = self._get_dry_spell_filepath()
        df = pd.read_csv(
            filepath_or_buffer=fp,
            parse_dates=["ds_first_date", "ds_confirmation", "ds_last_date"],
        )
        if filter and df.shape[0] > 0:
            t_col = "ds_confirmation"
            df = df[
                (df[t_col].dt.date >= self.date_min)
                & (df[t_col].dt.date <= self.date_max)
            ]
        return df

    # TODO: Unused functionality from Hannah's very first original code.
    # to be removed or used during refactor
    #
    # def _write_to_monitoring_file(self, dry_spells: Union[None, int] = None):
    #     monitoring_file = self._get_monitoring_filepath(
    #         self.country_iso3, self.date_max
    #     )
    #     """
    #     Write a simple output of the number
    #     of dry spells observed in the last 14 days.
    #     """
    #     result = ""
    #     with open(monitoring_file, "w") as f:
    #         if dry_spells is not None:
    #             result += "No dry spells identified in the last 14 days."
    #             f.write(result)
    #         else:
    #             f.write(
    #                 f"Dry spells identified in \
    #                 {len(dry_spells)} admin regions:\n{dry_spells}"
    #             )
    #     f.close()
    #     return

    def find_longest_runs(self, filter: bool = True):
        """Find longest runs under mm of rainfall

        Defaults to only finding the longest runs
        across the dates of interest, otherwise
        the algorithm for calculating would take
        an extremely long time.
        """
        df = self.load_aggregated_data(filter=filter)

        precip_col = f"mean_{self.bound_col}"
        adm_col = self.bound_col

        def _find_longest_run(x, threshold):
            for i in range(len(x)):
                rs = x.rolling(i).sum()
                if ~np.any(rs <= threshold):
                    return i - 1
            return i

        df_agg = df.groupby(adm_col).agg(
            longest_run=(
                precip_col,
                lambda x: _find_longest_run(x, self.rainfall_mm),
            )
        )

        return df_agg

    def count_dry_days(self, filter: bool = True):
        """Find number of days with no rainfall in period

        Defaults to only finding the days with no rainfall
        across the dates of interest.
        """
        df = self.load_aggregated_data(filter=filter)

        precip_col = f"mean_{self.bound_col}"
        adm_col = self.bound_col

        df_agg = df.groupby(adm_col).agg(
            dry_days=(
                precip_col,
                lambda x: sum(x == 0),
            )
        )

        return df_agg

    def count_rainy_days(self, filter: bool = True):
        """Find number of days with >= 4mm rainfall in period

        Defaults to only finding the days with no rainfall
        across the dates of interest.
        """
        df = self.load_aggregated_data(filter=filter)

        precip_col = f"mean_{self.bound_col}"
        adm_col = self.bound_col

        df_agg = df.groupby(adm_col).agg(
            rainy_days=(
                precip_col,
                lambda x: sum(x >= 4),
            )
        )

        return df_agg

    def count_dry_spells(self, filter: bool = True) -> int:
        """Return the number of admins in dry spell

        Defaults to returning the number of unique
        admins within the monitoring period, otherwise
        returns across the entire data frame.
        """
        df = self.load_dry_spell_data(filter=filter)
        return len(np.unique(df.iloc[:, 0]))

    def days_under_threshold(self, raster: bool = True) -> pd.DataFrame:
        """Calculate days precipitation has been under threshold

        From the most recent day in the dataset, calculates the
        number of days the rolling sum has remained under the
        threshold for each administrative area. The calculations
        are always performed solely on data in the monitoring
        period, and checks consecutive days from the end of the
        monitoring period.

        If `raster`, calculates on the raw raster file. Otherwise
        calculates on data aggregated to the administrative
        boundaries.
        """
        if raster:
            da = self.load_raw_data()
            da = da[
                (da.indexes[T_COL] >= self.date_min)
                & (da.indexes[T_COL] <= self.date_max),
                :,
                :,
            ]
            da = da.reindex(T=list(reversed(da.indexes[T_COL])))
            return xr.where(
                da.cumsum(dim=T_COL) <= self.rainfall_mm, 0, 1
            ).argmax(dim=T_COL)

        else:
            df = self.load_aggregated_data(filter=True)
            precip_col = f"mean_{self.bound_col}"
            adm_col = self.bound_col
            df = (
                df.iloc[::-1]
                .groupby(adm_col)
                .agg(
                    days_under_threshold=(
                        precip_col,
                        lambda x: sum(x.cumsum() <= self.rainfall_mm),
                    )
                )
                .reset_index()
            )
            return df

    def count_days_under_threshold(self, number_days: int) -> int:
        """Count number of admin areas under threshold for `number_days`

        Counts the number of administrative areas under
        the threshold for greater than or equal to
        `number_days`. The calculations
        are always performed solely on data in the monitoring
        period, and checks consecutive days from the end of the
        monitoring period.
        """
        df = self.days_under_threshold(raster=False)
        return sum(df.iloc[:, 1] >= number_days)

    def cumulative_rainfall(
        self, date_min: date = None, date_max: date = None
    ) -> xr.DataArray:
        """Calculate cumulative rainfall across monitoring period"""
        da = self.load_raw_data()
        da_date = da.indexes[T_COL]
        if date_min is None:
            date_min = self.date_min
        if date_max is None:
            date_max = self.date_max
        da = da[(da_date >= date_min) & (da_date <= date_max), :, :]
        return da.sum(dim=T_COL)

    def _get_aggregated_filepath(self) -> Path:
        """
        Return filepath to aggregated ARC2 data aggregated to ADM2 level
        using arithmetic mean. All data stored within a single main file
        for unique ISO3, aggregation method, and geographic range.
        """
        directory = self._get_directory(PROCESSED_DATA_DIR)
        filename = (
            f"arc2_{self.agg_method}_long_{self.country_iso3}_"
            f"{self.range_x[0]}_{self.range_x[1]}_"
            f"{self.range_y[0]}_{self.range_y[1]}_"
            f"main.csv"
        )
        return directory / filename

    def _calculate_rolling_sum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling sum on specific data frame.
        """
        precip_col = f"mean_{self.bound_col}"
        adm_col = self.bound_col

        rollsum_col = f"rolling_sum_{self.rolling_window}_days"

        rollsum = (
            df.groupby(adm_col)
            .apply(
                lambda x: x.set_index(T_COL)
                .rolling(
                    f"{self.rolling_window}D", min_periods=self.rolling_window
                )
                .sum()
            )[precip_col]
            .rename(rollsum_col)
        )

        df = df.join(rollsum, on=[adm_col, T_COL])

        df.dropna(subset=[rollsum_col], inplace=True)

        df = df[[T_COL, adm_col, rollsum_col]].reset_index(drop=True)

        return df

    def _get_rolling_sum_filepath(self) -> Path:
        """
        Return filepath to ARC2 rolling sum values.
        """
        directory = self._get_directory(PROCESSED_DATA_DIR)
        filename = (
            f"arc2_{self.agg_method}_long_{self.country_iso3}_"
            f"rolling_sum_{self.rolling_window}_days_"
            f"{self.range_x[0]}_{self.range_x[1]}_"
            f"{self.range_y[0]}_{self.range_y[1]}_"
            f"main.csv"
        )
        return directory / filename

    def _get_dry_spell_filepath(self) -> Path:
        """
        Return filepath to ARC2 rolling sum values.
        """
        directory = self._get_directory(PROCESSED_DATA_DIR)
        filename = (
            f"arc2_{self.agg_method}_{self.country_iso3}_"
            f"dry_spells_{self.rolling_window}_days_"
            f"{self.range_x[0]}_{self.range_x[1]}_"
            f"{self.range_y[0]}_{self.range_y[1]}_"
            f"main.csv"
        )
        return directory / filename
