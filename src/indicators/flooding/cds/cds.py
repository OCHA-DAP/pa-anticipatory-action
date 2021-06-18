"""
Download data from CDS
"""
from pathlib import Path
import logging
import os
from typing import Dict, List, Union

import numpy as np
import xarray as xr
import cdsapi

from src.indicators.flooding.cds.area import Area, Station


DATA_DIR = Path(os.environ["AA_DATA_DIR"])
PUBLIC_DATA_DIR = "public"
RAW_DATA_DIR = "raw"
PROCESSED_DATA_DIR = "processed"

logger = logging.getLogger(__name__)


class Cds:
    def __init__(
        self,
        data_directory: str,
        cds_variable_name: str,
        year_min: Union[int, Dict[int, int]],
        year_max: int,
        cds_name: str,
        dataset: List[str],
        dataset_variable_name: str = "product_type",
        date_variable_prefix: str = "",
    ):
        """
        Create an instance of a CDS object, from which you can download and process raw data, and
        read in the processed data. The specifics of the datasets are defined in the child classes.
        :param year_min: The earliest year that the dataset is available. Can be a single integer,
        or a dictionary with structure {major_version: year_min} if the minimum year depends on the GloFAS
        model version.
        :param year_max: The most recent that the dataset is available
        :param cds_name: The name of the dataset in CDS
        :param dataset: The sub-datasets that you would like to download (as a list of strings)
        :param dataset_variable_name: The variable name with which to pass the above datasets in the CDS query
        :param date_variable_prefix: Some GloFAS datasets have the prefix "h" in front of some query keys
        """
        self.data_directory = data_directory
        self.cds_variable_name = cds_variable_name
        self.year_min = year_min
        self.year_max = year_max
        self.cds_name = cds_name
        self.dataset = dataset
        self.dataset_variable_name = dataset_variable_name
        self.date_variable_prefix = date_variable_prefix

    def _download(
        self,
        country_iso3: str,
        area: Area,
        year: int,
        month: int = None,
        leadtime: int = None,
        toggle_hours: bool = False,
        version: int = None,
        use_cache: bool = True,
    ):
        filepath = self._get_raw_filepath(
            country_iso3=country_iso3,
            version=version,
            year=year,
            month=month,
            leadtime=leadtime,
        )
        # If caching is on and file already exists, don't downlaod again
        if use_cache and filepath.exists():
            logger.debug(
                f"{filepath} already exists and cache is set to True, skipping"
            )
            return filepath
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Querying for {filepath}...")
        cdsapi.Client().retrieve(
            name=self.cds_name,
            request=self._get_query(
                area=area,
                version=version,
                year=year,
                month=month,
                leadtime=leadtime,
                toggle_hours=toggle_hours,
            ),
            target=filepath,
        )
        logger.debug(f"...successfully downloaded {filepath}")
        return filepath

    def _get_raw_filepath(
        self,
        country_iso3: str,
        year: int,
        version: int = None,
        month: int = None,
        leadtime: int = None,
    ):
        directory = (
            DATA_DIR
            / PUBLIC_DATA_DIR
            / RAW_DATA_DIR
            / country_iso3
            / self.data_directory
        )
        if version is not None:
            directory = directory / f"version_{version}"
        directory = directory / self.cds_name
        filename = f"{country_iso3}_{self.cds_name}"
        if version is not None:
            filename += f"_v{version}"
        filename += f"_{year}"
        if month is not None:
            filename += f"-{str(month).zfill(2)}"
        if leadtime is not None:
            filename += f"_lt{str(leadtime).zfill(2)}d"
        filename += ".grib"
        return directory / Path(filename)

    def _get_query(
        self,
        area: Area,
        year: int,
        month: int = None,
        leadtime: int = None,
        toggle_hours: bool = False,
        **kwargs
    ) -> dict:
        query = {
            "variable": self.cds_variable_name,
            "format": "grib",
            self.dataset_variable_name: self.dataset,
            f"{self.date_variable_prefix}year": str(year),
            f"{self.date_variable_prefix}month": [
                str(x + 1).zfill(2) for x in range(12)
            ]
            if month is None
            else str(month).zfill(2),
            f"{self.date_variable_prefix}day": [str(x + 1).zfill(2) for x in range(31)],
            "area": area.list_for_api(),
        }
        if leadtime is not None:
            query["leadtime_hour"] = str(leadtime * 24)
        if toggle_hours:
            query["time"] = [f"{str(x).zfill(2)}:00" for x in range(24)]
        logger.debug(f"Query: {query}")
        return query

    @staticmethod
    def _read_in_control_and_perturbed_datasets(filepath_list: List[Path]):
        """
        Read in dataset that has both control and ensemble perturbed forecast
        and combine them
        """
        ds_list = []
        for data_type in ["cf", "pf"]:
            with xr.open_mfdataset(
                filepath_list,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"dataType": data_type},
                },
            ) as ds:
                # Delete history attribute in order to merge
                del ds.attrs["history"]
                # Extra processing require for control forecast
                if data_type == "cf":
                    ds = expand_dims(
                        ds=ds,
                        dataset_name="dis24",
                        coord_names=["number", "time", "latitude", "longitude"],
                        expansion_dim=0,
                    )
                ds_list.append(ds)
        ds = xr.combine_by_coords(ds_list)
        return ds

    def _write_to_processed_file(
        self,
        country_iso3: str,
        version: int,
        ds: xr.Dataset,
        leadtime: int = None,
    ) -> Path:
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3,
            version=version,
            leadtime=leadtime,
        )
        Path(filepath.parent).mkdir(parents=True, exist_ok=True)
        # Netcdf seems to have problems overwriting; delete the file if it exists
        filepath.unlink(missing_ok=True)
        logger.info(f"Writing to {filepath}")
        ds.to_netcdf(filepath)
        return filepath

    def _get_processed_filepath(
        self, country_iso3: str, version: int = None, leadtime: int = None
    ) -> Path:
        filename = f"{country_iso3}_{self.cds_name}"
        if version is not None:
            filename += f"_v{version}"
        if leadtime is not None:
            filename += f"_lt{str(leadtime).zfill(2)}d"
        filename += ".nc"
        return DATA_DIR / PROCESSED_DATA_DIR / country_iso3 / self.data_directory / filename

    def read_processed_dataset(
        self,
        country_iso3: str,
        version: int = None,
        leadtime: int = None,
    ):
        filepath = self._get_processed_filepath(
            country_iso3=country_iso3,
            version=version,
            leadtime=leadtime,
        )
        return xr.load_dataset(filepath)


def expand_dims(
    ds: xr.Dataset, dataset_name: str, coord_names: list, expansion_dim: int
):
    """
    Using expand_dims seems to cause a bug with Dask like the one described here:
    https://github.com/pydata/xarray/issues/873 (it's supposed to be fixed though)
    """
    coords = {coord_name: ds[coord_name] for coord_name in coord_names}
    coords[coord_names[expansion_dim]] = [coords[coord_names[expansion_dim]]]
    ds = xr.Dataset(
        data_vars={
            dataset_name: (
                coord_names,
                np.expand_dims(ds[dataset_name].values, expansion_dim),
            )
        },
        coords=coords,
    )
    return ds


class CoordsOutOfBounds(Exception):
    def __init__(
        self,
        station_name: str,
        param_name: str,
        coord_station: float,
        coord_min: float,
        coord_max: float,
    ):
        message = (
            f"Station {station_name} has out-of-bounds {param_name} value of {coord_station} "
            f"(GloFAS {param_name} ranges from {coord_min} to {coord_max})"
        )
        super().__init__(message)


def get_station_dataset(
    stations: Dict[str, Station], ds: xr.Dataset, coord_names: List[str]
) -> xr.Dataset:
    # Check that lat and lon are in the bounds
    for station_name, station in stations.items():
        if not ds.longitude.min() < station.lon < ds.longitude.max():
            raise CoordsOutOfBounds(
                station_name=station_name,
                param_name="longitude",
                coord_station=station.lon,
                coord_min=ds.longitude.min().values,
                coord_max=ds.longitude.max().values,
            )
        if not ds.latitude.min() < station.lat < ds.latitude.max():
            raise CoordsOutOfBounds(
                station_name=station_name,
                param_name="latitude",
                coord_station=station.lat,
                coord_min=ds.latitude.min().values,
                coord_max=ds.latitude.max().values,
            )
    # If they are then return the correct pixel
    return xr.Dataset(
        data_vars={
            station_name: (
                coord_names,
                ds.isel(
                    longitude=np.abs(ds.longitude - station.lon).argmin(),
                    latitude=np.abs(ds.latitude - station.lat).argmin(),
                )["dis24"],
            )
            for station_name, station in stations.items()
        },
        coords={coord_name: ds[coord_name] for coord_name in coord_names},
    )
