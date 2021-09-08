from unittest import mock
from pathlib import Path
import shutil

import numpy as np
import xarray as xr
import pandas as pd

from src.indicators.flooding.glofas import glofas
from src.indicators.flooding.glofas.area import Area, Station


TMP_PATH = Path("/tmp/glofas_test")


def test_expand_dims():
    rs = np.random.RandomState(12345)
    size_x, size_y = (10, 20)
    ds = xr.Dataset(
        data_vars={"var_a": (("x", "y"), rs.rand(size_x, size_y))},
        coords={"x": np.arange(size_x), "y": np.arange(size_y)},
    )
    ds.coords["z"] = 1
    assert "z" not in ds.dims.keys()
    ds = glofas.expand_dims(
        ds=ds,
        dataset_name="var_a",
        coord_names=["z", "x", "y"],
        expansion_dim=0,
    )
    assert "z" in ds.dims.keys()


@mock.patch("src.indicators.flooding.glofas.glofas.cdsapi.Client.retrieve")
@mock.patch("src.indicators.flooding.glofas.glofas.Path.mkdir")
@mock.patch.object(glofas, "DATA_DIR", TMP_PATH)
class TestDownload:
    def setup(self):
        self.country_iso3 = "abc"
        self.area = Area(north=1, south=-2, east=3, west=-4)
        self.year = 2000
        self.leadtimes = [10, 20]
        self.expected_area = [1.05, -4.05, -2.05, 3.05]
        self.expected_months = [str(x + 1).zfill(2) for x in range(12)]
        self.expected_days = [str(x + 1).zfill(2) for x in range(31)]
        self.expected_leadtime = ["240", "480"]

    def test_reanalysis_download(self, fake_mkdir, fake_retrieve):
        glofas_reanalysis = glofas.GlofasReanalysis()
        glofas_reanalysis.download(
            country_iso3=self.country_iso3,
            area=self.area,
            year_min=self.year,
            year_max=self.year,
        )
        expected_args = {
            "name": "cems-glofas-historical",
            "request": {
                "variable": "river_discharge_in_the_last_24_hours",
                "format": "grib",
                "dataset": ["consolidated_reanalysis"],
                "hyear": f"{self.year}",
                "hmonth": self.expected_months,
                "hday": self.expected_days,
                "area": self.expected_area,
                "system_version": "version_3_1",
                "hydrological_model": "lisflood",
            },
            "target": Path(
                f"{TMP_PATH}/public/raw/{self.country_iso3}"
                "/glofas/version_3/cems-glofas-historical"
                f"/{self.country_iso3}_cems-glofas-historical_v3_2000.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)

    def test_forecast_download(self, fake_mkdir, fake_retrieve):
        glofas_forecast = glofas.GlofasForecast()
        glofas_forecast.download(
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=self.leadtimes,
            year_min=self.year,
            year_max=self.year,
        )
        expected_args = {
            "name": "cems-glofas-forecast",
            "request": {
                "variable": "river_discharge_in_the_last_24_hours",
                "format": "grib",
                "product_type": [
                    "control_forecast",
                    "ensemble_perturbed_forecasts",
                ],
                "year": f"{self.year}",
                "month": self.expected_months,
                "day": self.expected_days,
                "area": self.expected_area,
                "system_version": "version_3_1",
                "hydrological_model": "lisflood",
                "leadtime_hour": self.expected_leadtime,
            },
            "target": Path(
                f"{TMP_PATH}/public/raw/{self.country_iso3}"
                f"/glofas/version_3/cems-glofas-forecast"
                f"/{self.country_iso3}_cems-glofas-forecast_v3_2000.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)

    def get_reforecast_expected_args(self):
        return {
            "name": "cems-glofas-reforecast",
            "request": {
                "variable": "river_discharge_in_the_last_24_hours",
                "format": "grib",
                "product_type": [
                    "control_reforecast",
                    "ensemble_perturbed_reforecasts",
                ],
                "hyear": f"{self.year}",
                "hmonth": self.expected_months,
                "hday": self.expected_days,
                "area": self.expected_area,
                "system_version": "version_3_1",
                "hydrological_model": "lisflood",
                "leadtime_hour": self.expected_leadtime,
            },
            "target": Path(
                f"{TMP_PATH}/public/raw/{self.country_iso3}"
                f"/glofas/version_3/cems-glofas-reforecast"
                f"/{self.country_iso3}_cems-glofas-reforecast_v3_2000.grib"
            ),
        }

    def test_reforecast_download(self, fake_mkdir, fake_retrieve):
        glofas_reforecast = glofas.GlofasReforecast()
        glofas_reforecast.download(
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=self.leadtimes,
            year_min=self.year,
            year_max=self.year,
        )
        fake_retrieve.assert_called_with(**self.get_reforecast_expected_args())

    def test_reforecast_download_split_by_leadtime(
        self, fake_mkdir, fake_retrieve
    ):
        glofas_reforecast = glofas.GlofasReforecast()
        glofas_reforecast.download(
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=self.leadtimes[:1],
            year_min=self.year,
            year_max=self.year,
            split_by_leadtimes=True,
        )
        expected_args = self.get_reforecast_expected_args()
        expected_args["request"]["leadtime_hour"] = self.expected_leadtime[:1]
        expected_args["target"] = Path(
            f"{TMP_PATH}/public/raw/{self.country_iso3}"
            f"/glofas/version_3/cems-glofas-reforecast"
            f"/{self.country_iso3}_cems-glofas-reforecast_v3_2000_lt10d.grib"
        )
        fake_retrieve.assert_called_with(**expected_args)


@mock.patch("src.indicators.flooding.glofas.glofas.xr.open_mfdataset")
@mock.patch.object(glofas, "DATA_DIR", TMP_PATH)
class TestProcess:
    def setup(self):
        self.country_iso3 = "abc"
        self.station_name = "fake_station"
        self.stations = {self.station_name: Station(lon=5.05, lat=10.05)}
        self.year = 2000
        self.leadtimes = [10, 20]
        self.numbers = [0, 1, 2, 3, 4, 5, 6]

    @staticmethod
    def teardown():
        shutil.rmtree(TMP_PATH)

    @staticmethod
    def get_raw_data(
        number_coord: [list, int] = None,
        include_step: bool = False,
        include_history: bool = False,
        dis24: np.ndarray = None,
    ):
        rng = np.random.default_rng(12345)
        coords = {}
        if number_coord is not None:
            coords["number"] = number_coord
        coords["time"] = pd.date_range("2014-09-06", periods=2)
        if include_step:
            coords["step"] = [np.datetime64(n + 1, "D") for n in range(5)]
        coords["latitude"] = [10.15, 10.05, 9.95]
        coords["longitude"] = [4.95, 5.05, 5.25, 5.35]
        dims = list(coords.keys())
        if number_coord is not None and isinstance(number_coord, int):
            dims = dims[1:]
        if dis24 is None:
            dis24 = 5000 + 100 * rng.random([len(coords[dim]) for dim in dims])
        attrs = {}
        if include_history:
            attrs = {"history": "fake history"}
        return xr.Dataset({"dis24": (dims, dis24)}, coords=coords, attrs=attrs)

    def get_processed_data(
        self,
        number_coord: [list, int] = None,
        include_step: bool = False,
        dis24: np.ndarray = None,
    ):
        raw_data = TestProcess.get_raw_data(
            number_coord=number_coord, include_step=include_step, dis24=dis24
        )
        station = self.stations[self.station_name]
        dis24 = raw_data["dis24"].sel(
            longitude=station.lon, latitude=station.lat, method="nearest"
        )
        coords = {}
        if number_coord is not None:
            coords = {"number": number_coord}
        coords["time"] = raw_data.time
        if include_step:
            coords["step"] = raw_data.step
        return xr.Dataset(
            {self.station_name: (list(coords.keys()), dis24)}, coords=coords
        )

    def get_enxemble_raw(self):
        cf_raw = TestProcess.get_raw_data(
            number_coord=self.numbers[0],
            include_step=True,
            include_history=True,
        )
        pf_raw = TestProcess.get_raw_data(
            number_coord=self.numbers[1:],
            include_step=True,
            include_history=True,
        )
        expected_dis24 = np.concatenate(
            (cf_raw["dis24"].values[np.newaxis, ...], pf_raw["dis24"].values)
        )
        return cf_raw, pf_raw, expected_dis24

    def test_reanalysis_process(self, fake_open_mfdataset):
        fake_open_mfdataset.return_value = self.get_raw_data()
        glofas_reanalysis = glofas.GlofasReanalysis()
        output_filepath = glofas_reanalysis.process(
            country_iso3=self.country_iso3,
            stations=self.stations,
            year_min=self.year,
            year_max=self.year,
        )
        output_ds = xr.load_dataset(output_filepath)
        assert output_ds.equals(self.get_processed_data())

    def test_reforecast_process(self, fake_open_mfdataset):
        cf_raw, pf_raw, expected_dis24 = self.get_enxemble_raw()
        fake_open_mfdataset.side_effect = [cf_raw, pf_raw]
        glofas_reforecast = glofas.GlofasReforecast()
        output_filepath = glofas_reforecast.process(
            country_iso3=self.country_iso3,
            stations=self.stations,
            leadtimes=self.leadtimes,
            year_min=self.year,
            year_max=self.year,
        )
        output_ds = xr.load_dataset(output_filepath)
        assert output_ds.equals(
            self.get_processed_data(
                number_coord=self.numbers,
                include_step=True,
                dis24=expected_dis24,
            )
        )

    def test_forecast_process(self, fake_open_mfdataset):
        cf_raw, pf_raw, expected_dis24 = self.get_enxemble_raw()
        fake_open_mfdataset.side_effect = [cf_raw, pf_raw]
        glofas_forecast = glofas.GlofasReforecast()
        output_filepath = glofas_forecast.process(
            country_iso3=self.country_iso3,
            stations=self.stations,
            leadtimes=self.leadtimes,
            year_min=self.year,
            year_max=self.year,
        )
        output_ds = xr.load_dataset(output_filepath)
        assert output_ds.equals(
            self.get_processed_data(
                number_coord=self.numbers,
                include_step=True,
                dis24=expected_dis24,
            )
        )
