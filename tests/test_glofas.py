import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import xarray as xr

from src.indicators.flooding.glofas import glofas
from src.indicators.flooding.glofas.area import Area, Station


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
        ds=ds, dataset_name="var_a", coord_names=["z", "x", "y"], expansion_dim=0
    )
    assert "z" in ds.dims.keys()


@mock.patch("src.indicators.flooding.glofas.glofas.cdsapi.Client.retrieve")
@mock.patch("src.indicators.flooding.glofas.glofas.Path.mkdir")
@mock.patch.object(glofas, "DATA_DIR", Path("/tmp"))
class TestDownload(unittest.TestCase):
    def setUp(self):
        self.country_name = "fake_country"
        self.country_iso3 = "abc"
        self.area = Area(north=1, south=-2, east=3, west=-4)
        self.year = 2000
        self.leadtime = 10
        self.expected_area = [1, -4, -2, 3]
        self.expected_months = [str(x + 1).zfill(2) for x in range(12)]
        self.expected_days = [str(x + 1).zfill(2) for x in range(31)]
        self.expected_leadtime = 240

    def test_reanalysis_download(self, fake_mkdir, fake_retrieve):
        glofas_reanalysis = glofas.GlofasReanalysis()
        glofas_reanalysis.download(
            country_name=self.country_name,
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
                f"/tmp/raw/{self.country_name}/GLOFAS_Data/version_3/cems-glofas-historical"
                f"/{self.country_iso3}_cems-glofas-historical_v3_2000.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)

    def test_forecast_download(self, fake_mkdir, fake_retrieve):
        glofas_forecast = glofas.GlofasForecast()
        glofas_forecast.download(
            country_name=self.country_name,
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=[self.leadtime],
            year_min=self.year,
            year_max=self.year,
        )
        expected_args = {
            "name": "cems-glofas-forecast",
            "request": {
                "variable": "river_discharge_in_the_last_24_hours",
                "format": "grib",
                "product_type": ["control_forecast", "ensemble_perturbed_forecasts"],
                "year": f"{self.year}",
                "month": self.expected_months,
                "day": self.expected_days,
                "area": self.expected_area,
                "system_version": "version_3_1",
                "hydrological_model": "lisflood",
                "leadtime_hour": f"{self.expected_leadtime}",
            },
            "target": Path(
                f"/tmp/raw/{self.country_name}/GLOFAS_Data/version_3/cems-glofas-forecast"
                f"/{self.country_iso3}_cems-glofas-forecast_v3_2000_lt10d.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)

    def test_reforecast_download(self, fake_mkdir, fake_retrieve):
        glofas_reforecast = glofas.GlofasReforecast()
        glofas_reforecast.download(
            country_name=self.country_name,
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=[self.leadtime],
            year_min=self.year,
            year_max=self.year,
        )
        expected_args = {
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
                "leadtime_hour": f"{self.expected_leadtime}",
            },
            "target": Path(
                f"/tmp/raw/{self.country_name}/GLOFAS_Data/version_3/cems-glofas-reforecast"
                f"/{self.country_iso3}_cems-glofas-reforecast_v3_2000_lt10d.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)


@mock.patch("src.indicators.flooding.glofas.glofas.xr.open_mfdataset")
@mock.patch("src.indicators.flooding.glofas.glofas.Path.mkdir")
@mock.patch.object(glofas, "DATA_DIR", Path("/tmp"))
class TestProcess(unittest.TestCase):
    def setUp(self):
        self.country_name = "fake_country"
        self.country_iso3 = "abc"
        self.stations = {'fake_station': Station(lon=1.0, lat=2.0)}
        self.raw_data = xr.Dataset(
        )

    def test_reanalysis_process(self, fake_mkdir, fake_open_mfdataset):
        glofas_reanalysis = glofas.GlofasReanalysis()
        glofas_reanalysis.year_min, glofas_reanalysis.year_max = (2000, 2001)
        glofas_reanalysis.process(
            country_name=self.country_name,
            country_iso3=self.country_iso3,
            stations=self.stations
        )
        fake_open_mfdataset.assert_called_with(x=1)
