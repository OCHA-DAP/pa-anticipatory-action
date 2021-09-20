import unittest
from unittest import mock
from pathlib import Path

from src.indicators.drought.ecmwf_seasonal import ecmwf_seasonal
from src.utils_general.area import Area


@mock.patch(
    "src.indicators.drought.ecmwf_seasonal."
    "ecmwf_seasonal.cdsapi.Client.retrieve"
)
@mock.patch("src.indicators.drought.ecmwf_seasonal.ecmwf_seasonal.Path.mkdir")
@mock.patch.object(ecmwf_seasonal, "DATA_DIR", Path("/tmp"))
class TestDownload(unittest.TestCase):
    def setUp(self):
        self.country_iso3 = "abc"
        self.area = Area(north=1.3, south=-2.02, east=3.55, west=-4)
        self.year = 2000
        self.months = [3]
        self.leadtime = [2, 4]
        self.expected_area = [2.0, -4.0, -3.0, 4.0]

    def test_forecast_download(self, fake_mkdir, fake_retrieve):
        ecmwf_forecast = ecmwf_seasonal.EcmwfSeasonalForecast()
        ecmwf_forecast.download(
            country_iso3=self.country_iso3,
            area=self.area,
            leadtimes=self.leadtime,
            months=self.months,
            year_min=self.year,
            year_max=self.year,
        )
        expected_args = {
            "name": "seasonal-monthly-single-levels",
            "request": {
                "variable": "total_precipitation",
                "originating_centre": "ecmwf",
                "system": 5,
                "format": "grib",
                "product_type": [
                    "monthly_mean",
                ],
                "year": f"{self.year}",
                "month": "03",
                "leadtime_month": ["2", "4"],
                "area": self.expected_area,
            },
            "target": Path(
                f"/tmp/public/raw/{self.country_iso3}"
                f"/ecmwf/seasonal-monthly-single-levels"
                f"/{self.country_iso3}_seasonal-monthly-single-levels"
                f"_v5_2000-03.grib"
            ),
        }
        fake_retrieve.assert_called_with(**expected_args)

    # def test_reforecast_download_split_by_leadtime(
    #     self, fake_mkdir, fake_retrieve
    # ):
    #     glofas_reforecast = glofas.GlofasReforecast()
    #     glofas_reforecast.download(
    #         country_iso3=self.country_iso3,
    #         area=self.area,
    #         leadtimes=self.leadtime[:1],
    #         year_min=self.year,
    #         year_max=self.year,
    #         split_by_leadtimes=True,
    #     )
    #     expected_args = self.get_reforecast_expected_args()
    #     expected_args["request"]["leadtime_hour"] =
    #     self.expected_leadtime[:1]
    #     expected_args["target"] = Path(
    #         f"/tmp/public/raw/{self.country_iso3}"
    #         f"/glofas/version_3/cems-glofas-reforecast"
    #         f"/{self.country_iso3}_cems-glofas-reforecast_v3_2000_lt10d.grib"
    #     )
    #     fake_retrieve.assert_called_with(**expected_args)
