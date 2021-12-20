import os
import sys
from datetime import datetime
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
# cannot have an utils file inside the drought directory, and import
# from the utils directory. So renamed to utils_general but this should
# of course be changed
from src.utils_general.utils import parse_yaml


class Config:
    # general directories
    RAW_DIR = "raw"
    PROCESSED_DIR = "processed"
    PUBLIC_DIR = "public"
    PRIVATE_DIR = "private"
    DATA_DIR = os.path.join(os.environ["AA_DATA_DIR"])
    GLOBAL_ISO3 = "glb"
    PLOT_DIR = "plots"

    def __init__(self):
        # get the absolute path to the root directory, i.e.
        # pa-anticipatory-action
        DIR_PATH = getattr(
            self,
            "DIR_PATH",
            Path(os.path.dirname(os.path.realpath(__file__))).parents[1],
        )
        self.DIR_PATH = DIR_PATH
        self.GLOBAL_DIR = os.path.join(
            self.DATA_DIR, self.PUBLIC_DIR, self.RAW_DIR, self.GLOBAL_ISO3
        )
        self._parameters = None

    def parameters(self, country):
        if self._parameters is None:
            self._parameters = parse_yaml(
                os.path.join(self.DIR_PATH, country.lower(), "config.yml")
            )
        return self._parameters

    # repo paths
    ANALYSES_DIR = "analyses"

    # General date objects Might also just want to download separate
    # file for every month, since that is the structure of the other
    # forecasts
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.strftime("%b")
    TODAY_YEAR = TODAY.year
    NEXT_YEAR = TODAY_YEAR + 1

    # Shapefiles
    # country specific shapefiles
    SHAPEFILE_DIR = "cod_ab"

    # TODO: check if this is a good world boundaries file and if there
    # is any copyright or so to it world shapefile so used by all
    # countries, save in drought folder
    WORLD_SHP_PATH = os.path.join(
        "private",
        "raw",
        "glb",
        "cod_ab",
        "TM_WORLD_BORDERS-0",
        "TM_WORLD_BORDERS-0.3.shp",
    )

    # Name mappings
    # to rename the variables of different providers to
    # set to x/y or longitude/latitude for rioxarray to recognize
    LOWERTERCILE = "prob_below"
    LONGITUDE = "longitude"
    LATITUDE = "latitude"

    # IRI
    # currently downloading from 2017 till the latest available forecast
    # set enddate to Jan next year, to make sure the latest date is
    # always included in the download If download takes too long, you
    # can change the url to only download from a shorter time period
    IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Mar%202017)/(Jan%20{NEXT_YEAR})/RANGEEDGES/data.nc"  # noqa: E501
    IRI_DIR = "iri"
    # downloads data from 2017 till the latest date available for the
    # current year
    IRI_NC_FILENAME_RAW = f"IRI_2017{TODAY_YEAR}.nc"
    IRI_NC_FILENAME_CLEAN = f"IRI_2017{TODAY_YEAR}_clean.nc"
    IRI_LOWERTERCILE = "prob"
    IRI_LON = "X"
    IRI_LAT = "Y"
    IRI_CRS = "EPSG:4326"

    # ICPAC
    ICPAC_GDRIVE_ZIPID = "13VQTVj5Lwm6jHYBMIf7oUZt5k1-alwjf"
    ICPAC_DIR = "icpac"
    # Raw data can be processed for all dates, for the one with crs, we
    # only want to one of interest
    ICPAC_PROBFORECAST_REGEX_RAW = "ForecastProb*.nc"
    # The raw ICPAC data consists of 3 bands: below normal, average, and
    # above normal when adding the crs, only one band can be saved
    # correctly, hence add to the name the tercile we are saving
    ICPAC_PROBFORECAST_REGEX_CRS = (
        "ForecastProb*{month}{year}_{tercile}_crs.nc"
    )
    ICPAC_LOWERTERCILE = "below"
    ICPAC_LON = "x"
    ICPAC_LAT = "y"

    # NMME
    # date should be string of YYYYMM TODO: in future might want to add
    # support for monthly (and temperature) forecasts
    NMME_FTP_URL_SEASONAL = "ftp://ftp.cpc.ncep.noaa.gov/NMME/prob//netcdf/prate.{date}.prob.adj.seas.nc"  # noqa: E501
    NMME_DIR = "nmme"
    NMME_NC_FILENAME_RAW = "nmme_prate_{date}_prob_adj_seas.nc"
    NMME_NC_FILENAME_CRS = "nmme_prate_{date}_prob_adj_seas_{tercile}_crs.nc"
    NMME_LOWERTERCILE = "prob_below"
    NMME_LON = "x"
    NMME_LAT = "y"

    # CHIRPS
    CHIRPS_DIR = "chirps"
    CHIRPS_MONTHLY_DIR = "monthly"
    CHIRPS_SEASONAL_DIR = "seasonal"
    # resolution can be 25 or 5
    CHIRPS_FTP_URL_GLOBAL_DAILY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p{resolution}/chirps-v2.0.{year}.days_p{resolution}.nc"  # noqa: E501
    CHIRPS_FTP_URL_AFRICA_DAILY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p{resolution}/{year}/chirps-v2.0.{year}.{month}{day}.tif"  # noqa: E501
    CHIRPS_FTP_URL_GLOBAL_MONTHLY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"  # noqa: E501
    CHIRPS_NC_FILENAME_RAW = "chirps_global_daily_{year}_p{resolution}.nc"
    CHIRPS_NC_FILENAME_CRS = "chirps_global_daily_{year}_p{resolution}_crs.nc"
    CHIRPS_MONTHLY_RAW_DIR = (
        Path(DATA_DIR)
        / PUBLIC_DIR
        / RAW_DIR
        / GLOBAL_ISO3
        / CHIRPS_DIR
        / CHIRPS_MONTHLY_DIR
    )
    CHIRPS_MONTHLY_RAW_PATH = (
        CHIRPS_MONTHLY_RAW_DIR / f"chirps_{GLOBAL_ISO3}_monthly.nc"
    )
    CHIRPS_MONTHLY_COUNTRY_FILENAME = "{country_iso3}_chirps_monthly.nc"
    CHIRPS_SEASONAL_LOWERTERCILE_COUNTRY_FILENAME = (
        "{country_iso3}_chirps_seasonal_lowertercile.nc"
    )
    CHIRPS_SEASONAL_TERCILE_BOUNDS_FILENAME = (
        "{country_iso3}_chirps_seasonal_tercile_bounds.nc"
    )
    CHIRPS_LON = "longitude"  # "x" #
    CHIRPS_LAT = "latitude"  # "y" #
    CHIRPS_VARNAME = "precip"

    # CHIRPS-GEFS
    CHIRPSGEFS_DIR = "chirpsgefs"
    CHIRPSGEFS_FTP_URL_AFRICA = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/{days_ahead}day/Africa/precip_mean/data.{year}.{start_day}.tif"  # noqa: E501
    CHIRPSGEFS_RAW_DIR = (
        Path(DATA_DIR) / PUBLIC_DIR / RAW_DIR / GLOBAL_ISO3 / CHIRPSGEFS_DIR
    )
    CHIRPSGEFS_RAW_FILENAME = "chirpsgefs_africa_{days_ahead}days_{date}.tif"
    CHIRPSGEFS_DATE_STR_FORMAT = "%Y%m%d"

    # ECMWF
    ECMWF_DIR = "ecmwf"
    DEFAULT_VERSION = 5

    # DRYSPELLS
    DRY_SPELLS_DIR = "dry_spells"

    # TRIGGER
    TRIGGER_METRICS_DIR = "trigger_metrics"
