from datetime import datetime,timedelta
from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
#cannot have an utils file inside the drought directory, and import from the utils directory.
#So renamed to utils_general but this should of course be changed
from src.utils_general.utils import parse_yaml


class Config:
    ### general directories
    RAW_DIR = "raw"
    PROCESSED_DIR = "processed"
    PUBLIC_DIR = "public"
    PRIVATE_DIR = "private"
    DATA_DIR = os.path.join(os.environ["AA_DATA_DIR"])
    GLOBAL_ISO3 = 'glb'
    PLOT_DIR = 'plots'

    def __init__(self):
        #get the absolute path to the root directory, i.e. pa-anticipatory-action
        DIR_PATH= getattr(
            self, "DIR_PATH", Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
        )
        self.DIR_PATH = DIR_PATH
        self.GLOBAL_DIR = os.path.join(self.DATA_DIR, self.PUBLIC_DIR, self.RAW_DIR,  self.GLOBAL_ISO3)
        self._parameters = None


    def parameters(self, country):
        if self._parameters is None:
            # self._parameters = parse_yaml(os.path.join(self.DIR_PATH,"indicators","food_insecurity","config.yml"))[country_iso3] #
            self._parameters = parse_yaml(os.path.join(self.DIR_PATH, country.lower(), 'config.yml'))
        return self._parameters

    ### repo paths
    ANALYSES_DIR = "analyses"

    #General date objects
    #Might also just want to download separate file for every month, since that is the structure of the other forecasts
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.strftime("%b")
    TODAY_YEAR = TODAY.year
    NEXT_YEAR = TODAY_YEAR+1

    ### Shapefiles
    #country specific shapefiles
    SHAPEFILE_DIR = 'cod_ab'

    #TODO: check if this is a good world boundaries file and if there is any copyright or so to it
    #world shapefile so used by all countries, save in drought folder
    WORLD_SHP_PATH=os.path.join('private', 'raw', 'glb', 'cod_ab', 'TM_WORLD_BORDERS-0','TM_WORLD_BORDERS-0.3.shp')

    ### Name mappings
    #to rename the variables of different providers to
    LOWERTERCILE = "prob_below"
    LONGITUDE = "lon"
    LATITUDE = "lat"

    ### IRI
    # currently downloading from 2017 till the latest available forecast
    #set enddate to Jan next year, to make sure the latest date is always included in the download
    #If download takes too long, you can change the url to only download from a shorter time period
    IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Mar%202017)/(Jan%20{NEXT_YEAR})/RANGEEDGES/data.nc"
    IRI_DIR = "iri"
    #downloads data from 2017 till the latest date available for the current year
    IRI_NC_FILENAME_RAW = f"IRI_2017{TODAY_YEAR}.nc"
    IRI_NC_FILENAME_CRS = f"IRI_2017{TODAY_YEAR}_crs.nc"
    IRI_LOWERTERCILE = "prob"
    IRI_LON = "X"
    IRI_LAT = "Y"


    ### ICPAC
    ICPAC_GDRIVE_ZIPID = "13VQTVj5Lwm6jHYBMIf7oUZt5k1-alwjf"
    ICPAC_DIR = "icpac"
    #Raw data can be processed for all dates, for the one with crs, we only want to one of interest
    ICPAC_PROBFORECAST_REGEX_RAW = "ForecastProb*.nc"
    #The raw ICPAC data consists of 3 bands: below normal, average, and above normal
    #when adding the crs, only one band can be saved correctly, hence add to the name the tercile we are saving
    ICPAC_PROBFORECAST_REGEX_CRS = "ForecastProb*{month}{year}_{tercile}_crs.nc"
    ICPAC_LOWERTERCILE = "below"
    ICPAC_LON = "x"
    ICPAC_LAT = "y"

    ### NMME
    #date should be string of YYYYMM
    #TODO: in future might want to add support for monthly (and temperature) forecasts
    NMME_FTP_URL_SEASONAL = "ftp://ftp.cpc.ncep.noaa.gov/NMME/prob//netcdf/prate.{date}.prob.adj.seas.nc"
    NMME_DIR = "nmme"
    NMME_NC_FILENAME_RAW = "nmme_prate_{date}_prob_adj_seas.nc"
    NMME_NC_FILENAME_CRS = "nmme_prate_{date}_prob_adj_seas_{tercile}_crs.nc"
    NMME_LOWERTERCILE = "prob_below"
    NMME_LON = "x"
    NMME_LAT = "y"


    ### CHIRPS
    #resolution can be 25 or 5
    CHIRPS_DIR = "chirps"
    CHIRPS_FTP_URL_GLOBAL_DAILY="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p{resolution}/chirps-v2.0.{year}.days_p{resolution}.nc"
    CHIRPS_FTP_URL_AFRICA_DAILY="https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p{resolution}/{year}/chirps-v2.0.{year}.{month}{day}.tif"
    CHIRPS_NC_FILENAME_RAW = "chirps_global_daily_{year}_p{resolution}.nc"
    CHIRPS_NC_FILENAME_CRS = "chirps_global_daily_{year}_p{resolution}_crs.nc"
    CHIRPS_LON = "longitude" #"x" #
    CHIRPS_LAT = "latitude" #"y" #
    CHIRPS_VARNAME = "precip"
