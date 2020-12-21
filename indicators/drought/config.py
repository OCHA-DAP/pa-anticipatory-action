from datetime import datetime,timedelta
from pathlib import Path
import os
import sys
path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
#cannot have an utils file inside the drought directory, and import from the utils directory.
#So renamed to utils_general but this should of course be changed
from utils_general.utils import parse_yaml


class Config:
    ### general directories
    ANALYSES_DIR="analyses"
    DATA_DIR = "Data"

    def __init__(self):
        #get the absolute path to the root directory, i.e. pa-anticipatory-action
        DIR_PATH= getattr(
            self, "DIR_PATH", Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
        )
        self.DIR_PATH = DIR_PATH
        self.DROUGHTDATA_DIR= os.path.join(DIR_PATH,'indicators','drought','Data')
        self._parameters = None


    def parameters(self, country):
        if self._parameters is None:
            # self._parameters = parse_yaml(os.path.join(self.DIR_PATH,"indicators","food_insecurity","config.yml"))[country_iso3] #
            self._parameters = parse_yaml(os.path.join(self.DIR_PATH, self.ANALYSES_DIR, country.lower(), 'config.yml'))
        return self._parameters



    #General date objects
    #TODO: decide if want to use this for IRI
    #Might also just want to download separate file for every month, since that is the structure of the other forecasts
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.strftime("%b")
    NEXT_YEAR = TODAY.year+1

    ### Shapefiles
    #country specific shapefiles
    SHAPEFILE_DIR = 'Shapefiles'

    #TODO: check if this is a good world boundaries file and if there is any copyright or so to it
    #world shapefile so used by all countries, save in drought folder
    WORLD_SHP_PATH=os.path.join('indicators','drought','Data','TM_WORLD_BORDERS-0','TM_WORLD_BORDERS-0.3.shp')

    ### Name mappings
    #to rename the variables of different providers to
    LOWERTERCILE = "prob_below"
    LONGITUDE = "lon"
    LATITUDE = "lat"

    ### IRI
    # currently downloading from november 2020 till latest date available
    # and only the belowaverage data
    # this can be changed but if downloading all historical data, it can take about half an hour
    #TODO: decide if want one file per month or one with general name that contains newest data
    # IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Oct%202020)/(Jan%20{NEXT_YEAR})/RANGEEDGES/data.nc"
    #TODO: change URL back to above to include newest data. Now using November for debugging
    # IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Oct%202020)/(Nov%202020)/RANGEEDGES/data.nc"
    IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Mar%202017)/(Jan%20{NEXT_YEAR})/RANGEEDGES/data.nc"
    # IRI_URL = f"https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob/F/(Jan%202020)/(Jan%20{NEXT_YEAR})/RANGEEDGES/C/(Below_Normal)/RANGEEDGES/data.nc"

    IRI_DIR = "IRI"
    #currently it is solely downloading belowaverage data, this can be changed but will make downloading times longer.
    IRI_NC_FILENAME_RAW = f"IRI_2020{NEXT_YEAR}.nc"
    IRI_NC_FILENAME_CRS = f"IRI_2020{NEXT_YEAR}_crs.nc"
    IRI_LOWERTERCILE = "prob"


    ### ICPAC
    #TODO: probably want to download directly from the ftp server instead of uploading to GDrive and downloading from there. But couldn't figure out how to do that
    ICPAC_GDRIVE_ZIPID = "13VQTVj5Lwm6jHYBMIf7oUZt5k1-alwjf"
    ICPAC_DIR = "icpac"
    #Raw data can be processed for all dates, for the one with crs, we only want to one of interest
    ICPAC_PROBFORECAST_REGEX_RAW = "ForecastProb*.nc"
    ICPAC_PROBFORECAST_REGEX_CRS = "ForecastProb*{month}{year}_crs.nc"
    ICPAC_LOWERTERCILE = "below"
    ICPAC_LON = "x"
    ICPAC_LAT = "y"

    #NMME
    #date should be string of YYYYMM
    #TODO: in future might want to add support for monthly (and temperature) forecasts
    NMME_FTP_URL_SEASONAL = "ftp://ftp.cpc.ncep.noaa.gov/NMME/prob//netcdf/prate.{date}.prob.adj.seas.nc"
    NMME_DIR = "nmme"
    NMME_NC_FILENAME_RAW = "nmme_prate_{date}_prob_adj_seas.nc"
    NMME_NC_FILENAME_CRS = "nmme_prate_{date}_prob_adj_seas_crs.nc"
    NMME_LOWERTERCILE = "prob_below"
    NMME_LON = "x"
    NMME_LAT = "y"

