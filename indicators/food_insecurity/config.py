import os
from pathlib import Path
from utils_general.utils import parse_yaml
from datetime import datetime
import ftplib
import re


def retrieve_worldpop_dirname():
    """
    Retrieve the name of the directory with the data of interest on Worldpop's ftp server
    This has to be done dynamically, since the folder includes a year in its name and we don't know when this year is changed
    Returns:
        item (str): the path to the folder with the Global population data on 1km resolution and UN adjusted numbers
    """
    site = "ftp.worldpop.org.uk"
    username = "anonymous"
    password = None
    ftp = ftplib.FTP(site, username, password)
    for item in ftp.nlst("GIS/Population"):
        if re.search(r"Global_.*_1km_UNadj", item):
            return item

class Config:
    ### general directories
    ANALYSES_DIR = "analyses"
    DATA_DIR = "Data"
    def __init__(self):
        #get the absolute path to the root directory, i.e. pa-anticipatory-action
        DIR_PATH = getattr(
            self, "DIR_PATH", Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
        )
        self.DIR_PATH = DIR_PATH
        self.FOODINSECURITYDATA_DIR = os.path.join(DIR_PATH, 'indicators', 'food_insecurity', 'Data')
        self._parameters = None


    def parameters(self, country):
        if self._parameters is None:
            self._parameters = parse_yaml(os.path.join(self.DIR_PATH, self.ANALYSES_DIR, country.lower(), 'config.yml'))
        return self._parameters

    #General date objects
    TODAY = datetime.now()
    TODAY_YEAR = TODAY.strftime("%Y")

    ### Shapefiles
    SHAPEFILE_DIR = 'Shapefiles'

    ### General values
    IPC_PERIOD_NAMES = ["CS", "ML1", "ML2"]
    ADMIN0_COL = "ADMIN0"
    ADMIN1_COL = "ADMIN1"
    ADMIN2_COL = "ADMIN2"

    #### FewsNet
    FEWSNET_RAW_DIR = "FewsNetRaw"
    #region can either be a part of a continent (e.g. east-africa) and a country (e.g. ethiopia)
    FEWSNET_FILENAME = "{region}{date}/{regionabb}_{date}_{period}.shp"

    #these are the standard dates fewsnet should have published data. In 2016 they changed the months of publication
    #in the config per country, dates can be added and removed
    FEWSWORLDPOP_PROCESSED_DIR = "FewsNetWorldPop"
    FEWSWORLDPOP_PROCESSED_FILENAME = "{country}_fewsnet_worldpop_admin{admin_level}{suffix}.csv"
    FEWSADMPOP_PROCESSED_DIR = "FewsNetAdmPop"
    FEWSNET_DATES = ["200907","200910"] + [f"{str(i)}{m}" for i in range(2010,2016) for m in ["01","04","07","10"]] + [f"{str(i)}{m}" for i in range(2016,int(TODAY_YEAR)+1) for m in ["02","06","10"]]



    #### Worldpop
    WORLDPOP_RAW_DIR = "WorldPop"
    # can make this more variable with a dict, e.g. if we want 1km and 100m or if we also want not UNadj
    # we are currently using 1km because this is generally granular enough and speeds up the calculations a lot
    WORLDPOP_FILENAME = "{country_iso3}_ppp_{year}_1km_Aggregated_UNadj.tif"
    #this dirname changes with the year, so dynamically retrieve it by using a regex
    WORLDPOP_FTP_DIRNAME = retrieve_worldpop_dirname()
    WORLDPOP_BASEURL=f"ftp://ftp.worldpop.org.uk/{WORLDPOP_FTP_DIRNAME}/"
    WORLDPOP_URL=WORLDPOP_BASEURL+"{year}/{country_iso3_upper}/{country_iso3_lower}_ppp_{year}_1km_Aggregated_UNadj.tif"

    # WORLDPOP_URL="ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{country_iso3_upper}/{country_iso3_lower}_ppp_{year}_1km_Aggregated_UNadj.tif"

    #### Worldbank historical national population
    WB_POP_FILENAME = "Worldbank_TotalPopulation.csv"

    #### Global IPC
    GLOBALIPC_RAW_DIR = "GlobalIPC"
    GLOBALIPC_PROCESSED_DIR = "GlobalIPCProcessed"
    GLOBALIPC_URL="http://mapipcissprd.us-east-1.elasticbeanstalk.com/api/public/population-tracking-tool/data/{min_year},{max_year}/?export=true&condition=A&country={country_iso2}"
    GLOBALIPC_FILENAME_RAW="{country}_globalipc_raw.xlsx"
    GLOBALIPC_FILENAME_NEWCOLNAMES="{country}_globalipc_newcolumnnames.xlsx"
    GLOBALIPC_FILENAME_PROCESSED="{country}_globalipc_admin{admin_level}{suffix}.csv"
    #Analysis name, Country Population, % of total county Pop, Area Phase are not being used in our current analysis so not mapping them
    # Not entirely sure if Area always equals Admin2 regions
    GLOBALIPC_COLUMNNAME_MAPPING = {'Country':'ADMIN0','Level 1 Name':'ADMIN1','Area':'ADMIN2','Area ID':'ADMIN2_ID','Date of Analysis':'date','#':'pop_CS','Analysis Period':"CS_val",'#.1':'CS_1','%':'perc_CS_1','#.2':'CS_2', '%.1':'perc_CS_2', '#.3':'CS_3', '%.2':'perc_CS_3',
       '#.4':'CS_4', '%.3':'perc_CS_4', '#.5':'CS_5','%.4':'perc_CS_5', '#.6':'CS_3p', '%.5':'perc_CS_3p', '#.7':'pop_ML1','Analysis Period.1':'ML1_val', '#.8':'ML1_1',
       '%.6':'perc_ML1_1', '#.9':'ML1_2', '%.7':'perc_ML1_2', '#.10':'ML1_3', '%.8':'perc_ML1_3', '#.11':'ML1_4', '%.9':'perc_ML1_4', '#.12':'ML1_5', '%.10':'perc_ML1_5',
       '#.13':'ML1_3p', '%.11':'perc_ML1_3p','#.14':'pop_ML2','Analysis Period.2':'ML2_val','#.15':"ML2_1", '%.12':'perc_ML2_1', '#.16':'ML2_2', '%.13':'perc_ML2_2', '#.17':'ML2_3', '%.14':'perc_ML2_3',
       '#.18':'ML2_4', '%.15':'perc_ML2_4', '#.19':'ML2_5', '%.16':'perc_ML2_5', '#.20':'ML2_3p', '%.17':'perc_ML2_3p'}

