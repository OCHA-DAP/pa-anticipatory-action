import os
from pathlib import Path
from utils import parse_yaml

class Config:
    def __init__(self):
        #get the absolute path to the root directory, i.e. pa-anticipatory-action
        self.DIR_PATH = getattr(
            self, "DIR_PATH", Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
        )
        self._parameters = None


    def parameters(self, country_iso3):
        if self._parameters is None:
            self._parameters = parse_yaml(os.path.join(self.DIR_PATH,"indicators","food_insecurity","config.yml"))[country_iso3] #os.path.join(self.CONFIG_DIR, f'{country_iso3.lower()}.yml'))
        return self._parameters

    #### FewsNet
    FEWSNET_DIR = "FewsNetRaw"
    #region can either be a part of a continent (e.g. east-africa) and a country (e.g. ethiopia)
    FEWSNET_FILENAME = "{region}{date}/{regionabb}_{date}_{period}.shp"

    #### Worldpop
    WORLDPOP_DIR = "WorldPop"
    # can make this more variable with a dict, e.g. if we want 1km and 100m or if we also want not UNadj
    # we are currently using 1km because this is generally granular enough and speeds up the calculations a lot
    WORLDPOP_FILENAME = "{country_iso3}_ppp_{year}_1km_Aggregated_UNadj.tif"
    WORLDPOP_URL="ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{country_iso3_upper}/{country_iso3_lower}_ppp_{year}_1km_Aggregated_UNadj.tif"

    #### Global IPC
    GLOBALIPC_DIR = "GlobalIPC"
    GLOBALIPC_URL="http://mapipcissprd.us-east-1.elasticbeanstalk.com/api/public/population-tracking-tool/data/{min_year},{max_year}/?export=true&condition=A&country={country_iso2}"
    GLOBALIPC_FILENAME="{country_iso3}_globalipc_raw.csv"
    #Analysis name, Country Population, % of total county Pop, Area Phase are not being used in our current analysis so not mapping them
    # Not entirely sure if Area always equals Admin2 regions
    GLOBALIPC_COLUMNNAME_MAPPING = {'Country':'ADMIN0','Level 1 Name':'ADMIN1','Area':'ADMIN2','Area ID':'ADMIN2_ID','Date of Analysis':'date','#':'pop_CS','Analysis Period':"CS_val",'#.1':'CS_1','%':'perc_CS_1','#.2':'CS_2', '%.1':'perc_CS_2', '#.3':'CS_3', '%.2':'perc_CS_3',
       '#.4':'CS_4', '%.3':'perc_CS_4', '#.5':'CS_5','%.4':'perc_CS_5', '#.6':'CS_3p', '%.5':'perc_CS_3p', '#.7':'pop_ML1','Analysis Period.1':'ML1_val', '#.8':'ML1_1',
       '%.6':'perc_ML1_1', '#.9':'ML1_2', '%.7':'perc_ML1_2', '#.10':'ML1_3', '%.8':'perc_ML1_3', '#.11':'ML1_4', '%.9':'perc_ML1_4', '#.12':'ML1_5', '%.10':'perc_ML1_5',
       '#.13':'ML1_3p', '%.11':'perc_ML1_3p','#.14':'pop_ML2','Analysis Period.2':'ML2_val','#.15':"ML2_1", '%.12':'perc_ML2_1', '#.16':'ML2_2', '%.13':'perc_ML2_2', '#.17':'ML2_3', '%.14':'perc_ML2_3',
       '#.18':'ML2_4', '%.15':'perc_ML2_4', '#.19':'ML2_5', '%.16':'perc_ML2_5', '#.20':'ML2_3p', '%.17':'perc_ML2_3p'}

