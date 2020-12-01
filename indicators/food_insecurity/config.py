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