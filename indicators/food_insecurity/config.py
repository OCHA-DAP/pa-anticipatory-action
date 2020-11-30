import os
from utils import parse_yaml

class Config:
    def __init__(self):
        self.DIR_PATH = getattr(
            self, "DIR_PATH", os.path.split(os.path.dirname(os.path.realpath(__file__)))
        )[0]
        self._parameters = None


    def parameters(self, country_iso3):
        if self._parameters is None:
            self._parameters = parse_yaml("config.yml")[country_iso3] #os.path.join(self.CONFIG_DIR, f'{country_iso3.lower()}.yml'))
        return self._parameters


    WORLDPOP_DIR = "WorldPop"
    # can make this more variable with a dict, e.g. if we want 1km and 100m or if we also want not UNadj
    # we are currently using 1km because this is generally granular enough and speeds up the calculations a lot
    WORLDPOP_FILENAME = "{country_iso3}_ppp_{year}_1km_Aggregated_UNadj.tif"
    WORLDPOP_URL="ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{country_iso3_upper}/{country_iso3_lower}_ppp_{year}_1km_Aggregated_UNadj.tif"