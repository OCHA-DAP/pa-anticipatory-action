from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
# cannot have an utils file inside the drought directory, and import from the utils directory.
# So renamed to utils_general but this should of course be changed
from src.utils_general.utils import parse_yaml


class Config:
    ### general directories
    RAW_DIR = "raw"
    PROCESSED_DIR = "processed"
    DATA_DIR = Path(os.environ["AA_DATA_DIR"]) / "public"
    DATA_PRIVATE_DIR = Path(os.environ["AA_DATA_DIR"]) / "private"
    ANALYSES_DIR = "analyses"

    def __init__(self):
        # get the absolute path to the root directory, i.e. pa-anticipatory-action
        DIR_PATH = getattr(
            self,
            "DIR_PATH",
            Path(os.path.dirname(os.path.realpath(__file__))).parents[1],
        )
        self.DIR_PATH = DIR_PATH
        self.FLOODDATA_DIR = os.path.join(self.DATA_DIR, self.RAW_DIR, "flood")
        self._parameters = None

    def parameters(self, country):
        if self._parameters is None:
            # self._parameters = parse_yaml(os.path.join(self.DIR_PATH,"indicators","food_insecurity","config.yml"))[country_iso3] #
            self._parameters = parse_yaml(
                os.path.join(self.DIR_PATH, country.lower(), "config.yml")
            )
        return self._parameters

    # General date objects
    # Might also just want to download separate file for every month, since that is the structure of the other forecasts
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.strftime("%b")
    TODAY_YEAR = TODAY.year
    NEXT_YEAR = TODAY_YEAR + 1

    ### Shapefiles
    # country specific shapefiles
    SHAPEFILE_DIR = "Shapefiles"

    # TODO: check if this is a good world boundaries file and if there is any copyright or so to it
    # world shapefile so used by all countries, save in drought folder
    WORLD_SHP_PATH = os.path.join(
        "raw", "drought", "TM_WORLD_BORDERS-0", "TM_WORLD_BORDERS-0.3.shp"
    )

    ### Name mappings
    # to rename the variables of different providers to
    LOWERTERCILE = "prob_below"
    LONGITUDE = "lon"
    LATITUDE = "lat"

    ### GloFAS parameters
    # TODO: Identify GloFAS params that are country-independent

