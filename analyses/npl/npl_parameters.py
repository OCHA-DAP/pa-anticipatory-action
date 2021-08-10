import os
import sys
from pathlib import Path

import matplotlib as mpl

# Allow import of items from src
path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[0]}/"
sys.path.append(path_mod)

# Setting for the figure sizes
mpl.rcParams["figure.dpi"] = 200

# Main directories
DATA_DIR = Path(os.environ["AA_DATA_DIR"])
DATA_DIR_PUBLIC = DATA_DIR / "public"
DATA_DIR_PRIVATE = DATA_DIR / "private"

# Commonly used subdirectories
RCO_DIR = DATA_DIR_PRIVATE / "exploration" / "npl" / "unrco"
GLOFAS_DIR = DATA_DIR_PUBLIC / "exploration" / "npl" / "glofas"

# Commonly used files
GLOFAS_RP_FILENAME = GLOFAS_DIR / "glofas_return_period_values.xlsx"

# Shapefile things
SHAPEFILE_DIR = DATA_DIR_PUBLIC / "raw" / "npl" / "cod_ab"
ADMIN_SHAPEFILE = (
    SHAPEFILE_DIR
    / "npl_admbnda_ocha_20201117"
    / "npl_admbnda_nd_20201117_shp.zip"
)
ADMIN_DISTRICTS_SHAPEFILE = "npl_admbnda_districts_nd_20201117.shp"
ADMIN2_SHAPEFILE = "npl_admbnda_adm2_nd_20201117.shp"

# GloFAS settings
COUNTRY_ISO3 = "npl"
LEADTIMES = [x + 1 for x in range(10)]
DURATION = 1  # How many consecutive days the event needs to occur for
MAIN_RP = 2
MAIN_FORECAST_PROB = 50
DAYS_BEFORE_BUFFER = 3  # When not using forecast leadtime
DAYS_AFTER_BUFFER = 30
# Use "_v3" for the GloFAS model v3 locs, or empty string for the original v2
VERSION_LOC = "_v3"
USE_INCORRECT_AREA_COORDS = False

FINAL_STATIONS = ["Chatara", "Chisapani"]
STATIONS_BY_BASIN = {
    "Koshi": ["Chatara_v3", "Simle_v3", "Majhitar_v3", "Kampughat_v3"],
    "Karnali": [
        "Chisapani_v3",
        "Asaraghat_v3",
        "Dipayal_v3",
        "Samajhighat_v3",
    ],
    "West Rapti": ["Kusum_v3"],
    "Bagmati": ["Rai_goan_v3"],
    "Babai": ["Chepang_v3"],
}
STATIONS_BY_MAJOR_BASIN = {
    "Koshi": [
        "Chatara_v3",
        "Simle_v3",
        "Majhitar_v3",
        "Kampughat_v3",
        "Rai_goan_v3",
    ],
    "Karnali": [
        "Chisapani_v3",
        "Asaraghat_v3",
        "Dipayal_v3",
        "Samajhighat_v3",
        "Kusum_v3",
        "Chepang_v3",
    ],
}

# DHM water level
LEVEL_TYPES = ["warning", "danger"]
DHM_DIR = DATA_DIR_PRIVATE / "exploration" / "npl" / "dhm"
DHM_STATION_INFO_FILENAME = DHM_DIR / "npl_dhm_station_info.xlsx"
WL_RAW_DIR = DHM_DIR / "raw" / "water_level"
WL_INPUT_FILENAME = "GHT_{}.txt"
WL_PROCESSED_DIR = DHM_DIR / "processed"
WL_OUTPUT_FILENAME = WL_PROCESSED_DIR / "waterl_level_procssed.csv"