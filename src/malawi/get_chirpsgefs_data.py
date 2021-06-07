import logging
import pandas as pd

# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.chirpsgefs_rainfallforecast import get_rainy_season_dates, download_chirpsgefs, compute_stats_rainyseason
from src.indicators.drought.config import Config

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

COUNTRY_NAME="malawi"
COUNTRY_ISO3="mwi"
CONFIG = Config()
DATA_PUBLIC_DIR = Path(CONFIG.DATA_DIR) / CONFIG.PUBLIC_DIR
COUNTRY_DATA_PROCESSED_DIR = DATA_PUBLIC_DIR / CONFIG.PROCESSED_DIR / COUNTRY_ISO3
DRY_SPELLS_DIR = "dry_spells"
RAINY_SEASON_PATH = COUNTRY_DATA_PROCESSED_DIR / DRY_SPELLS_DIR / "rainy_seasons_detail_2000_2020_mean_back.csv"
CHIRPSGEFS_PROCESSED_DIR = COUNTRY_DATA_PROCESSED_DIR / CONFIG.CHIRPSGEFS_DIR

#how many days ahead the forecast predicts. Can be 5, 10, or 15
DAYS_AHEAD = 15
ADM_LEVEL = 2

def main(download=False, process=True):
    if download:
        rainy_dates = get_rainy_season_dates(RAINY_SEASON_PATH)
        for d in rainy_dates:
            download_chirpsgefs(
                pd.to_datetime(d),
                CONFIG,
                DAYS_AHEAD)
    if process:
        compute_stats_rainyseason(
            COUNTRY_ISO3,
            ADM_LEVEL,
            DAYS_AHEAD,
            CHIRPSGEFS_PROCESSED_DIR,
            RAINY_SEASON_PATH)


if __name__ == "__main__":
    main()