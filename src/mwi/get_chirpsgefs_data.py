import logging
import os
import sys

# TODO: remove this after making top-level
from pathlib import Path

import pandas as pd

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.chirpsgefs_rainfallforecast import (
    compute_stats_rainyseason,
    download_chirpsgefs,
    get_rainy_season_dates,
)
from src.indicators.drought.config import Config

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

COUNTRY_ISO3 = "mwi"
CONFIG = Config()
DATA_PUBLIC_DIR = Path(CONFIG.DATA_DIR) / CONFIG.PUBLIC_DIR
COUNTRY_DATA_PROCESSED_DIR = (
    DATA_PUBLIC_DIR / CONFIG.PROCESSED_DIR / COUNTRY_ISO3
)
DRY_SPELLS_DIR = "dry_spells"
RAINY_SEASON_PATH = (
    COUNTRY_DATA_PROCESSED_DIR
    / DRY_SPELLS_DIR
    / "rainy_seasons_detail_2000_2020_mean_back.csv"
)
CHIRPSGEFS_PROCESSED_DIR = COUNTRY_DATA_PROCESSED_DIR / CONFIG.CHIRPSGEFS_DIR

# list of how many days ahead the forecast predicts. Can be 5, 10, or 15
DAYS_AHEAD = [5, 15]
ADM_LEVEL = 2
# list of thresholds to compute percentage of cells below the given
# threshold for.
THRESHOLD_LIST = [2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
EARLIEST_ONSET_MONTH = 11
LATEST_CESSATION_MONTH = 7


def main(download=False, process=True):
    for days in DAYS_AHEAD:
        if download:
            rainy_dates = get_rainy_season_dates(
                RAINY_SEASON_PATH, EARLIEST_ONSET_MONTH, LATEST_CESSATION_MONTH
            )
            for d in rainy_dates:
                download_chirpsgefs(pd.to_datetime(d), CONFIG, days)
        if process:
            compute_stats_rainyseason(
                COUNTRY_ISO3,
                CONFIG,
                ADM_LEVEL,
                days,
                CHIRPSGEFS_PROCESSED_DIR,
                RAINY_SEASON_PATH,
                THRESHOLD_LIST,
                EARLIEST_ONSET_MONTH,
                LATEST_CESSATION_MONTH,
            )


if __name__ == "__main__":
    main()
