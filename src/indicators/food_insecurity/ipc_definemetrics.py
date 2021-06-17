import pandas as pd
from pathlib import Path
import logging
import numpy as np
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.food_insecurity.config import Config
from src.indicators.food_insecurity.utils import parse_args
from src.utils_general.utils import config_logger

logger = logging.getLogger(__name__)

# TODO: check that all cols, so CS ML1 ML2 1 to 5 and pop cols are present in input data
# TODO: quality check that perc cols add up to 100

def define_trigger_percentage(row, period, level, perc):
    """
    Return 1 if percentage of population in row for period in level "level" or higher, equals or larger than perc
    """
    # range till 6 cause 5 is max level
    cols = [f"{period}_{l}" for l in range(level, 6)]
    if np.isnan(row[f"pop_{period}"]):
        return np.nan
    if round(row[cols].sum() / row[f"pop_{period}"] * 100) >= perc:
        return 1
    else:
        return 0

def define_trigger_increase_rel(row, level, perc):
    """
    Return 1 if population in row for >="level" at ML1 is expected to be larger than (current (CS) population in >=level) * (1+(perc/100))
    """
    # range till 6 cause 5 is max level
    cols_ml1 = [f"ML1_{l}" for l in range(level, 6)]
    cols_cs = [f"CS_{l}" for l in range(level, 6)]
    if row[["pop_CS", "pop_ML1"]].isnull().values.any():
        return np.nan
    elif row[cols_ml1].sum() == 0:
        return 0
    elif row[cols_ml1].sum() > 0 and row[cols_cs].sum() == 0:
        return 1
    elif (
        round((row[cols_ml1].sum() - row[cols_cs].sum()) / row[cols_cs].sum() * 100)
        >= perc
    ):
        return 1
    else:
        return 0

def define_trigger_increase(row, period, level, perc):
    """
    Return 1 for "row", if the expected increase in the percentage of the population in "level" or higher at time "period" compared to currently (CS) is expected to be larger than "perc"
    For Global IPC the population analysed in ML2 is sometimes different than in CS. That is why we work dirrectly with percentages and not anymore with (pop period level+ - pop CS level+) / pop CS
    """
    # range till 6 cause 5 is max level
    cols_perc_proj = [f"perc_{period}_{l}" for l in range(level, 6)]
    cols_perc_cs = [f"perc_CS_{l}" for l in range(level, 6)]
    if row[["pop_CS", f"pop_{period}"]].isnull().values.any():
        return np.nan
    if row[cols_perc_proj].sum() == 0:
        return 0
    if round(row[cols_perc_proj].sum() - row[cols_perc_cs].sum()) >= perc:
        return 1
    else:
        return 0
