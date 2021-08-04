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
DATA_DIR_PUBLIC = DATA_DIR / "public" / "exploration"
DATA_DIR_PRIVATE = DATA_DIR / "private" / "exploration"

# Commonly used subdirectories
RCO_DIR = DATA_DIR_PRIVATE / "npl" / "unrco"

# GloFAS settings
COUNTRY_ISO3 = "npl"
LEADTIMES = [x + 1 for x in range(10)]

#
STATIONS_BY_BASIN = {
    "Koshi": ["Chatara_v3", "Simle_v3", "Majhitar_v3", "Kampughat_v3"],
    "Karnali": [
        "Chisapani_v3",
        "Asaraghat_v3",
        "Dipayal_v3",
        "Samajhighat_v3",
    ],
    "Rapti": ["Kusum_v3"],
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
