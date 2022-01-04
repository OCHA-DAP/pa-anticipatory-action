import os
import sys
from datetime import timedelta
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought import arc2_precipitation
from src.indicators.drought.config import Config

COUNTRY_ISO3 = "mwi"
config = Config()
parameters = config.parameters(COUNTRY_ISO3)
ADM_SHP = (
    Path(config.DATA_DIR)
    / config.PUBLIC_DIR
    / config.RAW_DIR
    / COUNTRY_ISO3
    / "cod_ab"
    / parameters["path_admin1_shp"]
)

# TODO: Determine period of date queries
# right now set to pull date from the last 14 days
date_max = config.TODAY.strftime("%Y%m%d")
date_min = (config.TODAY - timedelta(14)).strftime("%Y%m%d")


def main(download=True, process=True, dry_spells=True):

    arc2 = arc2_precipitation.ARC2(
        country_iso3=COUNTRY_ISO3,
        date_min=date_min,
        date_max=date_max,
        range_x=None,
        range_y=None,
    )

    if download:
        arc2._download_date_ranges()

    if process:
        arc2.process_data(
            crs=parameters["crs_degrees"],
            clip_bounds=ADM_SHP,
            bound_col="ADM1_PCODE",
        )

    if dry_spells:
        arc2.identify_dry_spells()


if __name__ == "__main__":
    main()
