# TODO: remove this after making top-level
from pathlib import Path
import os
import sys

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.floodscan import floodscan

COUNTRY_NAME = "malawi"


def main(process=True):

    floodscan_data = floodscan.Floodscan()

    if process:
        floodscan_data.process(
            country_name=COUNTRY_NAME,
            adm_level=2,
        )


if __name__ == "__main__":
    main()
