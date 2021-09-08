# TODO: remove this after making top-level
import os
import sys
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.config import Config
from src.indicators.flooding.floodscan import floodscan

COUNTRY_NAME = "malawi"
config = Config()
parameters = config.parameters(COUNTRY_NAME)


def main(process=True):

    floodscan_data = floodscan.Floodscan()

    if process:
        floodscan_data.process(
            country_name=COUNTRY_NAME,
            adm_level=2,
        )


if __name__ == "__main__":
    main()
