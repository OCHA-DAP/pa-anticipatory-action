"""
Get an aggregated time series of an indicator or variable from Google Earth Engine for given time period and area of interest. 
Calculate daily summary stat values for admin areas and output as CSV files. 
To avoid potential memory limitations we need to run this year-by-year.
"""

import ee
import geemap
import geopandas as gpd
import os
import sys
from pathlib import Path


# ee.Authenticate()  # Need to run this the first time using GEE API
ee.Initialize()

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

# TODO: Refactor out so that user would specify from another script or config
# Start and end dates should be in years
START = 2000
END = 2001
DATA_DIR = os.environ["AA_DATA_DIR"]
OUT_DIR = os.path.join(DATA_DIR, "processed/malawi/dry_spells/gee_output/")
ADM0 = "Malawi"
ISO3 = "mwi"
ADM_LVL = 1
STAT = "MEDIAN"  # Allowed statistics type: MEAN, MAXIMUM, MINIMUM, MEDIAN, STD, MIN_MAX, VARIANCE, SUM
ADM = ee.FeatureCollection("users/hker/mwi_adm1")
COLLECTION = "ECMWF/ERA5/MONTHLY"
BAND = "mean_2m_air_temperature"
# Or for CHIRPS, for example
# COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
# BAND = "precipitation"


def get_stats(start, end):
    img = (
        ee.ImageCollection(COLLECTION)
        .filter(ee.Filter.date(start, end))
        .select(BAND)
        .filterBounds(ADM)
        .map(lambda image: image.clip(ADM))
        .toBands()
    )

    scale = img.projection().nominalScale()
    col_name = COLLECTION.lower().replace("/", "-")

    out_name = os.path.join(
        OUT_DIR,
        "{}_adm{}_{}_{}_{}.csv".format(
            ISO3, str(ADM_LVL), col_name, STAT.lower(), end[0:4]
        ),
    )
    geemap.zonal_statistics(
        img, ADM, out_name, statistics_type=STAT, scale=scale,
    )


def run_all_years():
    for i in range(START, END + 1):
        get_stats(str(i) + "-01-01", str(i) + "-12-31")


if __name__ == "__main__":
    run_all_years()
