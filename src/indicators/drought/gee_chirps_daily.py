"""
Access daily 0.05 degree CHIRPS data from Google Earth Engine for given time period and area of interest. 
Calculate daily summary stat values for admin areas and output as CSV files. 
To avoid potential memory limitations we need to run this year-by-year.

"""

import ee
import geemap
import geopandas as gpd
import os
import sys
from pathlib import Path

# ee.Authenticate() # Need to run this the first time using GEE API
ee.Initialize()

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

# TODO: Refactor out so that user would specify from another script of config
CHIRPS_START = 2000
CHIRPS_END = 2003
DATA_DIR = os.environ["AA_DATA_DIR"]
ADM0 = "Malawi"
ISO3 = "mwi"
ADM_LVL = 1
OUT_DIR = os.path.join(DATA_DIR, "processed/malawi/dry_spells/gee_max_chirps/")
STAT = "MAXIMUM"  # Allowed statistics type: MEAN, MAXIMUM, MINIMUM, MEDIAN, STD, MIN_MAX, VARIANCE, SUM


def chirps_summary(start, end, adm, stat, out_dir):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filter(
        ee.Filter.date(start, end)
    )
    chirps = (
        chirps.select("precipitation")
        .filterBounds(adm)
        .map(lambda image: image.clip(adm))
    )
    chirps_image = chirps.toBands()
    out_chirps_stats = os.path.join(
        out_dir,
        "{}_adm{}_chirps_{}_{}.csv".format(ISO3, str(ADM_LVL), stat.lower(), end[0:4]),
    )
    geemap.zonal_statistics(
        chirps_image, adm, out_chirps_stats, statistics_type=stat, scale=1000
    )


def run_all_years():
    # These admin boundaries are a best available from FAO as of 2015. This is the easiest to use since this data is already stored
    # within GEE. You should check that this admin data is in agreement with the other boundaries used for a project. If not, you
    # could also load a shapefile into your own GEE account as an asset and access it as shown in the commented code below.

    adm = ee.FeatureCollection("users/hker/mwi_adm2")
    # adm = ee.FeatureCollection(f"FAO/GAUL/2015/level{str(ADM_LVL)}").filterMetadata(
    #    "ADM0_NAME", "equals", ADM0
    # )
    for i in range(CHIRPS_START, CHIRPS_END + 1):
        chirps_summary(str(i) + "-01-01", str(i) + "-12-31", adm, STAT, OUT_DIR)


# TODO: Write this function.
# We need to clean up the csv files output by the zonal_statistics function and combine to a single output file.
# def make_tidy(out_dir):


if __name__ == "__main__":
    run_all_years()
