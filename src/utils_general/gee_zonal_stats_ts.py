"""
Get an aggregated time series of an indicator or variable from Google Earth Engine for given time period and area of interest. 
Calculate daily summary stat values for admin areas and output as CSV files. 
To avoid potential memory limitations we need to run this year-by-year.
"""

import ee
import geemap
import os
import sys
from pathlib import Path

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.utils import parse_args
from src.utils_general.utils import config_logger

#TODO: should this stay here or also be refactored to another script?
# ee.Authenticate()  # Need to run this the first time using GEE API
ee.Initialize()


def main(country, config=None):
    """
    Set all the parameters needed by run_all_years
    #TODO: Refactor out so that user would specify from another script or config
    """
    if config is None:
        config = Config()
    shp_collection_path = "users/tvalentijn/ssd_adm1" #"users/hker/mwi_adm1"
    indicator_folder="chirps"#"dry_spells"
    output_dir = os.path.join(config.DATA_DIR, config.PROCESSED_DIR, country, indicator_folder, "gee_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Start and end dates should be in years
    start_year = 2000 #1998
    end_year = 2001 #2020
    adm_lvl=1
    # collection = "ECMWF/ERA5/MONTHLY"
    # band = "mean_2m_air_temperature"
    # Or for CHIRPS, for example
    collection = "UCSB-CHG/CHIRPS/DAILY"
    band = "precipitation"
    stat = "MEAN"
    parameters = config.parameters(country)
    iso3= parameters["iso3_code"]

    run_all_years(iso3,start_year,end_year,output_dir,shp_collection_path,adm_lvl,collection,band,stat)


def get_stats(iso3,start_date, end_date,output_dir,shp_collection,adm_lvl,collection,band,stat):
    """
    Retrieve collection from Google Earth Engine and compute the stat per adm_lvl.
    The output is a csv with a row per adm and a column for each date containing the specified stat
    Args:
        iso3: country-iso3 code
        start_date: first date to retrieve data for
        end_date: last data to retrieve data for
        output_dir: directory to save the stats to
        shp_collection: gee collection that contains the shapefile of the country defined by iso3
        adm_lvl: admin level to compute the stat on
        collection: name of the dataset in GEE
        band: name of the band of the collection
        stat: the statistic to compute per adm in capitals. Allowed statistics type: MEAN, MAXIMUM, MINIMUM, MEDIAN, STD, MIN_MAX, VARIANCE, SUM
    """
    img = (
        ee.ImageCollection(collection)
        .filter(ee.Filter.date(start_date, end_date))
        .select(band)
        .filterBounds(shp_collection)
        .map(lambda image: image.clip(shp_collection))
        .toBands()
    )

    scale = img.projection().nominalScale()
    col_name = collection.lower().replace("/", "-")

    out_name = os.path.join(
        output_dir,
        "{}_adm{}_{}_{}_{}.csv".format(
            iso3, str(adm_lvl), col_name, stat.lower(), end_date[0:4]
        ),
    )
    geemap.zonal_statistics(
        img, shp_collection, out_name, statistics_type=stat, scale=scale,
    )


def run_all_years(iso3,start_year,end_year,output_dir,shp_collection_path,adm_lvl,collection,band,stat):
    """
    Loop through all the years and compute the stat per adm_lvl for the given collection for each of those years
    This function calls get_stats, which saves the computed stats per year to a csv
    Args:
        iso3: country-iso3 code
        start_year: first year to retrieve data for
        end_year: last year to retrieve data for
        output_dir: directory to save the stats to
        shp_collection_path: path to where the shapefile is saved as an asset in GEE
        adm_lvl: admin level to compute the stat on
        collection: name of the dataset in GEE
        band: name of the band of the collection
        stat: the statistic to compute per adm in capitals. Allowed statistics type: MEAN, MAXIMUM, MINIMUM, MEDIAN, STD, MIN_MAX, VARIANCE, SUM
    """
    shp_collection=ee.FeatureCollection(shp_collection_path)
    for i in range(start_year, end_year + 1):
        get_stats(iso3,str(i) + "-01-01", str(i) + "-12-31",output_dir,shp_collection,adm_lvl,collection,band,stat)


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country)