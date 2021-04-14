"""
Access daily 0.05 degree CHIRPS data from Google Earth Engine for given time period and area of interest. 
Calculate daily max values for admin areas and output as CSV files.  
"""

import ee
import geemap
import geopandas as gpd
import os
import sys
from pathlib import Path
#ee.Authenticate() # Need to run this the first time using GEE API
ee.Initialize()

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

CHIRPS_START = 2000
CHIRPS_END = 2020
DATA_DIR = os.environ['AA_DATA_DIR']
# TODO: Figure out how to programmatically upload assets from the notebook and then access them
MWI_ADM2 = ee.FeatureCollection('users/hker/mwi_adm2')

def max_chirps(start, end, adm, out_dir):
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date(start, end))
    chirps = chirps.select('precipitation').filterBounds(adm).map(lambda image: image.clip(adm))
    chirps_image = chirps.toBands()
    out_chirps_stats = out_dir + f'/processed/malawi/dry_spells/gee_max_chirps/chirps_max_{end[0:4]}.csv' 
    # TODO: Figure out the impact of the scale param (in meters)
    geemap.zonal_statistics(chirps_image, adm, out_chirps_stats, statistics_type='MAXIMUM', scale=1000)


def main():
    # TODO: Better management of dates. Not very flexible now.
    # We have to subset by year unfortunately because it hits a user memory error when trying to go all at once
    for i in range(CHIRPS_START, CHIRPS_END+1):
        max_chirps(str(i)+'-01-01', str(i)+'-12-31', MWI_ADM2, DATA_DIR) 

if __name__ == "__main__":
    main()