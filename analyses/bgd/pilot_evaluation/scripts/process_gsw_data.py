import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes

# This is a stand-alone script to perform basic processing to generate
# shapefiles to represent surface water extent for a given area of
# interest.

# TO BEGIN: Download the global surface water (GSW) tiles (Seasonality
# 2019) that you want from
# https://global-surface-water.appspot.com/download and place them
# within a single directory.

# This script will then:
# 1) Resample the GSW raster files to downscale for faster processing
# 2) Select areas with water coverage depending on the desired months of
#    coverage
# 3) Filter out water areas that are smaller than a given threshold (in
#    sq meters)
# 4) If desired, clip the water areas according to a shapefile, with
#    potential for a buffer to be specified or to clip to the bounding
#    box around the shapefile rather than the specific geometry
# 5) Export the remaining water areas to a shapefile (projected to the
#    specified UTM zone)

# SET PARAMETERS: Set the following parameters prior to running the
# script Resample factor (inverse, so 0.25 for example, will increase
# cell size X4)
RFAC = 0.25
# Processing takes a loooong time on larger files,
# so it's recommended to do a test run with a really low (eg. 0.01) RFAC
MIN_MONTHS = 10  # Minimum months of water coverage in 2019
MIN_AREA = 10000  # Minimum water area
OUTPUT_NAME = "gsw_processed_min10000.shp"  # Output file name
# Directory where the GWS .tif files are stored
DATA_DIR = "analyses/bangladesh/data/GSW_data/"
UTM = "EPSG:32646"  # UTM zone for the area of interest
CLIP = True  # Whether or not to clip the output by a given shp
# Path location to the clipping shapefile
CLIP_FILE = (
    "analyses/bangladesh/data/ADM_Shp/bgd_admbnda_adm0_bbs_20201113.shp"
)


def resample_raster(file_path, upscale_factor):
    with rasterio.open(file_path) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor),
            ),
            resampling=Resampling.bilinear,
        )
    data_reshape = data.reshape(-1, data.shape[-1])
    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
    )
    return data_reshape, transform


def process_tif(fname, factor):
    resampled, transform = resample_raster(
        os.path.join(DATA_DIR, fname), factor
    )
    print(f"Processing {fname}, resampled to {resampled.shape} cells")
    data_reclass = np.where(
        ((resampled >= MIN_MONTHS) & (resampled <= 12)), 1, 0
    )
    mask = None
    results = (
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(
            shapes(data_reclass, mask=mask, transform=transform)
        )
    )
    geoms = list(results)
    shp = gpd.GeoDataFrame.from_features(geoms)
    shp = shp.set_crs("EPSG:4326").to_crs(UTM)
    shp = shp[shp.raster_val > 0]
    shp["area"] = shp.area
    masked = shp.loc[shp.area > MIN_AREA]
    return masked


def clip_vector(to_clip, clip_geom, buffer_dist=0, bound_box=True):
    # If desired, extend a buffer (in meters) around the clip geometry
    print("Clipping geometry")
    to_clip = to_clip.buffer(0)  # Fixes some topology problems
    clip_geom = clip_geom.to_crs(to_clip.crs)
    clip_geom["geometry"] = clip_geom.buffer(buffer_dist)
    # Clip to the bounding box, rather than specific geometry
    if bound_box:
        clip_geom = clip_geom.envelope
    return gpd.clip(to_clip, clip_geom.envelope)


if __name__ == "__main__":
    files = [
        process_tif(f, RFAC)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".tif")
    ]
    all_data = gpd.GeoDataFrame(pd.concat(files, ignore_index=True))
    if CLIP:
        shp_clip = gpd.read_file(CLIP_FILE).to_crs(UTM)
        all_data = clip_vector(all_data, shp_clip)
    print(f"Writing output to {os.path.join(DATA_DIR, OUTPUT_NAME)}")
    all_data.to_file(os.path.join(DATA_DIR, OUTPUT_NAME))
