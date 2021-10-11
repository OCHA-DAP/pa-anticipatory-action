import glob
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats

from utils_general import utils

logger = logging.getLogger(__name__)


def get_ghs_data(
    ghs_type: str,
    column_row_pairs: List[Tuple[int, int]],
    country_iso3: str,
    output_dir: Union[str, Path],
    use_cache: bool = True,
):
    """
    Download GHS data from the JRC portal.

    :param ghs_type: One of either "SMOD" or "POP".
    :param column_row_pairs: List of column & row tuples specifying
        tiles to download.
    :param country_iso3: Country ISO3 code.
    :param output_dir: Path or string specifying output directory
        for output saving.
    :param use_cache: Logical, don't re-download existing files.

    :return: Mosaic raster
    """

    # URL paths for GHS download

    URL_BASE = "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
    GHS_URL = {
        "SMOD": "GHS_SMOD_POP_GLOBE_R2019A"
        "/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K/V2-0/tiles/"
        "GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0_{column}_{row}.zip",
        "POP": "GHS_POP_MT_GLOBE_R2019A"
        "/GHS_POP_E2015_GLOBE_R2019A_54009_1K/V1-0/tiles/"
        "GHS_POP_E2015_GLOBE_R2019A_54009_1K_V1_0_{column}_{row}.zip",
    }

    # Set filepaths
    download_dir = os.path.join(output_dir, "ghs", "zip")
    output_filepath = os.path.join(
        output_dir,
        "ghs",
        "{country_iso3}_{ghs_type}_2015_1km_mosaic.tif".format(
            country_iso3=country_iso3, ghs_type=ghs_type
        ),
    )

    logger.info(f"Getting GHS data for {ghs_type}")

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    for column, row in column_row_pairs:

        zip_filename = os.path.join(
            download_dir, f"{ghs_type}_2015_1km_{column}_{row}.zip"
        )

        if use_cache and os.path.exists(zip_filename):
            logger.debug(
                (
                    f"{zip_filename} already exists "
                    "and cache is set to True, skipping"
                )
            )
            continue

        utils.download_url(
            f"{URL_BASE}/{GHS_URL[ghs_type].format(column=column, row=row)}",
            zip_filename,
        )
        utils.unzip(zip_filename, download_dir)

    # Make a mosaic
    files_to_mosaic = [
        rasterio.open(f)
        for f in glob.glob(os.path.join(download_dir, f"*_{ghs_type}_*.tif"))
    ]

    logger.info(f"Making mosiac of {len(files_to_mosaic)} files")
    mosaic, out_trans = merge(files_to_mosaic)
    out_meta = files_to_mosaic[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    with rasterio.open(output_filepath, "w", **out_meta) as dest:
        dest.write(mosaic)

    logger.info(f"Wrote file to {output_filepath}")


def classify_urban_areas(
    polygons,
    ghs_smod,
    transform,
    urban_min_class: int = 21,
    urban_percent: float = 0.5,
):
    """
    Classifies polygons as urban based on SMOD raster data.

    :param shape: Shapefile to aggregate to.
    :param ghs_smod: Array of GHS SMOD data.
    :param transform: Transform for GHS SMOD data.
    :param urban_min_class: Minimum raster value to consider as urban area.
        Defaults to 21, corresponding to suburban or peri-urban grid cells.
    :param urban_percent: Percent of raster cells classified as urban to
        consider polygon as urban.
    """

    def raster_percent(x):
        return 100 * np.ma.sum(x >= urban_min_class) / np.ma.count(x)

    def urban_area(x):
        return np.ma.mean(x >= urban_min_class) >= urban_percent

    def urban_area_weighted(x):
        return np.ma.mean(x) >= 15

    return zonal_stats(
        polygons,
        ghs_smod,
        add_stats={
            "urban_percent": raster_percent,
            "urban_area": urban_area,
            "urban_area_weighted": urban_area_weighted,
        },
        affine=transform,
    )
