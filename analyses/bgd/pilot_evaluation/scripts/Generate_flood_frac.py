import logging
import os

import geopandas as gpd
import pandas as pd
import utils_evaluation as utils

# This script takes the flood extent shapefiles output from the GEE
# Sentinel-1 data processing script and outputs a .csv file that
# provides a time series of the flooding fraction by admin units in
# Bangladesh.

# Required inputs are:
# 1) The shapefiles of flood extent output from the GEE script, located
#    within the same 'gee_dir'
# 2) Shapefile of admin regions in Bangladesh, located in the 'adm_dir'
# 3) Shapefile with permanent water bodies in Bangladesh, located in the
#    'adm_dir' folder
# 4) The admin level used to calculate the flood fraction (Eg. ADM2,
#    ADM3, ADM4), located in the 'config.yml'

# Directory locations for the input and output files should be specified
# in the 'config.yml' file.

# TODO: Look for ways to optimize. Currently is very slow, likely due to
# the geopandas overlay operations.
# TODO: Fix variable hard-coding.
# Currently hard-coded variables include:
# - Bangladesh districts in the region of interest
# - Column names for shapefiles
# TODO: Move tests to separate file

DATA_DIR = os.environ["AA_DATA_DIR"]

dirs = utils.parse_yaml("analyses/bangladesh/pilot_evaluation/config.yml")[
    "DIRS"
]
SHP_DIR = os.path.join(DATA_DIR, dirs["gee_dir"])
OUT_DIR = os.path.join(DATA_DIR, dirs["data_dir"])

files = utils.parse_yaml("analyses/bangladesh/pilot_evaluation/config.yml")[
    "FILES"
]
ADM_SHP = os.path.join(DATA_DIR, files["adm_shp"])
RIVER_SHP = os.path.join(DATA_DIR, files["river_shp"])

params = utils.parse_yaml("analyses/bangladesh/pilot_evaluation/config.yml")[
    "PARAMS"
]
CRS = params["crs"]
ADM = params["adm"]

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def get_adm_shp(shp_admin: str) -> gpd.GeoDataFrame:
    """Loads and processes the admin region shapefile.

    :param shp_admin: name of admin shapefile
    :return: geopandas df with admin regions
    """

    shp = gpd.read_file(shp_admin).to_crs(CRS)
    aoi = ["Bogra", "Gaibandha", "Jamalpur", "Kurigram", "Sirajganj"]
    if ADM == "MAUZ":
        shp = shp[shp["DISTNAME"].isin(aoi)]
        shp.rename(
            columns={"OBJECTID": "MAUZ_PCODE", "MAUZNAME": "MAUZ_EN"},
            inplace=True,
        )
    else:
        shp = shp[shp["ADM2_EN"].isin(aoi)]
    shp.loc[:, "adm_area"] = shp["geometry"].area

    logging.info(f"Area of interest contains {len(shp.index)} admin units.")

    try:
        assert len(
            shp[f"{ADM}_PCODE"].unique() == len(shp.index)
        ), "PCODE field is not unique"
    except AssertionError as error:
        logging.error(error)

    return shp


def get_river_area(
    shp_river: str, gdf_admin: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Subtract the river area from admin units. River area shp comes
    from JRC Global Surface Water.

    :param shp_river: shapefile delineating the river area :param
    gdf_admin: geodataframe with admin areas :return: geodataframe with
    land area by admin unit
    """

    gdf_river = gpd.read_file(shp_river).to_crs(CRS)
    gdf_intersection = gpd.overlay(gdf_admin, gdf_river, how="difference")
    gdf_intersection["land_area"] = gdf_intersection["geometry"].area
    gdf_admin_river = gdf_admin.merge(
        gdf_intersection[["land_area", f"{ADM}_PCODE"]],
        on=f"{ADM}_PCODE",
        how="left",
    )
    gdf_admin_river.loc[:, "land_area"] = gdf_admin_river["land_area"].fillna(
        0
    )
    gdf_admin_river.loc[:, "river_area"] = (
        gdf_admin_river["adm_area"] - gdf_admin_river["land_area"]
    )

    num_all_river = len(gdf_admin.index) - len(gdf_intersection.index)
    if num_all_river > 0:
        logging.info(
            f"There are {num_all_river} admin units that are entirely covered"
            " by the river"
        )

    try:
        assert (
            len(gdf_admin_river[gdf_admin_river["river_area"] < 0]) == 0
        ), "Output has negative river area"
        assert (
            len(gdf_admin_river[gdf_admin_river["land_area"] < 0]) == 0
        ), "Output has negative land area"
        assert len(gdf_admin_river.index) == len(
            gdf_admin.index
        ), "Output does not have same number of admin units as input."

    except AssertionError as error:
        logging.error(error)

    return gdf_admin_river[
        [f"{ADM}_PCODE", "geometry", "adm_area", "land_area"]
    ]


def get_flood_area(
    gdf_admin_river: gpd.GeoDataFrame, shp_dir: str
) -> gpd.GeoDataFrame:
    """Calculate the flooded area for each admin region for a given
    point in time.

    :param gdf_admin_river: Shapefile with admin boundaries :param
    shp_dir: Shapefile directory :return: dataframe with the total
    flooded area by admin region
    """

    df_output = pd.DataFrame()

    for fname in os.listdir(shp_dir):
        if fname.startswith("BGD_Floods") and fname.endswith(".shp"):

            date = fname[11:21]
            logging.info(f"Processing flooding from {date}")

            gdf_flood = gpd.read_file(os.path.join(SHP_DIR, fname)).to_crs(CRS)

            # Challenge here is to make sure that both the admin units
            # with 0% flooding and 100% flooding are accounted for.
            gdf_intersection = gpd.overlay(
                gdf_admin_river, gdf_flood, how="difference"
            )
            gdf_intersection["not_flooded_area"] = gdf_intersection[
                "geometry"
            ].area
            gdf_admin_flooded = gdf_admin_river.merge(
                gdf_intersection[["not_flooded_area", f"{ADM}_PCODE"]],
                on=f"{ADM}_PCODE",
                how="left",
            )
            gdf_admin_flooded["not_flooded_area"].fillna(
                0, inplace=True
            )  # NA values encountered when the area is all flooded
            gdf_admin_flooded.loc[:, "flooded_area"] = (
                gdf_admin_flooded["adm_area"]
                - gdf_admin_flooded["not_flooded_area"]
            )
            gdf_admin_flooded.loc[:, "flooded_fraction"] = round(
                gdf_admin_flooded["flooded_area"]
                / gdf_admin_flooded["land_area"],
                4,
            )
            gdf_admin_flooded.loc[:, "date"] = date

            df_output = df_output.append(
                gdf_admin_flooded[[f"{ADM}_PCODE", "flooded_fraction", "date"]]
            )

            try:
                assert len(gdf_admin_flooded.index) == len(
                    gdf_admin_river.index
                ), (
                    f"Output from {date} does not have same number of admin"
                    " units as input"
                )
                assert (
                    len(
                        gdf_admin_flooded[
                            gdf_admin_flooded["flooded_fraction"] > 1
                        ]
                    )
                    == 0
                ), f"Output from {date} has flooded fraction greater than 1"
                assert (
                    len(
                        gdf_admin_flooded[
                            gdf_admin_flooded["flooded_fraction"] < 0
                        ]
                    )
                    == 0
                ), f"Output from {date} has flooded fraction less than 0"
            except AssertionError as error:
                logging.error(error)

    return df_output


def get_dates(shp_dir):
    """Get the dates with imagery by parsing the file names in the
    Google Earth Engine (GEE) output directory.

    Assumes the file naming convention specified in the GEE script. Also
    assumes that the GEE output files are the only things in the
    directory that start with 'BGD'.
    """

    dates = [
        fname[11:21]
        for fname in os.listdir(shp_dir)
        if fname.startswith("BGD")
    ]
    return set(dates)


def sentinel_output_qa(df_ts: pd.DataFrame, df_shp: gpd.GeoDataFrame) -> None:
    """Basic quality checks to validate calculations of flooding
    fraction by admin unit from the Sentinel-1 derived shapefiles:

    Number of unique admin units in the output csv matches those in the
    shapefile. Flooding fraction is within the [0,1] range. Number of
    data points for each admin unit is equal to the number of dates with
    imagery.

    Also reports on any admin units that did not experience any
    flooding. And the number of NA values for flooding in the admin
    units.
    """

    num_dates = len(get_dates(SHP_DIR))
    num_admin_ts = len(set(df_ts[f"{ADM}_PCODE"]))
    num_admin_shp = len(df_shp.index)
    df_group_count = df_ts.groupby(f"{ADM}_PCODE").count().reset_index()
    less_dates = df_group_count[df_group_count["date"] < num_dates]
    more_dates = df_group_count[df_group_count["date"] > num_dates]

    try:
        assert (
            num_admin_shp == num_admin_ts
        ), "Mismatching number of admin units between the shp and output file"
        assert (
            len(less_dates.index) == 0
        ), "Some admin units have missing data points"
        assert (
            len(more_dates.index) == 0
        ), "Some admin units have too many points"
        assert (
            len(df_ts[df_ts.flooded_fraction < 0].index) == 0
        ), "Flood fraction goes below 0 in some admin units"
        assert (
            len(df_ts[df_ts.flooded_fraction > 1].index) == 0
        ), "Flood fraction goes above 1 in some admin units"

    except AssertionError as error:
        logging.error(error)

    df_group = df_ts.groupby(f"{ADM}_PCODE").mean().reset_index()
    no_flood = len(df_group[df_group["flooded_fraction"] == 0])
    if no_flood > 0:
        logging.info(f"There are {no_flood} admin units with no flooding")

    num_nan = df_ts["flooded_fraction"].isna().sum()
    logging.info(
        f"There are {num_nan} instances of NaN values in the flooded fraction"
        " column"
    )


def main():
    gdf_admin = get_adm_shp(ADM_SHP)
    gdf_admin_land = get_river_area(RIVER_SHP, gdf_admin)
    df_flooded_frac = get_flood_area(gdf_admin_land, SHP_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    df_flooded_frac.to_csv(
        os.path.join(OUT_DIR, f"{ADM}_flood_extent_sentinel.csv"), index=False
    )
    sentinel_output_qa(df_flooded_frac, gdf_admin)


if __name__ == "__main__":
    main()
