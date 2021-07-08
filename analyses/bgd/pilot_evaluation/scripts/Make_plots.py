import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import utils_evaluation as utils
import logging

# This script takes the output interpolated data from the Generate_interpolated.py script
# and produces a series of basic choropleth maps to visualize the results

# Required inputs are:
# 1) A shapefile to delineate the admin units, located in the 'adm_dir' folder
# 2) The flood fractions by admin unit calculated from the satellite image processing. eg. 'ADM4_flood_extent_sentinel.csv'
# 3) The summary statistics from the interpolated flood fractions. eg. 'ADM4_flood_summary.csv'

# Directory locations for the input and output files should be specified in the 'config.yml' file.

DATA_DIR = os.environ["AA_DATA_DIR"]
params = utils.parse_yaml("analyses/bangladesh/pilot_evaluation/config.yml")[
    "PARAMS"
]
adm = params["adm"]
dirs = utils.parse_yaml("analyses/bangladesh/pilot_evaluation/config.yml")[
    "DIRS"
]
input_dir = os.path.join(DATA_DIR, dirs["data_dir"])
output_dir = os.path.join(DATA_DIR, dirs["plot_dir"], f"{adm}_Plots")
shp_dir = os.path.join(DATA_DIR, dirs["adm_dir"])

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# TODO: Implement functionality to remove outliers from map - (needed for FWHM)
# TODO: Reclass admin units with no flooding as zero rather than na?
# TODO: Fix hard coding with aoi selection, shapefile naming, plot numbers


def make_time_series(df_shp, df_data, id_col):
    # Time series of flooding - satellite
    df_merged = df_shp.merge(df_data, left_on=id_col, right_on=f"{adm}_PCODE")

    # Working on a figure of small multiples
    # Drawn from:
    # http://jonathansoma.com/lede/data-studio/classes/small-multiples/long-explanation-of-using-plt-subplots-to-create-small-multiples/
    # Get the axes set up - we have 27 dates in total
    fig, axes = plt.subplots(
        nrows=4, ncols=7, sharex=True, sharey=True, figsize=(15, 20)
    )
    axes_list = [item for sublist in axes for item in sublist]

    # Loop through to make the plots
    for date, selection in df_merged.groupby("date"):
        ax = axes_list.pop(0)
        selection.plot(
            column="flooded_fraction", label=date, ax=ax, legend=False
        )
        ax.set_title(date)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axis_off()

    # Now use the matplotlib .remove() method to
    # delete anything we didn't use
    for ax in axes_list:
        ax.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series.png"), dpi=300)


def make_choropleth(df_shp, df_data, id_col, map_col, title, outliers=False):
    df_merged = df_shp.merge(df_data, left_on=id_col, right_on="PCODE")
    fig, ax = plt.subplots()
    df_merged.plot(ax=ax, column=map_col, legend=True)
    plt.title(label=title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{map_col}.png"),
        dpi=300,
        bbox_inches="tight",
    )


def make_graphs(adm):
    df_sat_ext = pd.read_csv(
        os.path.join(input_dir, f"{adm}_flood_extent_sentinel.csv")
    )
    df_int_sum = pd.read_csv(
        os.path.join(input_dir, f"{adm}_flood_summary.csv")
    )

    # Read in the shp and make sure that it's within the aoi
    aoi = ["Bogra", "Gaibandha", "Jamalpur", "Kurigram", "Sirajganj"]
    id_col = ""

    if adm == "MAUZ":
        df_adm = gpd.read_file(
            os.path.join(shp_dir, "selected_distict_mauza.shp")
        )
        df_adm = df_adm[df_adm["DISTNAME"].isin(aoi)]
        id_col = "OBJECTID"
    else:
        adm_lower = adm.lower()
        df_adm = gpd.read_file(
            os.path.join(shp_dir, f"bgd_admbnda_{adm_lower}_bbs_20201113.shp")
        )
        df_adm = df_adm[df_adm["ADM2_EN"].isin(aoi)]
        df_adm = df_adm.drop(
            columns=["date"]
        )  # drop the date column from the shp
        id_col = adm + "_PCODE"

    make_time_series(df_adm, df_sat_ext, id_col)
    cols = ["FWHM", "MAX_G", "MAX_SAT", "COV", "RMSE"]
    for col in cols:
        make_choropleth(df_adm, df_int_sum, id_col, col, col)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    make_graphs(adm)
    logging.info(f"Output files saved to {output_dir}")
