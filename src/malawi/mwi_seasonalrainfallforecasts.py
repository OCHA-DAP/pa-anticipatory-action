from pathlib import Path
import sys
import os
import numpy as np
import cftime

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.drought.iri_rainfallforecast import get_iri_data
from src.indicators.drought.nmme_rainfallforecast import get_nmme_data
from src.utils_general.plotting import (
    plot_spatial_columns,
    plot_raster_boundaries,
    plot_raster_boundaries_clip,
)
from src.utils_general.raster_manipulation import compute_raster_statistics

from src.indicators.drought.config import Config
from src.indicators.drought.utils import parse_args
from src.utils_general.utils import config_logger


def main(download, config=None):
    country = "malawi"
    bins = np.arange(30, 70, 5)
    pubyear = 2020
    pubmonth = 12
    # iri and nmme publish their pubdate as months since 1960. iri uses
    # half months, e.g. 730.5, which is translated to the 16th of a
    # month nmme uses whole months, e.g. 730, which is translated to the
    # first of a month
    pubdate_cf_iri = cftime.Datetime360Day(pubyear, pubmonth, 16, 0, 0, 0, 0)
    pubdate_str = f"{pubyear}{pubmonth}"
    leadtime = 3
    # TODO: make this variable based on pubdate
    leadtime_cf_nmme = cftime.Datetime360Day(2021, 3, 1, 0, 0, 0, 0)

    if config is None:
        config = Config()
    parameters = config.parameters(country)
    output_dir = os.path.join(
        config.DATA_DIR,
        config.PUBLIC_DIR,
        config.PROCESSED_DIR,
        parameters["iso3_code"],
        config.PLOT_DIR,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    adm_path = os.path.join(
        config.DATA_DIR,
        config.PUBLIC_DIR,
        config.RAW_DIR,
        parameters["iso3_code"],
        config.SHAPEFILE_DIR,
        parameters["path_admin1_shp"],
    )
    statlist_plot = [
        "max_cell_touched",
        "max_cell",
        "avg_cell",
        "avg_cell_touched",
    ]

    provider = "IRI"
    iri_ds, iri_transform = get_iri_data(config, download=download)
    iri_ds = iri_ds.rename({"prob": "prob_below"})
    # C indicates the tercile where 0=below average
    iri_ds_sel = iri_ds.sel(L=leadtime, F=pubdate_cf_iri, C=0)
    iri_ds_sel_array = iri_ds_sel["prob_below"].values
    # this is mainly for debugging purposes, to check if forecasted
    # values and admin shapes correcltly align
    fig_bound = plot_raster_boundaries(iri_ds_sel, country, parameters, config)
    fig_bound.savefig(
        os.path.join(
            output_dir,
            f"{provider}_rasterbound_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
        bbox_inches="tight",
    )
    # comput statistics per admin
    iri_df = compute_raster_statistics(
        adm_path, iri_ds_sel_array, iri_transform, 50
    )
    # plot the statistics
    fig_stats = plot_spatial_columns(iri_df, statlist_plot)
    fig_stats.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
    )
    # plot the statistics with bins
    fig_stats_bins = plot_spatial_columns(
        iri_df, statlist_plot, predef_bins=bins
    )
    output_filename = (
        f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow_bins.png"
    )
    fig_stats_bins.savefig(
        os.path.join(
            output_dir,
            output_filename,
        ),
        format="png",
    )

    provider = "NMME"
    nmme_ds, nmme_transform = get_nmme_data(
        config, pubdate_str, download=download
    )
    nmme_ds_sel = nmme_ds.sel(target=leadtime_cf_nmme)
    nmme_ds_sel_array = nmme_ds_sel[config.LOWERTERCILE].values
    # this is mainly for debugging purposes, to check if forecasted
    # values and admin shapes correcltly align
    fig_bound = plot_raster_boundaries(
        nmme_ds_sel, country, parameters, config
    )
    fig_bound.savefig(
        os.path.join(
            output_dir,
            f"{provider}_rasterbound_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
        bbox_inches="tight",
    )

    # compute statistics per admin
    nmme_df = compute_raster_statistics(
        adm_path, nmme_ds_sel_array, nmme_transform, 50
    )
    # plot the statistics
    fig_stats = plot_spatial_columns(nmme_df, statlist_plot)
    fig_stats.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
    )
    # plot the statistics with bins
    fig_stats_bins = plot_spatial_columns(
        nmme_df, statlist_plot, predef_bins=bins
    )
    output_filename = (
        f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow_bins.png"
    )
    fig_stats_bins.savefig(
        os.path.join(output_dir, output_filename),
        format="png",
    )

    # Create plot of raw dat of the different providers that only shows
    # Malawi
    fig_clip = plot_raster_boundaries_clip(
        [iri_ds_sel, nmme_ds_sel],
        adm_path,
        title_list=["IRI", "NMME"],
        suptitle=(
            "Probability of below average rainfall \n for forecasts published"
            f" in {pubdate_str} with {leadtime} months leadtime and 3 months"
            " validity"
        ),
    )
    fig_clip.savefig(
        os.path.join(output_dir, "IRIICPACNMME_Cbelow_clipped.png"),
        format="png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.download_data)
