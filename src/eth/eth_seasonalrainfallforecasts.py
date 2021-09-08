import calendar
import os
import sys
from pathlib import Path

import cftime
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.icpac_rainfallforecast import get_icpac_data
from src.indicators.drought.iri_rainfallforecast import get_iri_data
from src.indicators.drought.nmme_rainfallforecast import get_nmme_data
from src.indicators.drought.utils import parse_args
from src.utils_general.plotting import (
    plot_raster_boundaries,
    plot_raster_boundaries_clip,
    plot_spatial_columns,
)
from src.utils_general.raster_manipulation import (
    compute_raster_statistics,
)
from src.utils_general.utils import config_logger


def main(download, config=None):
    country = "ethiopia"
    bins = np.arange(30, 70, 5)

    pubyear = 2021
    pubmonth = 2
    pubmonth_abbr = calendar.month_abbr[pubmonth]
    # iri and nmme publish their pubdate as months since 1960. iri uses
    # half months, e.g. 730.5, which is translated to the 16th of a
    # month nmme uses whole months, e.g. 730, which is translated to the
    # first of a month
    pubdate_cf_iri = cftime.Datetime360Day(pubyear, pubmonth, 16, 0, 0, 0, 0)
    # need a trailing 0 for nmme filename
    pubdate_str = f"{pubyear}{('0' + str(pubmonth))[-2:]}"
    leadtime = 1
    # determine month and year of start of forecast with pubdate and
    # leadtime
    forecmonth = (pubmonth + leadtime) % 12
    if forecmonth > pubmonth:
        forecyear = pubyear
    else:
        forecyear = pubyear + 1
    leadtime_cf_nmme = cftime.Datetime360Day(
        forecyear, forecmonth, 1, 0, 0, 0, 0
    )
    seasons = {
        1: "JFM",
        2: "FMA",
        3: "MAM",
        4: "AMJ",
        5: "MJJ",
        6: "JJA",
        7: "JAS",
        8: "ASO",
        9: "SON",
        10: "OND",
        11: "NDJ",
        12: "DJF",
    }
    forecseason = seasons[forecmonth]
    print(forecseason)

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
    df_bound = gpd.read_file(adm_path)

    provider = "IRI"
    iri_ds, iri_transform = get_iri_data(config, download=download)
    iri_ds = iri_ds.rename({"prob": "prob_below"})
    # C indicates the tercile where 0=below average
    iri_ds_sel = iri_ds.sel(L=leadtime, F=pubdate_cf_iri, C=0)

    iri_ds_clip = iri_ds_sel.rio.set_spatial_dims(
        x_dim="lon", y_dim="lat"
    ).rio.clip(
        df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True
    )
    print("IRI max value", iri_ds_clip[config.LOWERTERCILE].max())

    iri_ds_sel_array = iri_ds_sel["prob_below"].values
    # this is mainly for debugging purposes, to check if forecasted
    # values and admin shapes correcltly align
    fig_bound = plot_raster_boundaries(
        iri_ds_sel, country, parameters, config
    )  # ,forec_val="prob")
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

    # TODO: someting is broken with the plot_spatial_columns function
    # plot the statistics
    fig_stats = plot_spatial_columns(
        iri_df,
        ["max_cell_touched", "max_cell", "avg_cell", "avg_cell_touched"],
    )
    fig_stats.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
    )
    # plot the statistics with bins
    fig_stats_bins = plot_spatial_columns(
        iri_df,
        ["max_cell_touched", "max_cell", "avg_cell", "avg_cell_touched"],
        predef_bins=bins,
    )
    fig_stats_bins.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_"
            "Cbelow_bins.png",
        ),
        format="png",
    )

    provider = "ICPAC"
    icpac_ds, icpac_transform = get_icpac_data(
        config, pubyear, pubmonth_abbr, download=download
    )
    icpac_ds_clip = icpac_ds.rio.set_spatial_dims(
        x_dim="lon", y_dim="lat"
    ).rio.clip(
        df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True
    )
    print("ICPAC max value", icpac_ds_clip[config.LOWERTERCILE].max())
    # currently only contains one date.. should be adjusted later on

    # this is mainly for debugging purposes, to check if forecasted
    # values and admin shapes correcltly align
    # fig_bound = plot_raster_boundaries(icpac_ds, country, parameters, config)
    # fig_bound.savefig(os.path.join(output_dir,
    #                   f'{provider}_rasterbound_L{leadtime}_F{pubdate_str}_Cbelow.png'),
    #                   format='png', bbox_inches='tight') comput
    #                   statistics per admin
    icpac_ds_array = icpac_ds[config.LOWERTERCILE].values
    icpac_df = compute_raster_statistics(
        adm_path, icpac_ds_array, icpac_transform, 50
    )
    # plot the statistics
    fig_stats = plot_spatial_columns(icpac_df, statlist_plot)
    fig_stats.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
    )
    # # plot the statistics with bins
    # fig_stats_bins = plot_spatial_columns(icpac_df,
    # statlist_plot,predef_bins=bins)
    fig_stats_bins.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}"
            "_Cbelow_bins.png",
        ),
        format="png",
    )

    provider = "NMME"
    nmme_ds, nmme_transform = get_nmme_data(
        config, pubdate_str, download=download
    )
    nmme_ds_sel = nmme_ds.sel(target=leadtime_cf_nmme)
    nmme_ds_sel_array = nmme_ds_sel[config.LOWERTERCILE].values
    nmme_ds_clip = nmme_ds_sel.rio.set_spatial_dims(
        x_dim="lon", y_dim="lat"
    ).rio.clip(
        df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True
    )
    print("NMME max value", nmme_ds_clip[config.LOWERTERCILE].max())
    # this is mainly for debugging purposes, to check if forecasted
    # values and admin shapes correcltly align
    fig_bound = plot_raster_boundaries(
        nmme_ds_sel, country, parameters, config
    )
    # fig_bound.savefig(os.path.join(output_dir,
    # f'{provider}_rasterbound_L{leadtime}_F{pubdate_str}_Cbelow.png'),
    # format='png',bbox_inches='tight')

    # compute statistics per admin
    nmme_df = compute_raster_statistics(
        adm_path, nmme_ds_sel_array, nmme_transform, 50
    )

    # # plot the statistics
    fig_stats = plot_spatial_columns(nmme_df, statlist_plot)
    fig_stats.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}_Cbelow.png",
        ),
        format="png",
    )
    # # plot the statistics with bins
    # fig_stats_bins = plot_spatial_columns(nmme_df, statlist_plot,
    # predef_bins=bins)
    fig_stats_bins.savefig(
        os.path.join(
            output_dir,
            f"{provider}_statistics_L{leadtime}_F{pubdate_str}"
            "_Cbelow_bins.png",
        ),
        format="png",
    )

    # Create plot of raw dat of the different providers that only shows
    # Ethiopia
    bins = [0, 40, 45, 50, 55, 60, 70, 100]
    bins = [0, 37.5, 42.5, 47.5, 57.5, 67.5, 100]
    fig_clip = plot_raster_boundaries_clip(
        [iri_ds_sel, icpac_ds, nmme_ds_sel],
        adm_path,
        cmap="YlOrRd",
        predef_bins=bins,
        title_list=["IRI", "ICPAC", "NMME"],
        suptitle=(
            "Probability of below average rainfall \n for forecasts published"
            f" in {pubmonth_abbr} {pubyear}, forecasting {forecseason}"
            f" ({leadtime} month leadtime)"
        ),
        legend_label="Probability below average precipitation",
    )
    fig_clip.savefig(
        os.path.join(output_dir, "IRIICPACNMME_Cbelow_clipped.png"),
        format="png",
        bbox_inches="tight",
    )

    bins = [0, 40, 45, 50, 55, 60, 70, 100]
    fig_clip = plot_raster_boundaries_clip(
        [icpac_ds],
        adm_path,
        cmap="YlOrRd",
        predef_bins=bins,
        suptitle=(
            "ICPAC rainfall probabilistic forecast for March-May 2021,"
            f" published in {pubmonth_abbr} {pubyear}"
        ),
        legend_label="Probability of below-average rainfall",
    )
    # fig_clip.savefig(os.path.join(output_dir,
    # f'ICPAC_Cbelow_clipped.png'),format='png', bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.download_data)
