from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from indicators.drought.process_rainfallforecasts import get_iri_data, get_icpac_data,plot_raster_boundaries, compute_raster_statistics, plot_spatial_columns
from indicators.drought.config import Config
from indicators.drought.utils import parse_args
from utils_general.utils import config_logger, auth_googleapi, download_gdrive, unzip


def main(download, config=None):
    country='ethiopia'
    if config is None:
        config = Config()
    parameters = config.parameters(country)
    output_dir=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,'results','drought')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    iri_ds, iri_transform = get_iri_data(config, download=download)
    iri_lastdate=iri_ds['F'].max().values
    iri_lastdate_formatted=pd.to_datetime(str(iri_lastdate)).strftime('%Y%m')
    iri_ds_latest = iri_ds.sel(L=4,F=iri_lastdate,C=0)
    iri_ds_latest_array = iri_ds_latest["prob"][:].values
    #this is mainly for debugging purposes, to check if forecasted values and admin shapes correcltly align
    fig_bound = plot_raster_boundaries(iri_ds_latest, country, parameters, config)
    fig_bound.savefig(os.path.join(output_dir, f'IRI_rasterbound_L4_F{iri_lastdate_formatted}_Cbelow.png'), format='png')
    #comput statistics per admin
    adm_path=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR, parameters['path_admin1_shp'])
    iri_df=compute_raster_statistics(adm_path,iri_ds_latest_array,iri_transform,50)
    # plot the statistics
    fig_stats = plot_spatial_columns(iri_df, ['max_cell_touched', 'max_cell', 'avg_cell', 'avg_cell_touched'])
    fig_stats.savefig(os.path.join(output_dir, f'IRI_statistics_L4_F{iri_lastdate_formatted}_Cbelow.png'), format='png')
    # plot the statistics with bins
    bins=np.arange(30,70,5)
    fig_stats_bins = plot_spatial_columns(iri_df, ['max_cell_touched', 'max_cell', 'avg_cell', 'avg_cell_touched'],predef_bins=bins)
    fig_stats_bins.savefig(os.path.join(output_dir, f'IRI_statistics_L4_F{iri_lastdate_formatted}_Cbelow_bins.png'), format='png')


# get_icpac_data(config, download=download)

    #TODO: create plot without aggregated values that only shows Ethiopia


if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.download_data)