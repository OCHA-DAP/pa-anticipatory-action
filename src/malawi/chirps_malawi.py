from pathlib import Path
import os
import sys
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.indicators.drought.utils import parse_args
from src.utils_general.utils import config_logger
from src.indicators.drought.chirps_rainfallobservations import get_chirps_data_daily


def main(download, config=None):
    if config is None:
        config = Config()
    country = "malawi"
    parameters = config.parameters(country)
    output_dir=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,'results','drought')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resolution="05" #25
    # create list of years of interest
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    # years=[2020]
    # get data for each year
    # ADD ME warning message if data unavailable

    country_data_raw_dir = os.path.join(config.DATA_DIR, 'raw', country)
    adm1_path = os.path.join(country_data_raw_dir, config.SHAPEFILE_DIR,parameters[f'path_admin1_shp'])
    for i in years:
        #if only want to download uncomment everythin except the next line
        # download_chirps(config, i, resolution)
        ds,transform = get_chirps_data_daily(config, i, download = download,resolution=resolution)
        df_bound = gpd.read_file(adm1_path)
        #clip global to malawi to speed up calculating rolling sum
        ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.clip(
            df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
        # rolling sum of 14 days
        #TODO: check that there are no negative values!
        ds_roll=ds_clip.rolling(time=14).sum()
        print(ds_roll)
        print(np.sort(np.unique(ds_roll.precip.values.flatten()[~np.isnan(ds_roll.precip.values.flatten())])))
        # #select november month and plot those. Just because it speeds up plotting compared to 365 days
        # ds_roll_sel = ds_roll.sel(time=slice(f"{i}-11-01", f"{i}-11-30"))
        # ds_sel = ds_clip.sel(time=slice(f"{i}-11-01", f"{i}-11-30"))
        # predef_bins=np.linspace(ds_sel.precip.min(),ds_roll_sel.precip.max())

        #get the raster cell with the maximum rainfall for each date (can also do analysis when doing min instead)
        ds_roll_maxval=ds_roll.max(dim=["lon","lat"])
        df_maxprec=ds_roll_maxval.to_dataframe().reset_index()
        print("dates with all raster cells having less than 2 mm rainfall")
        print(df_maxprec[df_maxprec.precip<=2])
        # #function is not optimized yet, xlabels unreadable but gives an idea
        # fig_maxprec=plot_timeseries(df_maxprec,"time","precip")
        # fig_maxprec.savefig(os.path.join(output_dir,"yippie.png"))

""" 
    #want 2D array for plot_raster_boundaries. Without squeeze it remains 3D
    ds_3011=ds.sel(time="2019-11-30").squeeze()

    fig_bound = plot_raster_boundaries(ds_3011, country, parameters, config,forec_val="precip")
    fig_bound.savefig(os.path.join(output_dir,"chirps_20191130_boundaries.png"), format='png',bbox_inches='tight')

    ds_sel=ds.sel(time=slice("2019-11-01","2019-11-30"))
    print(ds_sel)
    adm1_path=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin1_shp'])
    fig_histo,fig_dates=chirps_plot_alldates(ds_sel, adm1_path, config)
    fig_histo.savefig(os.path.join(output_dir,"chirps_201911_histogram.png"), format='png',bbox_inches='tight')
    fig_dates.savefig(os.path.join(output_dir,"chirps_201911_rastervalues.png"), format='png',bbox_inches='tight')
"""

if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.download_data) 