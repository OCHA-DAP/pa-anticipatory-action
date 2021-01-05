from pathlib import Path
import os
import sys
import xarray as xr
import rioxarray

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from indicators.drought.config import Config
from utils_general.plotting import plot_raster_boundaries
from indicators.drought.utils import parse_args
from utils_general.utils import config_logger
from indicators.drought.chirps_rainfallobservations import get_chirps_data,chirps_plot_alldates


def main(download, config=None):
    if config is None:
        config = Config()
    country = "malawi"
    parameters = config.parameters(country)
    output_dir=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,'results','drought')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds,transform=get_chirps_data(config, 2020,download=download)

    #want 2D array for plot_raster_boundaries. Without squeeze it remains 3D
    ds_3011=ds.sel(time="2020-11-30").squeeze()
    fig_bound = plot_raster_boundaries(ds_3011, country, parameters, config,forec_val="precip")
    fig_bound.savefig(os.path.join(output_dir,"chirps_20201130_boundaries.png"), format='png',bbox_inches='tight')

    ds_sel=ds.sel(time=slice("2020-11-01","2020-11-30"))
    print(ds_sel)
    adm1_path=os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin1_shp'])
    fig_histo,fig_dates=chirps_plot_alldates(ds_sel, adm1_path, config)
    fig_histo.savefig(os.path.join(output_dir,"chirps_202011_histogram.png"), format='png',bbox_inches='tight')
    fig_dates.savefig(os.path.join(output_dir,"chirps_202011_rastervalues.png"), format='png',bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.download_data)