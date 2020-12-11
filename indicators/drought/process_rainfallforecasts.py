import requests
import os
import logging
import sys
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.drought.config import Config
from indicators.drought.utils import parse_args
#cannot have an utils file inside the drought directory, and import from the utils directory.
#So renamed to utils_general but this should of course be changed
from utils_general.utils import config_logger

logger = logging.getLogger(__name__)

def download_iri(country, iri_auth,config,chunk_size=128):
    IRI_dir = os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.IRI_DIR)
    Path(IRI_dir).mkdir(parents=True, exist_ok=True)
    #TODO: decide if only download if file doesn't exist. Also depends on whether want one IRI file with always newest data, or one IRI file per month
    IRI_filepath = os.path.join(IRI_dir, config.IRI_NC_FILENAME)
    cookies = {
        '__dlauth_id': iri_auth,
    }
    #TODO fix/understand missing certificate verification warning
    logger.info("Downloading IRI NetCDF file. This might take some time")
    response = requests.get(config.IRI_URL, cookies=cookies, verify=False)
    with open(IRI_filepath, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def fix_calendar(ds, timevar='F'):
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def plot_forecast_boundaries(ds_nc,country, parameters, config, lon='X',lat='Y',forec_val='prob'):
    provider="IRI"
    # plot the forecast values and eth shapefile
    # for some date and leadtime combinationn a part of the world is masked. This might causes that it seems that the rasterdata and shapefile are not overlapping. But double check before getting confused by the masked values (speaking from experience)
    # retrieve lats and lons
    lons = ds_nc.variables[lon]
    lats = ds_nc.variables[lat]
    prob = ds_nc.variables[forec_val][:]
    # plot forecast and admin boundaries

    boundaries_adm1_path = os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin2_shp'])
    boundaries_world_path = os.path.join(config.DIR_PATH,config.WORLD_SHP_PATH)
    # load admin boundaries shapefile
    df_adm = gpd.read_file(boundaries_adm1_path)
    df_world = gpd.read_file(boundaries_world_path)

    fig = plt.figure(figsize=(32, 32))
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(32,8))
    ax1.contourf(lons, lats, prob)

    df_adm.boundary.plot(linewidth=1, ax=ax1, color="red")
    # plot forecast and admin boundaries
    ax2.contourf(lons, lats, prob)
    df_world.boundary.plot(linewidth=1, ax=ax2, color="red")
    fig.suptitle(f'{country} {provider} forecasted values and shape boundaries')
    return fig


def get_iri_data(country, config, parameters, leadtime=3, tercile=0, forecast_date=None, download=False):
    if download:
        iri_auth = os.getenv('IRI_AUTH')
        if not iri_auth:
            logger.error("No authentication file found")
        download_iri(country,iri_auth,config)
    # datetime format is strange, i.e. months since 1960. xarray doesn't recognize this date format, so set decode_times=False else will give an error
    IRI_dir = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country, config.DATA_DIR, config.IRI_DIR)
    Path(IRI_dir).mkdir(parents=True, exist_ok=True)
    IRI_filepath = os.path.join(IRI_dir, config.IRI_NC_FILENAME)
    # the nc contains two bands, prob and C. Still not sure what C is used for but couldn't discover useful information in it and will give an error if trying to read both (cause C is also a variable in prob)
    #the date format is formatted as months since 1960. In principle xarray can convert this type of data to datetime, but due to a wrong naming of the calendar variable it cannot do this automatically
    #Thus first load with decode_times=False and then change the calendar variable and decode the months
    iri_ds = xr.open_dataset(IRI_filepath, decode_times=False, drop_variables='C')
    iri_ds = fix_calendar(iri_ds, timevar='F')
    iri_ds = xr.decode_cf(iri_ds)
    if forecast_date is None:
        forecast_date=iri_ds['F'].max().values
    iri_ds_sel=iri_ds.sel(L=leadtime,F=forecast_date,C=tercile)

    #TODO: check range longitude and change [0,360] to [-180,180]

    #this is mainly for debugging purposes, to check if forecasted values and admin shapes correcltly align
    fig = plot_forecast_boundaries(iri_ds_sel, country, parameters, config)
    fig.savefig(os.path.join(IRI_dir, config.FORECAST_BOUNDARIES_FIGNAME), format='png')

    return iri_ds_sel

def main(country, suffix, download, config=None):
    if config is None:
        config = Config()
    parameters = config.parameters(country)
    get_iri_data(country, config,parameters, download=download)

if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country.lower(), args.suffix, args.download_data)