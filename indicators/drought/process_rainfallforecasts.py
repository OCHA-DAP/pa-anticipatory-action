import requests
import os
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats
import pandas as pd
# rioxarray seems to do a better job at correctly reading the data than xarray related to coordinates etc.
# at the same time rioxarray is still under a lot of development so not always working as stably as xarray..
#so currently using both and hopefully in the future can do most with rioxarray
import rioxarray
import xarray as xr
from shapely.geometry import mapping
import shutil
import cartopy.crs as ccrs

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.drought.config import Config
from indicators.drought.utils import parse_args
#cannot have an utils file inside the drought directory, and import from the utils directory.
#So renamed to utils_general but this should maybe be changed
from utils_general.utils import config_logger, auth_googleapi, download_gdrive, unzip, download_ftp
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

#TODO: understand rioxarray vs xarray
#TODO: define names of vars in config instead of hardcoding in functions

#plot functions
def plot_raster_boundaries(ds_nc,country, parameters, config, lon='lon',lat='lat',forec_val='prob_below'):
    """
    plot a raster file and a shapefile on top of each other with the goal to see if their coordinate systems match
    Two plots are made, one with the adm2 shapefile of the country and one with the worldmap
    For some forecast providers a part of the world is masked. This might causes that it seems that the rasterdata and shapefile are not overlapping. But double check before getting confused by the masked values (speaking from experience)

    Args:
        ds_nc (xarray dataset): dataset that should be plotted, all bands contained in this dataset will be plotted
        country (str): country for which to plot the adm2 boundaries
        parameters (dict): parameters for the specific country
        config (Config): config for the drought indicator
        lon (str): name of the longitude coordinate in ds_nc
        lat (str): name of the latitude coordinate in ds_nc
        forec_val (str): name of the variable that contains the values to be plotted in ds_nc

    Returns:
        fig (fig): two subplots with the raster and the country and world boundaries
    """
    #initialize empty figure, to circumvent that figures from different functions are overlapping
    plt.figure()

    #TODO: test if this change works with all providers

    # retrieve lats and lons
    lons=ds_nc.coords[lon]
    lats=ds_nc.coords[lat]
    prob=ds_nc[forec_val]

    boundaries_adm1_path = os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin2_shp'])
    boundaries_world_path = os.path.join(config.DIR_PATH,config.WORLD_SHP_PATH)
    # load admin boundaries shapefile
    df_adm = gpd.read_file(boundaries_adm1_path)
    df_world = gpd.read_file(boundaries_world_path)

    # plot forecast and admin boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(32,8))
    ax1.pcolormesh(lons, lats, prob)
    df_adm.boundary.plot(linewidth=1, ax=ax1, color="red")
    ax2.pcolormesh(lons, lats, prob)
    df_world.boundary.plot(linewidth=1, ax=ax2, color="red")
    fig.suptitle(f'{country} forecasted values and shape boundaries')
    return fig

def plot_raster_boundaries_test(ds_list,country, parameters, config, lon='lon',lat='lat',forec_val='prob_below',title_list=None):
    #TODO: this function has to be finalized and integrated with plot_raster_boundaries!
    #compared to plot_raster_boundaries, this function is working with clipped values and a list of datasets
    """
    plot a raster file and a shapefile on top of each other with the goal to see if their coordinate systems match
    Two plots are made, one with the adm2 shapefile of the country and one with the worldmap
    For some forecast providers a part of the world is masked. This might causes that it seems that the rasterdata and shapefile are not overlapping. But double check before getting confused by the masked values (speaking from experience)

    Args:
        ds_nc (xarray dataset): dataset that should be plotted, all bands contained in this dataset will be plotted
        country (str): country for which to plot the adm2 boundaries
        parameters (dict): parameters for the specific country
        config (Config): config for the drought indicator
        lon (str): name of the longitude coordinate in ds_nc
        lat (str): name of the latitude coordinate in ds_nc
        forec_val (str): name of the variable that contains the values to be plotted in ds_nc

    Returns:
        fig (fig): two subplots with the raster and the country and world boundaries
    """
    #initialize empty figure, to circumvent that figures from different functions are overlapping
    plt.figure()

    boundaries_adm1_path = os.path.join(config.DIR_PATH,config.ANALYSES_DIR,country,config.DATA_DIR,config.SHAPEFILE_DIR,parameters['path_admin1_shp'])
    # load admin boundaries shapefile
    df_adm = gpd.read_file(boundaries_adm1_path)

    num_plots = len(ds_list)
    if num_plots>1:
        colp_num=2
    else:
        colp_num=1
    rows = num_plots // colp_num
    rows += num_plots % colp_num
    position = range(1, num_plots + 1)

    #TODO: find a better way to define bins that always capture full range of values but are consistent
    #TODO: define a missing values color
    predef_bins = np.arange(30, 61, 2.5)
    norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    cmap=plt.cm.jet
    # fig, axes = plt.subplots(nrows=rows, ncols=colp_num)
    fig, axes = plt.subplots(rows, colp_num)
    if num_plots==1:
        axes.set_axis_off()
    else:
        [axi.set_axis_off() for axi in axes.ravel()]
    # fig = plt.figure(figsize=(16, 8))
    for i, ds in enumerate(ds_list):
        #TODO: decide if want to use projection and if Robinson is then good one
        ax = fig.add_subplot(rows, colp_num, position[i],projection=ccrs.Robinson())

        #TODO: spatial_dims needed for ICPAC data but don't understand why yet
        clipped = ds.rio.set_spatial_dims(x_dim=lon,y_dim=lat).rio.clip(df_adm.geometry.apply(mapping), df_adm.crs, all_touched=True)
        lons = clipped.coords[lon]
        lats = clipped.coords[lat]
        prob = clipped[forec_val]

        im = plt.pcolormesh(lons, lats, prob, cmap=cmap, norm=norm)

        if title_list is not None:
            plt.title(title_list[i], size=8)

        df_adm.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    # cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label('prob', size=12, rotation=0, labelpad=15)
    cb.ax.tick_params(labelsize=10)

    # plt.tight_layout()
    # fig.suptitle(f'{country} forecasted values and shape boundaries')
    return fig

def plot_spatial_columns(df, col_list, title=None, predef_bins=None,cmap='YlOrRd'):
    """
    Create a subplot for each variable in col_list where the value for the variable per spatial region defined in "df" is shown
    If predef_bins are given, the values will be classified to these bins. Else a different color will be given to each unique value
    Args:
        df (GeoDataframe): dataframe containing the values for the variables in col_list and the geo polygons
        col_list (list of strs): indicating the column names that should be plotted
        title (str): title of plot
        predef_bins (list of numbers): bin boundaries
        cmap (str): colormap to use

    Returns:
        fig: figure with subplot for each variable in col_list
    """

    #initialize empty figure, to circumvent that figures from different functions are overlapping
    plt.figure()
    plt.clf()

    #define the number of columns and rows
    colp_num = 2
    num_plots = len(col_list)
    rows = num_plots // colp_num
    rows += num_plots % colp_num
    position = range(1, num_plots + 1)

    #TODO: messy now with the axes and ax objects, find neater method. If using plt.figure and then add_subplots calling function several times will overlap the plots :/
    # fig = plt.figure(1, figsize=(16, 6 * rows))
    fig, axes=plt.subplots(rows,colp_num)
    [axi.set_axis_off() for axi in axes.ravel()]
    #if bins, set norm to classify the values in the bins
    if predef_bins is not None:
        scheme = None
        norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
        legend_kwds=None
    else:
        scheme = "natural_breaks"
        norm = None
        legend_kwds = {'bbox_to_anchor': (1.6, 1)}

    for i, col in enumerate(col_list):
        ax = fig.add_subplot(rows, colp_num, position[i])

        #if no predef bins, set unique color for each unique value
        if predef_bins is None:
            colors = len(df[col].dropna().unique())
        #else colors will be determined by norm and cmap
        else:
            colors = None

        if df[col].isnull().values.all():
            print(f"No not-NaN values for {col}")
        #cannot handle missing_kwds if there are no missing values, so have to define those two cases separately
        elif df[col].isnull().values.any():
            df.plot(col, ax=ax, legend=True, k=colors, cmap=cmap, norm=norm, scheme=scheme,
                    legend_kwds=legend_kwds,
                    missing_kwds={"color": "lightgrey", "edgecolor": "red",
                                  "hatch": "///",
                                  "label": "No values"})
        else:
            df.plot(col, ax=ax, legend=True, k=colors, cmap=cmap, norm=norm, scheme=scheme,
                    legend_kwds=legend_kwds
                    )

        df.boundary.plot(linewidth=0.2, ax=ax)

        plt.title(col)
        ax.axis("off")
        plt.axis("off")

        #prettify legend if using individual color for each value
        if predef_bins is None and not df[col].isnull().values.all():
            leg = ax.get_legend()

            for lbl in leg.get_texts():
                label_text = lbl.get_text()
                upper = label_text.split(",")[-1].rstrip(']')

                try:
                    new_text = f'{float(upper):,.2f}'
                except:
                    new_text = upper
                lbl.set_text(new_text)

    #TODO: fix legend and prettify plot

        #     legend_elements= [Line2D([0], [0], marker='o',markersize=15,label=k,color=color_dict[k],linestyle='None') for k in color_dict.keys()]
    # leg=plt.legend(title='Legend',frameon=False,handles=legend_elements,bbox_to_anchor=(1.5,0.8))
    # leg._legend_box.align = 'left'

    if title:
        fig.suptitle(title, fontsize=14, y=0.92)
    fig.tight_layout()

    return fig


#data retrieving and manipulating
def invert_latlon(ds):
    """
    This function checks for inversion of latitude and longitude and changes them if needed
    Some datasets come with flipped coordinates.
    For latitude this means that the coordinates start with the smallest number while for longitude they are flipped when starting with the largest number.
    Some functions, such as zonal_stats, produce wrong results when these coordinates are flipped.
    The dataset given as input (ds) should have its lat coordinates named "lat" and lon coordinates "lon"
    Function largely copied from https://github.com/perrygeo/python-rasterstats/issues/218
    Args:
        ds (xarray dataset): dataset containing the variables and coordinates

    Returns:
        da (xarray dataset): dataset containing the variables and flipped coordinates
    """
    lat_start=ds.lat[0].item()
    lat_end=ds.lat[ds.dims["lat"]-1].item()
    lon_start=ds.lon[0].item()
    lon_end=ds.lon[ds.dims["lon"]-1].item()
    if lat_start<lat_end:
        lat_status='north down'
    else:
        lat_status='north up'
    if lon_start<lon_end:
        lon_status='lon normal'
    else:
        lon_status='lon inverted'

    # Flip the raster as necessary (based on the flags)
    if lat_status=='north down':
        logger.info("Dataset was north down, latitude coordinates have been flipped")
        ds=ds.reindex(lat=ds["lat"][::-1])
    #TODO: implement longitude inversion
    if lon_status=='inverted':
        logger.error("Inverted longitude still needs to be implemented..")
        # ds=np.flip(ds.squeeze('F'),axis=1)
    return ds

def fix_calendar(ds, timevar='F'):
    """
    Some datasets come with a wrong calendar attribute that isn't recognized by xarray
    So map this attribute that can be read by xarray
    Args:
        ds (xarray dataset): dataset of interest
        timevar (str): variable that contains the time in ds

    Returns:

    """
    if "calendar" in ds[timevar].attrs.keys():
        if ds[timevar].attrs['calendar'] == '360':
            ds[timevar].attrs['calendar'] = '360_day'
    elif "units" in ds[timevar].attrs.keys():
        if "months since" in ds[timevar].attrs['units']:
            ds[timevar].attrs['calendar'] = '360_day'
    return ds

#computations
def compute_raster_statistics(boundary_path, raster_array, raster_transform, threshold, upscale_factor=None):
    df = gpd.read_file(boundary_path)
    #TODO: decide if we want to upsample and if yes, implement
    # if upscale_factor:
    #     forecast_array, transform = resample_raster(raster_path, upscale_factor)
    # else:

    # extract statistics for each polygon. all_touched=True includes all cells that touch a polygon, with all_touched=False only those with the center inside the polygon are counted.
    df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=1, nodata=-9999))["max"]
    df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=1, nodata=-9999))["max"]

    df["avg_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=1, nodata=-9999))[
        "mean"]
    df["avg_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=1, nodata=-9999))[
        "mean"]

    # calculate the percentage of the area within an admin that has a value larger than threshold
    forecast_binary = np.where(raster_array >= threshold, 1, 0)
    bin_zonal = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, nodata=-999, affine=raster_transform, stats=['count', 'sum']))
    df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
    bin_zonal_touched = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, nodata=-999, affine=raster_transform, all_touched=True, stats=['count', 'sum']))
    df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100

    return df

def download_iri(iri_auth,config,chunk_size=128):
    """
    Download the IRI seasonal tercile forecast as NetCDF file from the url as given in config.IRI_URL
    Saves the file to the path as defined in config
    Args:
        iri_auth: iri key for authentication. An account is needed to get this key
        config (Config): config for the drought indicator
        chunk_size (int): number of bytes to download at once
    """
    #TODO: it would be way nicer to download with opendap instead of requests, since then the file doesn't even have to be saved and is hopefully faster. Only, cannot figure out how to do that with cookie authentication
    IRI_dir = os.path.join(config.DROUGHTDATA_DIR,config.IRI_DIR)
    Path(IRI_dir).mkdir(parents=True, exist_ok=True)
    #TODO: decide if only download if file doesn't exist. Also depends on whether want one IRI file with always newest data, or one IRI file per month
    IRI_filepath = os.path.join(IRI_dir, config.IRI_NC_FILENAME_RAW)
    # strange things happen when just overwriting the file, so delete it first if it already exists
    if os.path.exists(IRI_filepath):
        os.remove(IRI_filepath)

    #have to authenticate by using a cookie
    cookies = {
        '__dlauth_id': iri_auth,
    }
    #TODO fix/understand missing certificate verification warning
    logger.info("Downloading IRI NetCDF file. This might take some time")
    response = requests.get(config.IRI_URL, cookies=cookies, verify=False)
    with open(IRI_filepath, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    #TODO: explore if can also open with rioxarray instead of xarray. Getting an error with current settings
    iri_ds = xr.open_dataset(IRI_filepath, decode_times=False, drop_variables='C')
    IRI_filepath_crs = os.path.join(IRI_dir, config.IRI_NC_FILENAME_CRS)

    if os.path.exists(IRI_filepath_crs):
        os.remove(IRI_filepath_crs)

    #TODO: not sure if this block fits better here or in get_iri_data
    #invert_latlon assumes lon and lat to be the names of the coordinates
    iri_ds=iri_ds.rename({"X": "lon", "Y": "lat"})
    #often IRI latitude is flipped so check for that and invert if needed
    iri_ds = invert_latlon(iri_ds)
    # The iri data is in EPSG:4326 but this isn't included in the filedata (at least from experience)
    # This Coordinate Reference System (CRS) information is later on needed, so add it to the file
    iri_ds.rio.write_crs("EPSG:4326").to_netcdf(IRI_filepath_crs)

    #TODO: check range longitude and change [0,360] to [-180,180]


def get_iri_data(config, download=False):
    """
    Load IRI's NetCDF as a xarray dataset
    Args:
        config (Config): config for the drought indicator
        download (bool): if True, download data

    Returns:
        iri_ds (xarray dataset): dataset continaing the information in the netcdf file
        transform (numpy array): affine transformation of the dataset based on its CRS
    """
    if download:
        #need a key, assumed to be saved as an env variable with name IRI_AUTH
        iri_auth = os.getenv('IRI_AUTH')
        if not iri_auth:
            logger.error("No authentication file found. Needs the environment variable 'IRI_AUTH'")
        download_iri(iri_auth,config)
    IRI_filepath = os.path.join(config.DROUGHTDATA_DIR, config.IRI_DIR, config.IRI_NC_FILENAME_CRS)
    # the nc contains two bands, prob and C. Still not sure what C is used for but couldn't discover useful information in it and will give an error if trying to read both (cause C is also a variable in prob)
    #the date format is formatted as months since 1960. In principle xarray can convert this type of data to datetime, but due to a wrong naming of the calendar variable it cannot do this automatically
    #Thus first load with decode_times=False and then change the calendar variable and decode the months
    iri_ds = xr.open_dataset(IRI_filepath, decode_times=False, drop_variables='C')
    iri_ds = fix_calendar(iri_ds, timevar='F')
    iri_ds = xr.decode_cf(iri_ds)

    #TODO: understand rasterio warnings "CPLE_AppDefined in No UNIDATA NC_GLOBAL:Conventions attribute" and "CPLE_AppDefined in No 1D variable is indexed by dimension C"
    with rasterio.open(IRI_filepath) as src:
        transform = src.transform

    return iri_ds, transform


#WIP
def download_icpac(config):
    #TODO: would be nicer to directly download from their ftp but couldn't figure out yet how (something with certificates)
    gclient = auth_googleapi()
    gzip_output_file=os.path.join(config.DROUGHTDATA_DIR, f'{config.ICPAC_DIR}.zip')
    if os.path.exists(os.path.join(config.DROUGHTDATA_DIR, config.ICPAC_DIR)):
        shutil.rmtree(os.path.join(config.DROUGHTDATA_DIR, config.ICPAC_DIR))
    download_gdrive(gclient, config.ICPAC_GDRIVE_ZIPID, gzip_output_file)
    unzip(gzip_output_file,config.DROUGHTDATA_DIR)
    os.remove(gzip_output_file)
    for path in Path(os.path.join(config.DROUGHTDATA_DIR,config.ICPAC_DIR)).rglob(config.ICPAC_PROBFORECAST_REGEX_RAW):
        #opening with rioxarray better than xarray, with xarray gets some lat lon inversion
        icpac_ds = rioxarray.open_rasterio(path)
        #TODO: selection of below is needed to save crs correctly but don't understand why
        icpac_sel = icpac_ds[config.ICPAC_LOWERTERCILE]
        path_crs = f"{str(path)[:-3]}_crs.nc"
        if os.path.exists(path_crs):
            os.remove(path_crs)
        icpac_sel.rio.write_crs("EPSG:4326").to_netcdf(path_crs)


def get_icpac_data(config,pubyear,pubmonth,download=False):
    if download:
        download_icpac(config)
    #TODO: check if better way to find the file
    try:
        for path in Path(os.path.join(config.DROUGHTDATA_DIR,config.ICPAC_DIR)).rglob(config.ICPAC_PROBFORECAST_REGEX_CRS.format(month=pubmonth,year=pubyear)):

            #rioxarray reads the icpac data correctly while xarray somehow messes up stuff but still not sure what exactly goes wrong there
            icpac_ds = rioxarray.open_rasterio(path).squeeze()
            icpac_ds = icpac_ds.rename({config.ICPAC_LON: config.LONGITUDE, config.ICPAC_LAT: config.LATITUDE, config.ICPAC_LOWERTERCILE:config.LOWERTERCILE})

            #assume all transforms of the different files are the same, so just select the one that is read the latest
            with rasterio.open(path) as src:
                transform = src.transform
        return icpac_ds, transform
    except UnboundLocalError:
        logger.error(f"ICPAC forecast with regex {config.ICPAC_PROBFORECAST_REGEX_CRS.format(month=pubmonth,year=pubyear)} not found")

def download_nmme(config,date):
    NMME_dir = os.path.join(config.DROUGHTDATA_DIR, config.NMME_DIR)
    Path(NMME_dir).mkdir(parents=True, exist_ok=True)
    NMME_filepath = os.path.join(NMME_dir, config.NMME_NC_FILENAME_RAW.format(date=date))
    # TODO: decide if only download if file doesn't exist. Not sure if ever gets updated
    if not os.path.exists(NMME_filepath):
        download_ftp(config.NMME_FTP_URL_SEASONAL.format(date=date), NMME_filepath)

        # TODO: explore if can also open with rioxarray instead of xarray. Getting an error with current settings
        nmme_ds = xr.open_dataset(NMME_filepath, decode_times=False)

        #generally nmme coordinates are not inverted, but this is a double check
        nmme_ds = invert_latlon(nmme_ds)
        nmme_ds = fix_calendar(nmme_ds, timevar='target')
        nmme_ds = fix_calendar(nmme_ds, timevar='initial_time')
        nmme_ds = xr.decode_cf(nmme_ds)
        NMME_filepath_crs = os.path.join(NMME_dir, config.NMME_NC_FILENAME_CRS.format(date=date))

        # strange things happen when just overwriting the file, so delete it first if it already exists
        if os.path.exists(NMME_filepath_crs):
            os.remove(NMME_filepath_crs)
        #TODO: crs is only saved correctly when saving one var. Don't understand why
        nmme_ds[config.NMME_LOWERTERCILE].rio.write_crs("EPSG:4326").to_netcdf(NMME_filepath_crs)

def get_nmme_data(config,date,download=False):
    if download:
        download_nmme(config,date)
    NMME_filepath = os.path.join(config.DROUGHTDATA_DIR, config.NMME_DIR, config.NMME_NC_FILENAME_CRS.format(date=date))

    nmme_ds = rioxarray.open_rasterio(NMME_filepath)
    #nmme's data comes in fractions while other sources come in percentages, so convert to percentages
    nmme_ds[config.NMME_LOWERTERCILE]=nmme_ds[config.NMME_LOWERTERCILE]*100
    nmme_ds = nmme_ds.rename({config.NMME_LON: config.LONGITUDE, config.NMME_LAT: config.LATITUDE,config.NMME_LOWERTERCILE:config.LOWERTERCILE})

    with rasterio.open(NMME_filepath) as src:
        transform = src.transform

    return nmme_ds, transform

def main(country, download, config=None):
    #mainly used for testing, in principle there should be a separate script per country
    if config is None:
        config = Config()
    parameters = config.parameters(country)
    get_iri_data(country, config,parameters, download=download)
    get_icpac_data(config,download=download)

if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country.lower(), args.download_data)