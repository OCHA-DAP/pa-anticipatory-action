import requests
import os
import logging
import sys
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats
import pandas as pd
import rioxarray

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from indicators.drought.config import Config
from indicators.drought.utils import parse_args
#cannot have an utils file inside the drought directory, and import from the utils directory.
#So renamed to utils_general but this should maybe be changed
from utils_general.utils import config_logger, auth_googleapi, download_gdrive, unzip

logger = logging.getLogger(__name__)

def invert_latlon(ds):
    #invert lat if this ranges from -90 to 90 instead of 90 to -90
    #inversion of lon still has to be implemented
    #this inversion is done since some functions don't account for this inversion when reading the data and thus produce wrong results
    #this is for example apparent when using xr.open_dataset() to read an array and input it to zonal_stats
    #the lat coordinates should be named "lat" and lon coordinates "lon", so rename those coordinates in the dataset before inputting to this function if needed
    #function largely copied from https://github.com/perrygeo/python-rasterstats/issues/218
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
        da=ds.reindex(lat=ds["lat"][::-1])
    #TODO: implement longitude inversion
    if lon_status=='inverted':
        logger.error("Inverted longitude still needs to be implemented..")
        # da=np.flip(da.squeeze('F'),axis=1)
    return da

def download_iri(iri_auth,config,chunk_size=128):
    #TODO: it would be way nicer to download with opendap instead of requests, since then the file doesn't even have to be saved and is hopefully faster. Only, cannot figure out how to do that with cookie authentication
    IRI_dir = os.path.join(config.DROUGHTDATA_DIR,config.IRI_DIR)
    Path(IRI_dir).mkdir(parents=True, exist_ok=True)
    #TODO: decide if only download if file doesn't exist. Also depends on whether want one IRI file with always newest data, or one IRI file per month
    IRI_filepath = os.path.join(IRI_dir, config.IRI_NC_FILENAME_RAW)
    if os.path.exists(IRI_filepath):
        os.remove(IRI_filepath)
    cookies = {
        '__dlauth_id': iri_auth,
    }
    #TODO fix/understand missing certificate verification warning
    logger.info("Downloading IRI NetCDF file. This might take some time")
    response = requests.get(config.IRI_URL, cookies=cookies, verify=False)
    with open(IRI_filepath, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    #add crs to the nc file
    iri_ds = xr.open_dataset(IRI_filepath, decode_times=False, drop_variables='C')
    IRI_filepath_crs = os.path.join(IRI_dir, config.IRI_NC_FILENAME_CRS)
    if os.path.exists(IRI_filepath_crs):
        os.remove(IRI_filepath_crs)
    #TODO: not sure if these 3 lines fit better here or in get_iri_data
    iri_ds=iri_ds.rename({"X": "lon", "Y": "lat"}) #, "F": "pubdate", "L": "leadtime"})
    iri_ds = invert_latlon(iri_ds)
    iri_ds.rio.write_crs("EPSG:4326").to_netcdf(IRI_filepath_crs)

    #TODO: check range longitude and change [0,360] to [-180,180]



def fix_calendar(ds, timevar='F'):
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def plot_raster_boundaries(ds_nc,country, parameters, config, lon='lon',lat='lat',forec_val='prob'):
    #to circumvent that figures from different functions are overlapping
    plt.figure()
    #TODO: generalize to other providers
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

    # plot forecast and admin boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(32,8))
    ax1.contourf(lons, lats, prob)
    df_adm.boundary.plot(linewidth=1, ax=ax1, color="red")
    ax2.contourf(lons, lats, prob)
    df_world.boundary.plot(linewidth=1, ax=ax2, color="red")
    fig.suptitle(f'{country} {provider} forecasted values and shape boundaries')
    return fig

def plot_spatial_columns(df, col_list, title=None, predef_bins=False,cmap='YlOrRd'):
    #to circumvent that figures from different functions are overlapping
    plt.figure()

    num_plots = len(col_list)
    colp_num = 2
    rows = num_plots // colp_num
    rows += num_plots % colp_num
    position = range(1, num_plots + 1)

    if predef_bins:
        scheme = None
        bins_list = np.arange(30, 70, 5)
        norm2 = mcolors.BoundaryNorm(boundaries=bins_list, ncolors=256)
    else:
        scheme = "natural_breaks"
        norm2 = None

    fig = plt.figure(1, figsize=(16, 6 * rows))

    for i, col in enumerate(col_list):
        ax = fig.add_subplot(rows, colp_num, position[i])

        if not predef_bins:
            colors = len(df[col].dropna().unique())
        else:
            colors = None

        if df[col].isnull().values.all():
            print(f"No not-NaN values for {c}")
        elif df[col].isnull().values.any():
            df.plot(col, ax=ax, legend=True, k=colors, cmap=cmap, norm=norm2, scheme=scheme,
                    missing_kwds={"color": "lightgrey", "edgecolor": "red",
                                  "hatch": "///",
                                  "label": "No values"})
        else:
            df.plot(col, ax=ax, legend=True, k=colors, cmap=cmap, norm=norm2, scheme=scheme)

        df.boundary.plot(linewidth=0.2, ax=ax)

        plt.title(col)
        ax.axis("off")
        if not predef_bins and not df[col].isnull().values.all():
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
    return fig


def get_iri_data(config, download=False):
    if download:
        iri_auth = os.getenv('IRI_AUTH')
        if not iri_auth:
            logger.error("No authentication file found")
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

def download_icpac(config):
    #TODO: would be nicer to directly download from their ftp but couldn't figure out yet how (something with certificates)
    gclient = auth_googleapi()
    gzip_output_file=os.path.join(config.DROUGHTDATA_DIR, f'{config.ICPAC_DIR}.zip')
    download_gdrive(gclient, config.ICPAC_GDRIVE_ZIPID, gzip_output_file)
    unzip(gzip_output_file,config.DROUGHTDATA_DIR)
    os.remove(gzip_output_file)
    for path in Path(os.path.join(config.DROUGHTDATA_DIR,config.ICPAC_DIR)).rglob(config.ICPAC_PROBFORECAST_REGEX):
        icpac_ds = xr.open_dataset(path)
        icpac_ds.rio.write_crs("EPSG:4326").to_netcdf(path)


def get_icpac_data(config,download=False):
    if download:
        download_icpac(config)

    for path in Path(os.path.join(config.DROUGHTDATA_DIR,'icpac')).rglob("ForecastProb*nc"):
        #TODO: merge datasets if there are several
        icpac_ds = xr.open_dataset(path)
        print(icpac_ds)
        #assume all transforms of the different files are the same, so just select the one that is read the latest
        with rasterio.open(path) as src:
            transform = src.transform
    return icpac_ds, transform


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

def main(country, download, config=None):
    if config is None:
        config = Config()
    parameters = config.parameters(country)
    get_iri_data(country, config,parameters, download=download)
    get_icpac_data(config,download=download)

if __name__ == "__main__":
    args = parse_args()
    config_logger(level="info")
    main(args.country.lower(), args.download_data)