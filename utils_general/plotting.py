import os
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
#TODO: drought config used for world shapefile boundary path.. Used for plot_raster_boundaries which is mainly debugging function. In future world shp boundary path should be given as input to the function
from indicators.drought.config import Config

logger = logging.getLogger(__name__)


def plot_raster_boundaries(ds_nc,country, parameters, config, lon='lon',lat='lat',forec_val='prob_below'):
    #TODO: at some point this function can be deleted/integrated with plot_raster_boundaries_test, but for now keeping for debugging
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

def plot_raster_boundaries_clip(ds_list, boundary_path, clipped=True, lon='lon',lat='lat',forec_val='prob_below',title_list=None,suptitle=None):
    #compared to plot_raster_boundaries, this function is working with clipped values and a list of datasets
    """
    Plot a raster file and a shapefile on top of each other.
    Several datasets can given and for each dataset a separate subplot will be generated
    Args:
        ds_nc (xarray dataset): dataset that should be plotted, all bands contained in this dataset will be plotted
        boundary_path (str): path to the shapefile
        clipped (bool): if True, clip the raster extent to the shapefile
        lon (str): name of the longitude coordinate in ds_nc
        lat (str): name of the latitude coordinate in ds_nc
        forec_val (str): name of the variable that contains the values to be plotted in ds_nc
        title_list (list of strs): titles of subplots. If None, no subtitles will be used
        suptitle (str): general figure title. If None, no title will be used
    Returns:
        fig (fig): two subplots with the raster and the country and world boundaries
    """
    #initialize empty figure, to circumvent that figures from different functions are overlapping
    plt.figure()

    # load admin boundaries shapefile
    df_bound = gpd.read_file(boundary_path)


    num_plots = len(ds_list)
    if num_plots>1:
        colp_num=2
    else:
        colp_num=1
    rows = num_plots // colp_num
    rows += num_plots % colp_num
    position = range(1, num_plots + 1)

    #TODO: find a better way to define bins that always capture full range of values but are consistent
    predef_bins = np.arange(30, 61, 2.5)
    norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    cmap=plt.cm.jet
    fig, axes = plt.subplots(rows, colp_num)
    if num_plots==1:
        axes.set_axis_off()
    else:
        [axi.set_axis_off() for axi in axes.ravel()]

    for i, ds in enumerate(ds_list):
        #TODO: decide if want to use projection and if Robinson is then good one
        ax = fig.add_subplot(rows, colp_num, position[i],projection=ccrs.Robinson())

        #TODO: spatial_dims needed for ICPAC data but don't understand why yet
        if clipped:
            ds = ds.rio.set_spatial_dims(x_dim=lon,y_dim=lat).rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
        lons = ds.coords[lon]
        lats = ds.coords[lat]
        prob = ds[forec_val]

        im = plt.pcolormesh(lons, lats, prob, cmap=cmap, norm=norm)

        if title_list is not None:
            plt.title(title_list[i], size=8)

        df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label('prob', size=12, rotation=0, labelpad=15)
    cb.ax.tick_params(labelsize=10)

    if suptitle is not None:
        fig.suptitle(suptitle, size=10)
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
    if title:
        fig.suptitle(title, fontsize=14, y=0.92)
    fig.tight_layout()

    return fig