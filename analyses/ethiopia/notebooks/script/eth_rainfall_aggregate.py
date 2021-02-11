#!/usr/bin/env python
# coding: utf-8

# # Exploration of seasonal precipitation forecasts
# For anticipatory action on drought, we are exploring indicators that signal the future occurence of a drought. 
# The indicator we are exploring is seasonal precipitation forecasts. More specifically, we look at the probability of below-average precipitation of the probabilistic tercile forecasts. 
# 
# To develop the approach further there are two main aspects we have to better understand:
# 1) How can we combine the forecasts provided by different organizations? \
# 2) How can we aggregate the forecasts from raster cells to the spatial level which is used for decision-making?    
# 
# This highest priority is with 2), and this is what this notebook explores
# 
# For 2), three main factors have to be decided upon, namely     
# - the spatial level of decision making     
# - the aggregation from raster cells to that spatial level     
# - the probability threshold after aggregation.

# ### The level of decision making
# Ultimately we would like to have a drought-indicator per livelihood zone but this is not realistic due to the high uncertainty in forecasts with a longer lead time.   
# At the minimum we need an indication per admin1.    
# For large admin1 regions with different weather patterns, it would be great to be able to have an indication per admin2 (or weather type region).     
# We should also take into account the livelihood types and seasonal calendar to determine the impact of below-average rainfall

# ### Possible methodologies for aggregating to decision level
# Ultimately we want to know which regions meet a certain threshold of probability of below-average precipitation. 
# Whether a region meets this threshold can be computed in different manners. 
# 
# We determine which raster cells fall within an admin region either by selecting those with **their centre** in the region OR by selecting cells that have **any part** inside the admin region. Each raster cell (similar to pixel) has a probability. 
# 
# **Detailed options:**
# 
# *(Each option possible whether cells were selected based on their centre or any parts being in the region.)*
# 
# 1) Set the value of each admin region as the value of the cell with the **maximum probability** and trigger if that value >= *trigger_threshold*   
# 2) Set the value of each admin regtion as the **average value** of all cells included in the admin region and trigger if that value >= *trigger_threshold*      
# 3) Compute the **percentage of the admin region** that has a value higher than *probability_threshold* and trigger if this percentage is larger than *trigger_threshold_percentage*

# ### Our proposal
# **Note: this is a very early stage proposal and meant to guide the discussion**    
# a) Overlay with livelihood types and seasonal calendar to determine the regions impacted by less rainfall    
# b) Compute the average value of all cells with their centre in an admin1 region. If this value is higher than 45% probability, activate       
# c) Compute the maximum value of cells touching an admin2 per admin2. If any of the admin2's have a maximum value that is higher than 47.5%, activate    

# ### Spatial level questions:
# - Could we look beyond admin regions but instead focus on e.g. climatic regions?
# - Should we take into account the livelihood types and if so how?

# ### Aggregation methodology questions:
# - Are there other possible aggregation methodologies than those mentioned above?
# - Which methodology best fits our purpose?
# - What is a reasonable threshold?
# 
# 

# ## CODE
# Implements the above mentioned aggregation methodologies for admin2 and admin 1 level. This is done for the current forecast (2021-01) for the MAM season (2 months leadtime)     
# The final part of the notebook computes the dates the proposed threshold has been met since 2017

# ### Load packages

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.enums import Resampling
import matplotlib.colors as mcolors
import xarray as xr
import cftime
import math
import rioxarray
from shapely.geometry import mapping
import cartopy.crs as ccrs
import matplotlib as mpl


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
print(path_mod)
sys.path.append(path_mod)
from indicators.drought.config import Config


# #### Set config values

country="ethiopia"
config=Config()
parameters = config.parameters(country)
country_folder = os.path.join(config.DIR_PATH, config.ANALYSES_DIR, country)


adm1_bound_path=os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_folder,config.DATA_DIR,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


# #### Define functions

def load_iri(config):    
    IRI_filepath = os.path.join(config.DROUGHTDATA_DIR, config.IRI_DIR, config.IRI_NC_FILENAME_CRS)
    # the nc contains two bands, prob and C. Still not sure what C is used for but couldn't discover useful information in it and will give an error if trying to read both (cause C is also a variable in prob)
    #the date format is formatted as months since 1960. In principle xarray can convert this type of data to datetime, but due to a wrong naming of the calendar variable it cannot do this automatically
    #Thus first load with decode_times=False and then change the calendar variable and decode the months
    iri_ds = xr.open_dataset(IRI_filepath, decode_times=False, drop_variables='C')
    iri_ds['F'].attrs['calendar'] = '360_day'
    iri_ds = xr.decode_cf(iri_ds)

    with rasterio.open(IRI_filepath) as src:
        transform = src.transform
        
    iri_ds=iri_ds.rename({"prob":"prob_below"})
    #C indicates the tercile where 0=below average
    iri_ds_below = iri_ds.sel(C=0)

    return iri_ds_below, transform


def fix_calendar(ds, timevar='F'):
    """
    Some datasets come with a wrong calendar attribute that isn't recognized by xarray
    So map this attribute that can be read by xarray
    Args:
        ds (xarray dataset): dataset of interest
        timevar (str): variable that contains the time in ds

    Returns:
        ds (xarray dataset): modified dataset
    """
    if "calendar" in ds[timevar].attrs.keys():
        if ds[timevar].attrs['calendar'] == '360':
            ds[timevar].attrs['calendar'] = '360_day'
    elif "units" in ds[timevar].attrs.keys():
        if "months since" in ds[timevar].attrs['units']:
            ds[timevar].attrs['calendar'] = '360_day'
    return ds


def resample_raster(file_path,upscale_factor):
    with rasterio.open(file_path) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
    
    data_reshape=data.reshape(-1,data.shape[-1])

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )
    
    return data_reshape, transform


def compute_raster_statistics(boundary_path, raster_array, raster_transform, threshold, band=1, nodata=-9999, upscale_factor=None):
    """
    Compute statistics of the raster_array per geographical region defined in the boundary_path file
    Currently several methods are implemented, namely the maximum and mean per region, and the percentage of the area with a value larger than threshold.
    For all three methods, two variations are implemented: one where all raster cells touching a region are counted, and one where only the raster cells that have their center within the region are counted.
    Args:
        boundary_path (str): path to the shapefile
        raster_array (numpy array): array containing the raster data
        raster_transform (numpy array): array containing the transformation of the raster data, this is related to the CRS
        threshold (float): minimum probability of a raster cell to count that cell as meeting the criterium
        upscale_factor: currently not implemented

    Returns:
        df (Geodataframe): dataframe containing the computed statistics
    """
    df = gpd.read_file(boundary_path)
    #TODO: decide if we want to upsample and if yes, implement
    # if upscale_factor:
    #     forecast_array, transform = resample_raster(raster_path, upscale_factor)
    # else:

    # extract statistics for each polygon. all_touched=True includes all cells that touch a polygon, with all_touched=False only those with the center inside the polygon are counted.
    df["max_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=band, nodata=nodata))["max"]
    df["max_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=band, nodata=nodata))["max"]

    df["avg_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, band=band, nodata=nodata))[
        "mean"]
    df["avg_cell_touched"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=raster_array, affine=raster_transform, all_touched=True, band=band, nodata=nodata))[
        "mean"]

    # calculate the percentage of the area within an geographical area that has a value larger than threshold
    forecast_binary = np.where(raster_array >= threshold, 1, 0)
    bin_zonal = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'], band=band, nodata=nodata))
    df['perc_threshold'] = bin_zonal['sum'] / bin_zonal['count'] * 100
    bin_zonal_touched = pd.DataFrame(
        zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, all_touched=True, stats=['count', 'sum'], band=band, nodata=nodata))
    df['perc_threshold_touched'] = bin_zonal_touched['sum'] / bin_zonal_touched['count'] * 100

    return df


def plot_spatial_columns(df, col_list, title=None, predef_bins=None,cmap='YlOrRd',colp_num=2):
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

    #define the number of columns and rows
    num_plots = len(col_list)
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)

    fig = plt.figure(1,figsize=(16,6*rows))
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


def plot_raster_boundaries_clip(ds_list, boundary_path, clipped=True, lon='lon',lat='lat',forec_val='prob_below',title_list=None,suptitle=None,colp_num=2,predef_bins = np.arange(30, 61, 2.5),figsize=(6.4, 4.8),labelsize=8):
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
        colp_num (int): number of columns in figure
        predef_bins (array/list): array/list with bins to classify the values in
        figsize(tuple): width, height of the figure in inches
        labelsize(float): size of legend labels and subplot titles
    Returns:
        fig (fig): two subplots with the raster and the country and world boundaries
    """
    #initialize empty figure, to circumvent that figures from different functions are overlapping
    plt.figure()

    # load admin boundaries shapefile
    df_bound = gpd.read_file(boundary_path)

    num_plots = len(ds_list)
    rows = math.ceil(num_plots / colp_num)
    position = range(1, num_plots + 1)

    norm = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
    cmap="YlOrRd" #plt.cm.jet
    fig, axes = plt.subplots(rows, colp_num,figsize=figsize)
    if num_plots==1:
        axes.set_axis_off()
    else:
        [axi.set_axis_off() for axi in axes.ravel()]

    for i, ds in enumerate(ds_list):
        #TODO: decide if want to use projection and if Robinson is then good one
        ax = fig.add_subplot(rows, colp_num, position[i],projection=ccrs.Robinson())

        if clipped:
            ds = ds.rio.set_spatial_dims(x_dim=lon,y_dim=lat).rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
        lons = ds.coords[lon]
        lats = ds.coords[lat]
        prob = ds[forec_val]

        im = plt.pcolormesh(lons, lats, prob, cmap=cmap, norm=norm)

        if title_list is not None:
            plt.title(title_list[i], size=labelsize)

        df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
        ax.axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label(forec_val, size=labelsize, rotation=90, labelpad=15)
    cb.ax.tick_params(labelsize=labelsize)

    if suptitle is not None:
        fig.suptitle(suptitle, size=10)
    return fig


# ### Load data and interpolate

ds,transform=load_iri(config)


ds


# Interpolate the values to get a more granular grid. This gives a more realistic idea of the actual forecast when using an average or percentage-based aggregation. 
# On boundaries of two raster cells, the interpolation doesn't always seem entirely correct. I don't know what this is caused by and [asked the StackOverflow](https://stackoverflow.com/questions/65934263/how-does-xarrays-interp-nearest-method-choose-the-nearest-center) community but without success so far. 
# 

def interpolate_ds(ds,transform,upscale_factor):
    # Interpolated data
    new_lon = np.linspace(ds.lon[0], ds.lon[-1], ds.dims["lon"] * upscale_factor)
    new_lat = np.linspace(ds.lat[0], ds.lat[-1], ds.dims["lat"] * upscale_factor)

    #choose nearest as interpolation method to assure no new values are introduced but instead old values are divided into smaller raster cells
    dsi = ds.interp(lat=new_lat, lon=new_lon,method="nearest")
    transform_interp=transform*transform.scale(len(ds.lon)/len(dsi.lon),len(ds.lat)/len(dsi.lat))
    
    return dsi, transform_interp


ds_interp,transform_interp=interpolate_ds(ds,transform,4)


df_bound=gpd.read_file(adm1_bound_path)
ds_interp_clip = ds_interp.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)


#inspired from https://xarray-contrib.github.io/xarray-tutorial/scipy-tutorial/04_plotting_and_visualization.html#facet
ds_interp_clip_l2=ds_interp_clip.sel(L=2)
g=ds_interp_clip_l2.prob_below.plot(
    col="F",
    col_wrap=4,
#     row="L",
    cmap=mpl.cm.YlOrRd, #mpl.cm.RdORYlBu_r,
#     robust=True,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")


# ### Compute aggregations for 2021 MAM season
# Focus on one date first for experimentation

#threshold of probability of below average, used to compute percentage of area above that probability
probability_threshold=50


#MAM season
ds_l2=ds.sel(F=cftime.Datetime360Day(2021, 1, 16, 0, 0, 0, 0),L=2)
ds_l2_interp=ds_interp.sel(F=cftime.Datetime360Day(2021, 1, 16, 0, 0, 0, 0),L=2)


fig=plot_raster_boundaries_clip([ds_l2,ds_l2_interp],adm1_bound_path,colp_num=2,predef_bins=np.arange(30,70,2.5),figsize=(32,9),suptitle="Raw data of Jan 2021 forecast for MAM",title_list=["Original","Interpolated"])


#compute the different aggregation methodologies per admin1
df_stats=compute_raster_statistics(adm1_bound_path,ds_l2_interp["prob_below"].values,transform_interp,probability_threshold)


stats_cols=['max_cell', 'max_cell_touched', 'avg_cell', 'avg_cell_touched']
stats_perc_cols = ['perc_threshold', 'perc_threshold_touched']
bins_list=np.arange(30,70,2.5)
bins_perc_list=np.arange(0,50,5)


#highlight the max values
df_stats[["ADM1_EN"]+stats_cols+stats_perc_cols].style.highlight_max(color = 'orange', axis = 0)


#plot the value per region
plot_spatial_columns(df_stats,stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


plot_spatial_columns(df_stats,stats_perc_cols,
                title="Analysis of IRI forecast",predef_bins=bins_perc_list)


# #### Aggregations on ADM2 level

#compute the different aggregation methodologies per admin1
df_stats_adm2=compute_raster_statistics(adm2_bound_path,ds_l2_interp["prob_below"].values,transform_interp,probability_threshold)


#orange is max values
# df_stats_adm2[["ADM2_EN"]+stats_cols+stats_perc_cols].style.highlight_max(color = 'orange', axis = 0)


#red = nan values. Occurs in left column if no cell has its center within that region
plot_spatial_columns(df_stats_adm2,stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


#to determine if different patterns within large adm1
plot_spatial_columns(df_stats_adm2[df_stats_adm2.ADM1_EN=="Oromia"],stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


# ### Aggregation for all historical data
# We have data from March 2017

probability_threshold=50
percentage_threshold=20
leadtime=2


def prob_histograms(df,col_list,ylim=None,xlim=None):
    #plot histogram for each entry in col_list
    fig = plt.figure(1,figsize=(16,4))

    for i,col in enumerate(col_list):
        ax = fig.add_subplot(1,len(col_list),i+1)
        df[col].plot.hist(bins=20,ax=ax)
        plt.title(f"Histogram {col} \n NaN values: {df[col].isna().sum()}")
        ax.set_xlabel(col)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(df[col_list].min().min()-5,df[col_list].max().max()+5)
        if ylim:
            ax.set_ylim(ylim[0],ylim[1])
        if xlim:
            ax.set_xlim(xlim[0],xlim[1])


def alldates_statistics(ds,transform,prob_threshold,perc_threshold,leadtime,adm_path):
    #compute statistics on level in adm_path for all dates in ds
    df_list=[]
    ds=ds.sel(L=leadtime)
    print(f"Number of forecasts: {len(ds.F)}")
    for date in ds.F.values:
        ds_date=ds.sel(F=date)
        df=compute_raster_statistics(adm_path,ds_date["prob_below"].values,transform,prob_threshold)
        df["date"]=pd.to_datetime(date.strftime("%Y-%m-%d"))
        df_list.append(df)
    df_hist=pd.concat(df_list)
    df_hist=df_hist.sort_values(by="date")

    for m in ["max_cell","max_cell_touched","avg_cell","avg_cell_touched"]:
        df_hist[f"{m}_ge{prob_threshold}"]=np.where(df_hist[m]>=prob_threshold,1,0)
    df_hist[f"perc_threshold_ge{perc_threshold}"]=np.where(df_hist["perc_threshold"]>=perc_threshold,1,0)
    df_hist[f"perc_threshold_touched_ge{perc_threshold}"]=np.where(df_hist["perc_threshold_touched"]>=perc_threshold,1,0)
    
    df_hist["date_str"]=df_hist["date"].dt.strftime("%Y-%m")
    df_hist["pred_date"]=df_hist.date+pd.offsets.DateOffset(months=leadtime)
    #valid for period of 3 months, i.e. till 2 months later than first month
    df_hist["pred_date_end"]=df_hist.date+pd.offsets.DateOffset(months=leadtime+2)
    df_hist["pred_date_form"]=df_hist["pred_date"].apply(lambda x: x.strftime("%b"))
    df_hist["pred_date_end_form"]=df_hist["pred_date_end"].apply(lambda x: x.strftime("%b %Y"))
    df_hist["forec_valid"]=df_hist[["pred_date_form","pred_date_end_form"]].agg(" - ".join,axis=1)
    
    return df_hist


#compute aggregated values on adm1 for all dates since Jan 2017
df_hist=alldates_statistics(ds_interp,transform_interp,probability_threshold,percentage_threshold,leadtime,adm1_bound_path)


df_hist[df_hist.date=="2021-01-16"][["ADM1_EN"]+stats_cols+stats_perc_cols]


#highlight the max values
df_stats[["ADM1_EN"]+stats_cols+stats_perc_cols].style.highlight_max(color = 'orange', axis = 0)


prob_histograms(df_hist,stats_cols,xlim=(0,60))


prob_histograms(df_hist,stats_perc_cols,xlim=(0,df_hist.perc_threshold.max()))


#binned values of max cell touched on adm1 for all history
pd.cut(df_hist.groupby("date").max()["max_cell_touched"],np.arange(30,60,2.5)).value_counts().sort_index(ascending=False)


#binned values of avg cell on adm1 for all history
pd.cut(df_hist.groupby("date").max()["avg_cell"],np.arange(30,60,2.5)).value_counts().sort_index(ascending=False)


#values of perc threhold on adm1 for all history
df_hist.value_counts("perc_threshold").sort_index(ascending=False)


# #### Historical analysis admin2

df_hist_adm2=alldates_statistics(ds_interp,transform_interp,probability_threshold,percentage_threshold,leadtime,adm2_bound_path)


prob_histograms(df_hist_adm2,stats_cols,xlim=(0,60))


prob_histograms(df_hist_adm2,stats_perc_cols,xlim=(0,df_hist.perc_threshold.max()))


plot_spatial_columns(df_hist_adm2[df_hist_adm2.date=="2020-10-16"],stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


# December-February 2021 has been one of the driest season forecasted by IRI since 2017. With CHIRPS data from Dec 1st till mid Jan, it seems there is indeed below average rainfall, but in the lower part of ethipia instead of mid/upper part as forecasted    
# CHIRPS data: https://data.chc.ucsb.edu/products/CHIRPS-2.0/moving_12pentad/pngs/africa_east/Anomaly_12PentAccum_Current.png

# ### Test thresholds
# Test different possibilities for the trigger threshold and analyze on which historical dates, the trigger would be met

# #### Option 1: the area with a below average probability of 50% of larger is at least 1 % of the total area

df_hist[df_hist.perc_threshold>0][["date","forec_valid","ADM1_EN","perc_threshold"]].set_index(["date","forec_valid","ADM1_EN"])


# #### Option 2  
# On ADMIN 1 level the average of all cells with a value larger than 40, is larger than 45
# OR    
# On ADMIN 2 level the maximum value of all cells within the region is >= 47.5
# 
# Variations on this idea are to not first select cells that are larger than 40, and to use the maximum value of cells touching a region instead of having their centre within the region

#select only the cells with a value larger than 40
ds_interp_40=ds_interp.where(ds_interp.prob_below>=40)


#example of cells above 40
fig=plot_raster_boundaries_clip([ds_interp.sel(F=cftime.Datetime360Day(2021, 1, 16, 0, 0, 0, 0),L=2),ds_interp_40.sel(F=cftime.Datetime360Day(2021, 1, 16, 0, 0, 0, 0),L=2)],adm1_bound_path,colp_num=2,predef_bins=np.arange(30,70,2.5),figsize=(19,6),title_list=["All cells","Cells value >=40"],suptitle="Raw data of Jan 2021 forecast for MAM")


#inspired from https://xarray-contrib.github.io/xarray-tutorial/scipy-tutorial/04_plotting_and_visualization.html#facet
ds_interp_40_l2=ds_interp_40.sel(L=2)
g=ds_interp_40_l2.prob_below.plot(
    col="F",
    col_wrap=4,
#     row="L",
    cmap=mpl.cm.YlOrRd, #mpl.cm.RdORYlBu_r,
#     robust=True,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
    },
)
df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")


#compute aggregated values on adm1 for all dates since Jan 2017
df_hist_40=alldates_statistics(ds_interp_40,transform_interp,probability_threshold,percentage_threshold,leadtime,adm1_bound_path)


act_adm1_40=df_hist_40[df_hist_40.avg_cell>=45][["date","forec_valid","ADM1_EN","avg_cell"]]


act_adm1_40.sort_values(by=["date","ADM1_EN"]).set_index(["date","forec_valid","ADM1_EN"])


#Date-ADM1 combinations that are not in the avg>=40 list, but where the maximum value of the cells within an admin2 is >= 47.5
act_adm1_tuples = pd.MultiIndex.from_frame(act_adm1_40[["date","ADM1_EN"]])
df_hist_adm2[(df_hist_adm2.max_cell>=47.5) & ~pd.MultiIndex.from_frame(df_hist_adm2[["date","ADM1_EN"]]).isin(act_adm1_tuples)][["date","forec_valid","ADM1_EN","ADM2_EN","max_cell"]].sort_values(by=["date","ADM1_EN"]).set_index(["date","forec_valid","ADM1_EN","ADM2_EN"])


# None of the two methods would trigger now, while the consensus that the coming MAM season is expected to have below average rainfall that might cause drought.. 

df_hist[df_hist.forec_valid=="Mar - May 2021"][["date","forec_valid","ADM1_EN","perc_threshold","perc_threshold_touched"]+stats_cols]


# ### Analyze CHIRPS data and correlation forecasts

from indicators.drought.chirps_rainfallobservations import get_chirps_data


#years to load data for
years=range(1982,2021)
#years used to compute average "climatology"
#should maybe take years more back in the past
years_climate=slice('1982-01-01', '2010-12-31') #slice('2000-01-01', '2016-12-31')


chirps_dir = os.path.join(config.DROUGHTDATA_DIR, config.CHIRPS_DIR)





#load all chirps data
years=range(1982,2021)
dict_ds={}
for i in years:
    ds,transform = get_chirps_data(config, i,download=True)
    ds_year=ds.groupby("time.year").sum(dim='time').rename({'year':'time'})
#     df_bound = gpd.read_file(adm1_bound_path)
#     #clip global to ethiopia to speed up calculating rolling sum
#     #for some unexplainable reason chirps of 1995 doesn't want to include the CRS when saving it to a file, so add write_crs here...
#     ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    dict_ds[i]=ds_year
ds_year_all=xr.merge([dict_ds[i] for i in years])


ds_year_all.to_netcdf("../Data/CHIRPS_19822020_yearsum.nc")





#load all chirps data
years=range(1982,2021)
dict_ds={}
for i in years:
    ds,transform = get_chirps_data(config, i,download=True)
    df_bound = gpd.read_file(adm1_bound_path)
    #clip global to ethiopia to speed up calculating rolling sum
    #for some unexplainable reason chirps of 1995 doesn't want to include the CRS when saving it to a file, so add write_crs here...
    ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    dict_ds[i]=ds
ds_all=xr.merge([dict_ds[i] for i in years])


ds_all.to_netcdf("../Data/CHIRPS_19822020.nc")


#load all chirps data
years=range(1982,2021)
dict_ds={}
for i in years:
    ds,transform = get_chirps_data(config, i,download=True)
    df_bound = gpd.read_file(adm1_bound_path)
    #clip global to ethiopia to speed up calculating rolling sum
    #for some unexplainable reason chirps of 1995 doesn't want to include the CRS when saving it to a file, so add write_crs here...
    ds_clip = ds.rio.set_spatial_dims(x_dim=config.LONGITUDE, y_dim=config.LATITUDE).rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping), df_bound.crs, all_touched=True)
    dict_ds[i]=ds_clip
ds_all=xr.merge([dict_ds[i] for i in years])


# ds_all.to_netcdf("../Data/CHIRPS_merged_eth.nc")


#for each combination of 3 months (refered to season), compute the climatological average and the sum for each year
#key refers to start month of season
seasons={1:"JFM",2:"FMA",3:"MAM",4:"AMJ",5:"MJJ",6:"JJA",7:"JAS",8:"ASO",9:"SON",10:"OND",11:"NDJ",12:"DJF"}
seas_len=3
dict_ds_allseas={v:{} for v in seasons.values()}

for k,v in seasons.items():
    #months within the season
    months=[p%12 if p%12!=0 else 12 for p in range(k,k+seas_len)]
    ds_months=ds_all.sel(time=ds_all.time.dt.month.isin([months]))
    #total precipitation for season per year
    dict_ds_allseas[v]["year"]=ds_months.groupby('time.year').sum(dim='time').rename({'year':'time'})
    ds_months_climate=dict_ds_allseas[v]["year"].sel(time=years_climate)
    #average total precipitation for season
    dict_ds_allseas[v]["avg"]=ds_months_climate.mean(dim='time')
    #anomaly in total precipitation per year for season
    dict_ds_allseas[v]["anom"]=dict_ds_allseas[v]["year"]-dict_ds_allseas[v]["avg"]


dict_ds_allseas["MAM"]["anom"]


for v in seasons.values():
    dict_ds_allseas[v]["anom"]=dict_ds_allseas[v]["anom"].expand_dims({"seas":[v]})
    #select data where precipitation is below average
    dict_ds_allseas[v]["belowavg"]=dict_ds_allseas[v]["anom"].where(dict_ds_allseas[v]["anom"].precip<0).dropna("time",how="all")
#     dict_ds_allseas[v]["sel"]["seas_time"]=[f"{v} {y}" for y in dict_ds_allseas[v]["sel"]["time"].values]


ds_belowavg=xr.merge([dict_ds_allseas[v]["belowavg"] for v in dict_ds_allseas.keys()])
ds_belowavg=ds_belowavg.reindex(seas=list(seasons.values()))


#check anomaly values
np.unique(ds_belowavg.precip.values.flatten()[~np.isnan(ds_belowavg.precip.values.flatten())])


#select only years IRI forecasts available
ds_belowavg_sel=ds_belowavg.sel(time=slice(2010,2020))


g=ds_belowavg_sel.precip.plot(
    col="seas",
    row="time",
#     cmap=mpl.cm.RdYlBu_r,
#     robust=True,
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.8,
        "aspect": 40,
        "pad": 0.1,
        "label":"Anomaly total precipitation (mm)"
    },
#     figsize=(100,20)
)

df_bound = gpd.read_file(adm1_bound_path)
for ax in g.axes.flat:
    df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
    ax.axis("off")


# g.fig.savefig('../results/drought/chirps_anom.png', bbox_inches='tight')


# From the CHIRPS plot we can see that for example FMA 2019 had quite some below average rainfall, so check how the forecasts were for that season (with two months leadtime). 
# 
# From there we can see that the forecast didn't indicate a very high probability of below average rainfall. The admin with the highest probability of below average rainfall was SNNP and this does correspond with the rainfall pattern

df_hist[df_hist.forec_valid=="Feb - Apr 2019"][["date","forec_valid","ADM1_EN","perc_threshold","perc_threshold_touched"]+stats_cols]


# ### RPSS score
# Compute RPSS score per 3-month period for a given leadtime

leadtime


rpss_ds = xr.open_dataset(f"http://iridl.ldeo.columbia.edu/home/.jingyuan/.NMME_seasonal_hindcast_verification/.monthly_RPSS_seasonal_hindcast_precip_ELR/.lead{leadtime}/RPSS/dods",decode_times=False,)
rpss_ds=rpss_ds.rename({"X":"lon","Y":"lat"})
rpss_ds=fix_calendar(rpss_ds,timevar="T")
rpss_ds = xr.decode_cf(rpss_ds)


rpss_ds_clipped=rpss_ds.rio.set_spatial_dims(x_dim="lon",y_dim="lat").rio.write_crs("EPSG:4326").rio.clip(df_bound.geometry.apply(mapping),df_bound.crs,all_touched=True)


rpss_ds_clipped


rpss_ds_list_clipped=[rpss_ds_clipped.sel(T=m) for m in rpss_ds_clipped["T"]]
rpss_bins_clipped=np.linspace(rpss_ds_clipped["RPSS"].min(),rpss_ds_clipped["RPSS"].max(),10)
seasons={1:"JFM",2:"FMA",3:"MAM",4:"AMJ",5:"MJJ",6:"JJA",7:"JAS",8:"ASO",9:"SON",10:"OND",11:"NDJ",12:"DJF"}
list_seasons=[(m+leadtime)%12 if (m+leadtime)%12!=0 else 12 for m in rpss_ds_clipped["T"].dt.month.values]
title_list_clipped=[f'{seasons[s]} (issued in {m})' for s,m in zip(list_seasons,rpss_ds_clipped["T"].dt.strftime("%b").values)]


#RPSS per season
fig_clip = plot_raster_boundaries_clip(rpss_ds_list_clipped, adm1_bound_path, figsize=(20,10),title_list=title_list_clipped, forec_val="RPSS", colp_num=4,clipped=False,predef_bins=rpss_bins_clipped)


# #### Experimentation

# #this si not correct yet!
# from datetime import timedelta
# seasons={1:"JFM",2:"FMA",3:"MAM",4:"AMJ",5:"MJJ",6:"JJA",7:"JAS",8:"ASO",9:"SON",10:"OND",11:"NDJ",12:"DJF"}
# ds_interp_clip.expand_dims({"seas":[seasons[(m+d)%12] if (m+d)%12!=0 else seasons[12] for m in ds_interp_clip.F.dt.month.values for d in ds_interp_clip.L.values]})
# ds_interp_clip_l2=ds_interp_clip.sel(L=2).expand_dims({"seas":[f"{seasons[(m.dt.month.values+2)%12]} {m.dt.year.values}" if (m.dt.month.values+2)%12!=0 else seasons[12] for m in ds_interp_clip.F]})


# g=ds_interp_clip.sel(F=cftime.Datetime360Day(2018, 12, 16, 0, 0, 0, 0)).prob_below.plot(
#     col="L",
#     col_wrap=4,
# #     row="L",
#     cmap=mpl.cm.YlOrRd, #mpl.cm.RdORYlBu_r,
# #     robust=True,
#     cbar_kwargs={
#         "orientation": "horizontal",
#         "shrink": 0.8,
#         "aspect": 40,
#         "pad": 0.1,
#     },
# )
# df_bound = gpd.read_file(adm1_bound_path)
# for ax in g.axes.flat:
#     df_bound.boundary.plot(linewidth=1, ax=ax, color="red")
#     ax.axis("off")

