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
# For 2), two main factors have to be decided upon, namely the spatial level of decision making, and the aggregation from raster cells to that spatial level.

# ### The level of decision making
# Ultimately we would like to have a drought-indicator per livelihood zone but this is not realistic due to the high uncertainty in forecasts with a longer lead time.   
# At the minimum we need an indication per admin1.    
# For large admin1 regions with different weather patterns, it would be great to be able to have an indication per admin2 (or weather type region). 

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
# Compute the average value of all cells with their centre in an admin1 region. If this value is higher than 45% probability, activate    
# Compute the maximum value of cells touching an admin2 per admin2. If any of the admin2's have a maximum value that is higher than 47.5%, raise a warning and decide with the team if that area should have an activation

# ### Spatial level questions:
# - Could we look beyond admin regions but instead focus on e.g. climatic regions?
# - Should we take into account the livelihood types?

# ### Aggregation methodology questions:
# - Are there other possible aggregation methodologies than those mentioned above?
# - Which methodology best fits our purpose?
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


# ### Load data and compute aggregations

ds,transform=load_iri(config)


ds


#MAM season
ds_l2=ds.sel(F=cftime.Datetime360Day(2021, 1, 16, 0, 0, 0, 0),L=2)


#threshold of probability of below average, used to compute percentage of area above that probability
probability_threshold=50


#compute the different aggregation methodologies per admin1
df_stats=compute_raster_statistics(adm1_bound_path,ds_l2["prob_below"].values,transform,probability_threshold)


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


# ### ADM2

#compute the different aggregation methodologies per admin1
df_stats_adm2=compute_raster_statistics(adm2_bound_path,ds_l2["prob_below"].values,transform,probability_threshold)


df_stats_adm2[["ADM2_EN"]+stats_cols+stats_perc_cols].style.highlight_max(color = 'orange', axis = 0)


#red = nan values. Occurs in left column if no cell has its center within that region
plot_spatial_columns(df_stats_adm2,stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


#to determine if different patterns within large adm1
plot_spatial_columns(df_stats_adm2[df_stats_adm2.ADM1_EN=="Oromia"],stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


# ### Historical analysis

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
df_hist=alldates_statistics(ds,transform,probability_threshold,percentage_threshold,leadtime,adm1_bound_path)


prob_histograms(df_hist,stats_cols,xlim=(0,60))


prob_histograms(df_hist,stats_perc_cols,xlim=(0,df_hist.perc_threshold.max()))


#binned values of max cell touched on adm1 for all history
pd.cut(df_hist.groupby("date").max()["max_cell_touched"],np.arange(30,60,2.5)).value_counts().sort_index(ascending=False)


#binned values of avg cell on adm1 for all history
pd.cut(df_hist.groupby("date").max()["avg_cell"],np.arange(30,60,2.5)).value_counts().sort_index(ascending=False)


#values of perc threhold on adm1 for all history
df_hist.value_counts("perc_threshold").sort_index(ascending=False)


# #### Historical analysis admin2

df_hist_adm2=alldates_statistics(ds,transform,probability_threshold,percentage_threshold,leadtime,adm2_bound_path)


prob_histograms(df_hist_adm2,stats_cols,xlim=(0,60))


prob_histograms(df_hist_adm2,stats_perc_cols,xlim=(0,df_hist.perc_threshold.max()))


plot_spatial_columns(df_hist_adm2[df_hist_adm2.date=="2020-10-16"],stats_cols,
                title="Analysis of IRI forecast",predef_bins=bins_list)


# December-February 2021 has been one of the driest season forecasted by IRI since 2017. With CHIRPS data from Dec 1st till mid Jan, it seems there is indeed below average rainfall, but in the lower part of ethipia instead of mid/upper part as forecasted    
# CHIRPS data: https://data.chc.ucsb.edu/products/CHIRPS-2.0/moving_12pentad/pngs/africa_east/Anomaly_12PentAccum_Current.png

# ### Past activations 
# If using     
# ADMIN 1 average of all cells with its center inside the region is >=45    
# OR    
# ADMIN 2 maximum value of all cells touching the region is >= 47.5

act_adm1=df_hist[df_hist.avg_cell>=45][["date","forec_valid","ADM1_EN","avg_cell"]]


act_adm1


df_hist_adm2[(df_hist_adm2.max_cell>=47.5) & ~(df_hist_adm2.date.isin(act_adm1.date.values))][["date","forec_valid","ADM1_EN","ADM2_EN","max_cell_touched"]].sort_values(by=["date","ADM1_EN"]).set_index(["date","forec_valid","ADM1_EN","ADM2_EN"])


#for reference, the difference when using max cell touched
df_hist_adm2[(df_hist_adm2.max_cell_touched>=47.5) & ~(df_hist_adm2.date.isin(act_adm1.date.values))][["date","forec_valid","ADM1_EN","ADM2_EN","max_cell_touched"]].sort_values(by=["date","ADM1_EN"]).set_index(["date","forec_valid","ADM1_EN","ADM2_EN"])


# #### Detail questions later in process
# - Do we want to upsample the resolution if using an average or percentage based methodology?   
#     - If so, what is the best upsampling methodology for our use case? Main difference between methodologies is smoother values vs introducing new values. [This](https://gisgeography.com/raster-resampling/) is a nice explanation of the different methods (now using bilinear)
# - How should we define a dry mask?






