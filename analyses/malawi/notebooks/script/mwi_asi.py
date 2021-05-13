#!/usr/bin/env python
# coding: utf-8

# ### Agricultural Stress Index (ASI)
# Exploration of ASI for Malawi and its correlation with dry spells.  
# Note that the ASI is not specifically callibrated for Malawi and thus might not result in the most accurate capturing of the actual agricultural stress
# 
# Idea: look at the ASI at the first dekad after a dry spell

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
import seaborn as sns


from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
print(path_mod)
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_url


def plot_geo_var(df, sub_col, var_col, title=None, predef_bins=None, colp_num=2, cmap='YlOrRd',figsize=(16,9)):
    #plots one legend for all subplots. Version in eth notebook plots legend per subplot
    """
    Plot the values of "col" for the dates present in df_trig
    If giving predef_bins then the data will be colored according to the bins, else a different colour will be assigned to each unique value in the data for each date
    df: DataFrame containing all the data of all regions
    df_trig: DataFrame containing the dates for which plots should be shown (generally those dates that the trigger is met)
    col: string with column to plot
    shape_path: relative path to the admin1 shapefile
    title: string with title of whole figure (so not the subplots)
    predef_bins: list with bin values
    """

    num_plots = len(df[sub_col].unique())
    if num_plots==1:
        colp_num=1
    rows = math.ceil(num_plots / colp_num)

    position = range(1, num_plots + 1)
    fig, axs = plt.subplots(rows,colp_num,figsize=figsize, 
                        facecolor='w',
                        constrained_layout=True, 
                        sharex=True, sharey=True, 
                        subplot_kw=dict(aspect='equal'))

    axs = axs.ravel()

    if predef_bins is not None:
        scheme = None
        norm2 = mcolors.BoundaryNorm(boundaries=predef_bins, ncolors=256)
        legend_kwds=None
    else:
        scheme = "natural_breaks"
        norm2 = None
        legend_kwds = {'bbox_to_anchor': (1.6, 1)}


    for i, c in enumerate(df[sub_col].unique()):
        df_sub = df[df[sub_col] == c]
        #if no predef bins, set unique color for each unique value
        if predef_bins is None:
            colors = len(df_sub[var_col].dropna().unique())
        #else colors will be determined by norm and cmap
        else:
            colors = None

        if df_sub[var_col].isnull().values.all():
            print(f"No not-NaN values for {c}")
        elif df_sub[var_col].isnull().values.any():
            df_sub.plot(var_col, ax=axs[i], cmap=cmap, k=colors, norm=norm2, legend=False, scheme=scheme,
                        missing_kwds={"color": "lightgrey", "edgecolor": "red",
                                      "hatch": "///",
                                      "label": "Missing values"})
        else:
            df_sub.plot(var_col, ax=axs[i], cmap=cmap, k=colors, norm=norm2, legend=False, scheme=scheme)
        df.boundary.plot(linewidth=0.2, ax=axs[i])

        axs[i].axis("off")
        axs[i].set_title(c)
    patch_col = axs[0].collections[0]
    cb = fig.colorbar(patch_col, ax=axs, shrink=0.5)

        
    if title:
        fig.suptitle(title, fontsize=14, y=0.92)

    return fig


def get_new_name(name, n_dict):
    """
    Return the values of a dict if name is in the keys of the dict
    Args:
        name: string of interest
        n_dict: dict with possibly "name" as key

    Returns:

    """
    if name in n_dict.keys():
        return n_dict[name]
    else:
        return name


# #### Set config values

country="malawi"
config=Config()
parameters = config.parameters(country)
country_dir = os.path.join(config.DIR_PATH, "analyses", country)
country_data_raw_dir = os.path.join(config.DATA_DIR, 'raw', country)
country_data_processed_dir = os.path.join(config.DATA_DIR, 'processed', country)
country_data_exploration_dir = os.path.join(config.DATA_DIR,"exploration",country)


adm1_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin1_shp"])
adm2_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters["path_admin2_shp"])


#annual summary
asi_url_annual=f"http://www.fao.org/giews/earthobservation/asis/data/country/{parameters['iso3_code']}/MAP_ASI/DATA/ASI_AnnualSummary_Season1_data.csv"
##dekad summary
asi_url_dekad=f"http://www.fao.org/giews/earthobservation/asis/data/country/{parameters['iso3_code']}/MAP_ASI/DATA/ASI_Dekad_Season1_data.csv"


# #only needed if not downloaded yet
# download_url(asi_url_annual,os.path.join(country_data_exploration_dir,"ASI",f"{country}_asi_annual.csv"))
# download_url(asi_url_dekad,os.path.join(country_data_exploration_dir,"ASI",f"{country}_asi_dekad.csv"))


df_bound = gpd.read_file(adm1_bound_path)


# ### Per year
# For dry spells we probably need to use the dekad data, but to get an idea of the values
# ASI is only reported at ADM1 so need to see how we map that with the ADM2 dry spell data

df=pd.read_csv(os.path.join(country_data_exploration_dir,"ASI",f"{country}_asi_annual.csv"))
df.columns = map(str.lower, df.columns)


df


df[df.year==2019]


df.province.unique()


df_bound.ADM1_EN.unique()


#map ASI adm1 names to shapefile adm1 names
province_mapping={"Central Region":"Central","Northern Region":"Northern","Southern Region":"Southern"}


df["province"]=df["province"].apply(
            lambda x: get_new_name(x, province_mapping))


dfg=df_bound[["ADM1_EN","geometry"]].merge(df,left_on="ADM1_EN",right_on="province")


#distribution of values across the adm1 regions
fig, ax = plt.subplots(1, 1,figsize=(35,10))
g=sns.boxplot(x="province", y="data",data=dfg, ax=ax)
plt.xticks(rotation=90);


fig=plot_geo_var(dfg.sort_values("year"),"year","data",predef_bins=np.arange(0,100,10),colp_num=9,figsize=(20,20))


# ### Per dekad
# Now only selecting the third dekad, but have to figure a way how we want to connect it with dry spells. Possibly looking at the ASI at the end of a dry spell? Or ASI at the end of a dekad and how many adm2's within an adm1 were experiencing a dry spell during that time?

dfd=pd.read_csv(os.path.join(country_data_exploration_dir,"ASI",f"{country}_asi_dekad.csv"))
dfd.columns = map(str.lower, dfd.columns)
dfd["date"]=pd.to_datetime(dfd.date)


dfd[(dfd.year==2019)&(dfd.dekad==3)].head()


#map ASI adm1 names to shapefile adm1 names
province_mapping={"Central Region":"Central","Northern Region":"Northern","Southern Region":"Southern"}
dfd["province"]=dfd["province"].apply(
            lambda x: get_new_name(x, province_mapping))


dfdg=df_bound[["ADM1_EN","geometry"]].merge(dfd,left_on="ADM1_EN",right_on="province")


#distribution of values across the adm1 regions
fig, ax = plt.subplots(1, 1,figsize=(35,10))
g=sns.boxplot(x="province", y="data",data=dfdg, ax=ax)
plt.xticks(rotation=90);


# #### Load dry spells

df_ds=pd.read_csv(os.path.join(country_data_exploration_dir,"dryspells","mwi_dry_spells_list.csv")) #"../Data/transformed/mwi_dry_spells_list.csv")


df_ds["dry_spell_first_date"]=pd.to_datetime(df_ds["dry_spell_first_date"])
df_ds["dry_spell_confirmation"]=pd.to_datetime(df_ds["dry_spell_confirmation"])
df_ds["dry_spell_last_date"]=pd.to_datetime(df_ds["dry_spell_last_date"])
# df_ds["ds_conf_m"]=df_ds.dry_spell_confirmation.dt.to_period("M")


df_ds.head()


df_bound_adm2=gpd.read_file(adm2_bound_path)


#add ADMIN1 regions
df_ds=df_ds.merge(df_bound_adm2[["ADM2_EN","ADM1_EN"]],how="left",on="ADM2_EN")


df_ds


df_ds_adm1=df_ds[["dry_spell_last_date","ADM1_EN","ADM2_EN"]].groupby(["dry_spell_last_date","ADM1_EN"],as_index=False).count()
df_ds_adm1.rename(columns={"ADM2_EN":"num_ADM2"},inplace=True)


df_ds_adm1


#merge the dry spells with the info if a month had below average rainfall
#merge on outer such that all dates present in one of the two are included
df_comb=df_ds_adm1.merge(dfdg,how="outer",left_on=["dry_spell_last_date","ADM1_EN"],right_on=["date","ADM1_EN"])


len(dfdg.date.unique())


1,11,21


def compute_dekad(date_col):
    day=date_col.day
    if day >1 and day <=11:
        dekad=2
        dekad_day=11
    elif day >11 and day <=21:
        dekad=3
        dekad_day=21
    elif day >21 or day==1:
        dekad=1
        dekad_day=1
        dekad_date=pd.to_datetime(f"{date_col.year}-{date_col.month+1}-{dekad_day}")
    return dekad


df_ds_adm1


df_ds_adm1["dekad"]=df_ds_adm1["dry_spell_last_date"].apply(lambda x: compute_dekad(x))


dfdg.date.dt.day.unique()


df_comb


#dates that are not present in the dry spell list, but are in the observed rainfall df, thus have no dry spells
df_comb.dry_spell_confirmation=df_comb.dry_spell_confirmation.replace(np.nan,0)


#contigency matrix rainfall and dry spells for all months
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

y_target =    df_comb["dry_spell_confirmation"]
y_predicted = df_comb["min_cell_touched_se0.33"]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted)
print(cm)

fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True) #,class_names=["No","Yes"])
ax.set_ylabel("Dry spell in ADMIN2 during month")
ax.set_xlabel("Lower tercile precipitation in ADMIN2 during month")
plt.show()

