import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import logging
from datetime import timedelta
import rioxarray

from pathlib import Path
import sys
import os

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[2]}/"
sys.path.append(path_mod)
from src.indicators.drought.config import Config
from src.utils_general.utils import download_ftp

logger = logging.getLogger(__name__)

# #### Set config values

#adm level to aggregate the raster cells to 
adm_level="adm2"#"adm2" #adm1
days_ahead=5 #15
if adm_level=="adm1":
    adm_col="ADM1_EN"
if adm_level=="adm2":
    adm_col="ADM2_EN"


country="malawi"
config=Config()
parameters = config.parameters(country)
country_iso3=parameters["iso3_code"]
data_public_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR
country_data_raw_dir = data_public_dir / config.RAW_DIR / country_iso3
country_data_processed_dir = os.path.join(config.DATA_DIR,config.PUBLIC_DIR, config.PROCESSED_DIR,country_iso3)
#TODO: this is the old dir, to be changed
chirpsgefs_dir = data_public_dir / config.RAW_DIR / config.GLOBAL_ISO3 / "chirps_gefs"

def compute_raster_stats(ds, raster_transform, gdf,gdf_id_col,ds_thresh_list=[2,4,5,10,15,20,25,30,35,40,45,50]):
    """
    compute statistics of raster data, ds per polygon in gdf
    Args:
        ds: xarray dataset
        raster_transform: affine transformation of ds
        gdf: geodataframe which indicates the shape boundaries
        ds_thresh_list: list of thresholds to compute percentage of cells below the given threshold for
    """
    df = gdf.loc[:,[gdf_id_col,"geometry"]]
    df.loc[:,"max_cell"] = pd.DataFrame(
        zonal_stats(vectors=gdf, raster=ds.values, affine=raster_transform, nodata=np.nan))["max"]
    df.loc[:,"min_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["min"]
    df.loc[:,"mean_cell"] = pd.DataFrame(
        zonal_stats(vectors=df, raster=ds.values, affine=raster_transform, nodata=np.nan))["mean"]

    for thres in ds_thresh_list:
        # compute the percentage of the admin area that has cells below the threshold
        # set all values with below average rainfall to 1 and others to 0
        forecast_binary = np.where(ds.values <= thres, 1, 0)
        # compute number of cells in admin region (sum) and number of cells in admin region with below average rainfall (count)
        bin_zonal = pd.DataFrame(
            zonal_stats(vectors=df, raster=forecast_binary, affine=raster_transform, stats=['count', 'sum'],
                        nodata=np.nan))
        df[f'perc_se{thres}'] = bin_zonal['sum'] / bin_zonal['count'] * 100

    return df


def compute_stats_rainyseason(country_iso3, adm_level,days_ahead, output_dir, rainy_season_path,use_cache=True):
    # this takes some time to compute, couple of hours at max
    adm_col=parameters[f"shp_adm{adm_level}c"]
    adm_bound_path=os.path.join(country_data_raw_dir,config.SHAPEFILE_DIR,parameters[f"path_admin{adm_level}_shp"])
    rainy_dates = get_rainy_season_dates(rainy_season_path)
    gdf_adm = gpd.read_file(adm_bound_path)

    # load the tif file for each date and compute the statistics
    for d in rainy_dates:
        date_str=pd.to_datetime(d).strftime("%Y%m%d")
        output_path = output_dir / f"{country_iso3}_chirpsgefs_stats_{date_str}.csv"

        if use_cache and output_path.exists():
            logger.debug(
                f"{output_path} already exists and cache is set to True, skipping"
            )
        else:
            Path(output_path.parent).mkdir(parents=True, exist_ok=True)
            chirpsgefs_raster_filepath = config.CHIRPSGEFS_RAW_DIR / config.CHIRPSGEFS_RAW_FILENAME.format(days_ahead=days_ahead, date=date_str)
            if not chirpsgefs_raster_filepath.exists():
                logger.info(f'CHIRPS-GEFS data has not been downloaded for {date_str}')
            else:
                logger.debug(f"Retrieving stats for {date_str}")
                rds=rioxarray.open_rasterio(chirpsgefs_raster_filepath)
                df = compute_raster_stats(rds.sel(band=1), rds.rio.transform(), gdf_adm, adm_col)
                df.loc[:,"date"] = d
                df.loc[:,"date_forec_end"] = df.loc[:,"date"] + timedelta(days=days_ahead - 1)
                df.drop("geometry", axis=1).to_csv(output_path, index=False)



# def compute_stats_rainyseason(adm_level, days_ahead, rainy_season_path, rainy_adm_col, onset_col="onset_date",
#                               cessation_col="cessation_date"):
#     # this takes some time to compute, couple of hours at max
#     adm_col = parameters[f"shp_adm{adm_level}c"]
#     adm_bound_path = os.path.join(country_data_raw_dir, config.SHAPEFILE_DIR, parameters[f"path_admin{adm_level}_shp"])
#     rainy_dates = get_rainy_season_dates(rainy_season_path)
#     gdf_adm = gpd.read_file(adm_bound_path)
#
#     df_list = []
#     # load the tif file for each date and compute the statistics
#     for d in rainy_dates:
#         date_str = pd.to_datetime(d).strftime("%Y%m%d")
#         chirpsgefs_filepath = config.CHIRPSGEFS_RAW_DIR / config.CHIRPSGEFS_RAW_FILENAME.format(days_ahead=days_ahead,
#                                                                                                 date=date_str)
#         try:
#             logger.debug(f"Retrieving stats for {date_str}")
#             rds = rioxarray.open_rasterio(chirpsgefs_filepath)
#             df_date = compute_raster_stats(rds.sel(band=1), rds.rio.transform(), gdf_adm, adm_col)
#             df_date["date"] = d
#             df_date["date_forec_end"] = df_date["date"] + timedelta(days=days_ahead - 1)
#             df_list.append(df_date)
#         except Exception as e:
#             logger.info(f'CHIRPS-GEFS data not available for {date_str}')
#     df_hist_all = pd.concat(df_list)

#TODO: add this to processing notebook
#     df_rain = pd.read_csv(rainy_season_path, parse_dates=[onset_col, cessation_col])
#     df_rain_adm = df_rain.groupby([rainy_adm_col, "season_approx"], as_index=False).agg(
#         {onset_col: np.min, cessation_col: np.max})
#     list_hist_rain_adm = []
#     for a in df_hist_all[adm_col].unique():
#         dates_adm = pd.Index([])
#         df_rain_seladm = df_rain_adm[df_rain_adm[adm_col] == a]
#         for i in df_rain_seladm.season_approx.unique():
#             df_rain_seladm_seas = df_rain_seladm[df_rain_adm.season_approx == i]
#             seas_range = pd.date_range(df_rain_seladm_seas.onset_date.values[0],
#                                        df_rain_seladm_seas.cessation_date.values[0])
#             dates_adm = dates_adm.union(seas_range)
#         list_hist_rain_adm.append(df_hist_all[(df_hist_all[adm_col] == a) & (df_hist_all.date.isin(dates_adm))])
#     df_hist_rain = pd.concat(list_hist_rain_adm)
#
#     output_path = country_data_processed_dir / "chirpsgefs" / f"{country_iso3}_chirpsgefs_stats_rainyseas.csv"
#     # # #save file
#     hist_path = os.path.join(country_data_processed_dir, "dry_spells", "chirpsgefs",
#                              f"mwi_chirpsgefs_rainyseas_stats_mean_back_{adm_level}{days_string}.csv")
#     df_hist_rain.drop("geometry", axis=1).to_csv(output_path, index=False)


def get_rainy_season_dates(rainy_season_path,onset_col="onset_date",cessation_col="cessation_date"):
    """
    Compute a datetimeindex with all dates across all rainy seasons.
    This is used to only download data for these dates, since the files are large
    Args:
        rainy_season_path: filepath to csv with end and start dates of the rainy season, possibly per admin
    """
    df_rain=pd.read_csv(rainy_season_path,parse_dates=[onset_col,cessation_col])

    #remove entries where there is no onset and no cessation date, i.e. no rainy season
    df_rain=df_rain[(df_rain.onset_date.notnull())|(df_rain.cessation_date.notnull())]
    #if onset date or cessation date is missing, set it to Nov 1/Jul1 to make sure all data of that year is downloaded. This happens if e.g. the rainy season hasn't ended yet
    df_rain[df_rain.onset_date.isnull()]=df_rain[df_rain.onset_date.isnull()].assign(onset_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]}-11-01"))
    df_rain[df_rain.cessation_date.isnull()]=df_rain[df_rain.cessation_date.isnull()].assign(cessation_date=lambda df: pd.to_datetime(f"{df.season_approx.values[0]+1}-07-01"))

    #get min onset and max cessation for each season across all admins
    df_rain_seas=df_rain.groupby("season_approx",as_index=False).agg({onset_col: np.min,cessation_col:np.max})

    #create a daterange index including all dates within all rainy seasons
    all_dates=pd.Index([])
    for i in df_rain_seas.season_approx.unique():
        seas_range=pd.date_range(df_rain_seas[df_rain_seas.season_approx==i].onset_date.values[0],df_rain_seas[df_rain_seas.season_approx==i].cessation_date.values[0])
        all_dates=all_dates.union(seas_range)
    return all_dates

def download_chirpsgefs(date,config,days_ahead,use_cache=True):
    """
    Download the chirps-gefs africa forecast for the given year and day of the year
    We are focussing on the Africa data, since global data gets massive. Nevertheless, even for Africa it gets massive.
    Note that a part of the data for 2020 is missing due to a change in model
    days: number of days forecast predicts ahead. When using 15 day, set to empty string
    """

    date_str=date.strftime("%Y%m%d")
    chirpsgefs_filepath = config.CHIRPSGEFS_RAW_DIR / config.CHIRPSGEFS_RAW_FILENAME.format(days_ahead=days_ahead,date=date_str)
    if use_cache and chirpsgefs_filepath.exists():
        logger.debug(
            f"{chirpsgefs_filepath} already exists and cache is set to True, skipping"
        )
        return chirpsgefs_filepath
    Path(config.CHIRPSGEFS_RAW_DIR).mkdir(parents=True, exist_ok=True)
    try:
        year = date.year
        day_year = str(date.timetuple().tm_yday).zfill(3)
        # start_day is the day of the year for which the forecast starts
        download_ftp(config.CHIRPSGEFS_FTP_URL_AFRICA.format(days_ahead=str(days_ahead).zfill(2),year=year,start_day=day_year), chirpsgefs_filepath)
        logger.debug(f"...successfully downloaded {chirpsgefs_filepath}")
    except Exception as e:
        logger.info(f'CHIRPS-GEFS data not available for {date_str}')
    return chirpsgefs_filepath


