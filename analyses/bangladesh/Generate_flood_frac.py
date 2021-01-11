from scripts import utils
import geopandas as gpd
import pandas as pd
import datetime
import os

# This script takes the flood extent shapefiles output from the GEE Sentinel-1 data processing script
# and outputs a .csv file that provides a time series of the flooding fraction by admin units in Bangladesh.

# Required inputs are:
# 1) The shapefiles of flood extent output from the GEE script, located within the same 'gee_dir'
# 2) Shapefile of admin regions in Bangladesh, located in the 'adm_dir'
# 3) Shapefile with permanent water bodies in Bangladesh, located in the 'adm_dir' folder
# 4) The admin level used to calculate the flood fraction (Eg. ADM2, ADM3, ADM4), specified as a command-line argument

# Directory locations for the input and output files should be specified in the 'config.yml' file.

# TODO: Fix variable hard-coding. Currently hard-coded variables include:
# - Bangladesh shapefile name
# - River extent shapefile name
# - CRS for shapefiles
# - Bangladesh districts in the region of interest

parameters = utils.parse_yaml('config.yml')['DIRS']
adm_dir = parameters['adm_dir']
gee_dir = parameters['gee_dir']
output_dir = parameters['data_dir']


def get_gee_files(gee_dir):
    """
    Get the file names from the GEE outputs.
    """
    dates = []
    for d in os.listdir(gee_dir):
        sp = d.split('-')
        if d.startswith('BGD') and d.endswith('shp'):
            dates.append((sp[1] + '-' + sp[2] + '-' + sp[3] + '-' + sp[4] + '-' + sp[5]).split('.')[0])
    return dates


def get_adm_shp(adm, adm_dir):
    """
    Loads and processes the admin region shapefile.
    :param adm: admin level of aggregation
    :param adm_dir: shapefile directory, specified in config file
    :return: geopandas df with admin regions
    """
    if adm == 'MAUZ':
        adm_shp = gpd.read_file(os.path.join(adm_dir, 'selected_distict_mauza.shp'))
        adm_shp = adm_shp[adm_shp['DISTNAME'].isin(['Bogra', 'Gaibandha', 'Jamalpur', 'Kurigram', 'Sirajganj'])]
        adm_shp.rename(columns={'OBJECTID': 'MAUZ_PCODE', 'MAUZNAME': 'MAUZ_EN'}, inplace=True) # Treat OBJECTID as PCODE field
    else:
        #adm_grp = adm + '_PCODE'  # Need to do by pcode because admin names are not unique
        adm_shp = gpd.read_file(os.path.join(adm_dir, f'bgd_admbnda_{adm}_bbs_20201113.shp'))
        adm_shp = adm_shp[adm_shp['ADM2_EN'].isin(['Bogra', 'Gaibandha', 'Jamalpur', 'Kurigram', 'Sirajganj'])]
        #if adm_grp != 'ADM4_EN':  # Dissolve the shp if necessary
        #    adm_shp = adm_shp.dissolve(by=adm_grp).reset_index()
    adm_shp = adm_shp.to_crs('ESRI:54009')
    adm_shp.loc[:, 'adm_area'] = adm_shp['geometry'].area
    return adm_shp


def get_river_area(adm, adm_dir):
    """
    Calculates the river area by admin unit. River area shp comes from JRC Global Surface Water
    :param adm: admin level of aggregation
    :param adm_dir: shapefile directory, specified in config file
    :return: geopandas df with the river area by adm unit
    """
    adm_grp = adm + '_PCODE'
    river_shp = gpd.read_file(os.path.join(adm_dir, 'river_extent.shp'))
    river_shp = river_shp.to_crs('ESRI:54009')
    adm_shp = get_adm_shp(adm, adm_dir)
    intersection = gpd.overlay(adm_shp, river_shp, how='difference')
    intersection = intersection.dissolve(by=adm_grp)
    river_extent = intersection['geometry'].area
    river_extent = river_extent.rename('not_river_area')
    output_df = pd.merge(adm_shp, river_extent.to_frame(), left_on=adm_grp, right_index=True)
    output_df.loc[:, 'river_area'] = output_df['adm_area'] - output_df['not_river_area']
    output_df = output_df[[adm_grp, adm+'_EN', 'adm_area', 'river_area']]
    return output_df


def get_flood_area(adm_grp, adm_shp, date, gee_dir):
    """
    Calculate the flooded area for each admin region for a given point in time
    :param adm_grp: Unique column to identify admin levels (eg. ADM4_PCODE)
    :param adm_shp: Shapefile with admin boundaries
    :param date: Date of flooding - used to read in the flooding shapefiles which are named by date
    :param gee_dir: Shapefile directory
    :return: dataframe with the total flooded area by admin region
    """
    fname = os.path.join(gee_dir + f'/BGD_Floods-{date}.shp')
    flood_shp = gpd.read_file(fname)
    flood_shp = flood_shp.to_crs('ESRI:54009')

    # We need to calculate the flood area with a method that is robust
    # to admin areas that have zero intersection (ie. zero flooding).
    # So here we identify the area that ISN'T flooded and subtract from
    # the total area to get the area that IS flooded.
    intersection = gpd.overlay(adm_shp, flood_shp, how='difference')
    intersection = intersection.dissolve(by=adm_grp)
    not_flooded = intersection['geometry'].area
    not_flooded = not_flooded.rename('not_flooded_area')
    output_df_part = pd.merge(adm_shp, not_flooded.to_frame(), left_on=adm_grp, right_index=True)
    output_df_part.loc[:, 'flooded_area'] = round(output_df_part['adm_area'] - output_df_part['not_flooded_area'],5)
    output_df_part.loc[:, 'date'] = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    return output_df_part


def main(adm, adm_dir, gee_dir, data_dir):
    """
    Calculate the extent of flooding for ADM4 regions using the shapefiles output from GEE script.
    dirname = name of folder with the shapefiles output from
    """
    adm_grp = adm + '_PCODE'  # Need to do by pcode because admin names are not unique
    adm_shp = get_adm_shp(adm, adm_dir)
    dates = get_gee_files(gee_dir)
    river_area = get_river_area(adm, adm_dir)
    output_df = pd.DataFrame()
    # Loop through all shapefiles and calculate the flood extent
    for date in dates:
        print(date)
        df_flood_frac = get_flood_area(adm_grp, adm_shp, date, gee_dir)
        output_df = output_df.append(df_flood_frac)
    # Merge with the river area measurements
    output_df = pd.merge(output_df, river_area, on=adm_grp)
    # Subtract river area from flooded area
    output_df.loc[:, 'non_river_area'] = output_df['adm_area_y'] - output_df['river_area']
    # Calculate the flooded fraction
    output_df.loc[:, 'flood_fraction'] = output_df['flooded_area'] / output_df['non_river_area']
    # Clean the dataframe
    output_df['date'] = pd.to_datetime(output_df['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    output_df = output_df[[(adm + '_EN_y'), (adm + '_PCODE'), 'flood_fraction', 'date']]
    output_df.columns = [adm + '_EN', (adm + '_PCODE'), 'flood_fraction', 'date']
    output_df.to_csv(os.path.join(data_dir, f'{adm}_flood_extent_sentinel.csv'), index=False)
    return output_df


if __name__ == "__main__":
    arg = utils.parse_args()
    main(arg.adm_level, adm_dir, gee_dir, output_dir)
