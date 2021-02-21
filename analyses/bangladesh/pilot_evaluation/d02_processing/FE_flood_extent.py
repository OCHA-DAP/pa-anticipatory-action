import geopandas as gpd
import pandas as pd
import datetime
import os


def get_gee_files(shp_dir):
    """
    Get the file names from the GEE outputs.
    """
    dates = []
    for d in os.listdir(shp_dir):
        sp = d.split('-')
        if d.startswith('BGD') and d.endswith('shp'):
            dates.append((sp[1] + '-' + sp[2] + '-' + sp[3] + '-' + sp[4] + '-' + sp[5]).split('.')[0])
    return dates


def clean_df(df, adm):
    """
    Gets rid of redundant columns, converts to date time format
    df: dataframe output from FE_flood_extent.py
    group: ADM4, ADM3, ADM2, ADM1
    """
    name = adm + '_EN_y'
    pcode = adm + '_PCODE'
    copy = df
    copy['date'] = pd.to_datetime(copy['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    output = copy[[name, pcode, 'flood_fraction', 'date']]
    output.columns = [adm+'_EN', pcode, 'flood_fraction', 'date']
    return output


def get_adm_shp(adm, shp_dir):
    """
    Loads and processes the admin region shapefile.
    :param adm: admin level of aggregation
    :param shp_dir: shapefile directory, specified in config file
    :return: geopandas df with admin regions
    """
    adm_grp = adm + '_PCODE'  # Need to do by pcode because admin names are not unique
    adm_shp = gpd.read_file(os.path.join(shp_dir, 'bdg_shp/bgd_admbnda_adm4_bbs_20180410.shp'))
    adm_shp = adm_shp[adm_shp['ADM2_EN'].isin(['Bogra', 'Gaibandha', 'Jamalpur', 'Kurigram', 'Sirajganj'])]
    adm_shp = adm_shp.to_crs('ESRI:54009')
    if adm_grp != 'ADM4_EN':  # Dissolve the shp if necessary
        adm_shp = adm_shp.dissolve(by=adm_grp).reset_index()
    adm_shp.loc[:, 'adm_area'] = adm_shp['geometry'].area
    return adm_shp


def get_river_area(adm, shp_dir):
    """
    Calculates the river area by admin unit. River area shp comes from JRC Global Surface Water
    :param adm: admin level of aggregation
    :param shp_dir: shapefile directory, specified in config file
    :return: geopandas df with the river area by adm unit
    """
    adm_grp = adm + '_PCODE'
    river_shp = gpd.read_file(os.path.join(shp_dir, 'river_extent.shp'))
    river_shp = river_shp.to_crs('ESRI:54009')
    adm_shp = get_adm_shp(adm, shp_dir)
    intersection = gpd.overlay(adm_shp, river_shp, how='difference')
    intersection = intersection.dissolve(by=adm_grp)
    river_extent = intersection['geometry'].area
    river_extent = river_extent.rename('not_river_area')
    output_df = pd.merge(adm_shp, river_extent.to_frame(), left_on=adm_grp, right_index=True)
    output_df.loc[:, 'river_area'] = output_df['adm_area'] - output_df['not_river_area']
    output_df = output_df[[adm_grp, adm+'_EN', 'adm_area', 'river_area']]
    return output_df


def get_flood_area(adm_grp, adm_shp, date, shp_dir):
    """
    Calculate the flooded area for each admin region for a given point in time
    :param adm_grp: Unique column to identify admin levels (eg. ADM4_PCODE)
    :param adm_shp: Shapefile with admin boundaries
    :param date: Date of flooding - used to read in the flooding shapefiles which are named by date
    :param shp_dir: Shapefile directory
    :return: dataframe with the total flooded area by admin region
    """
    fname = os.path.join(shp_dir + f'/BGD_Floods-{date}.shp')
    flood_shp = gpd.read_file(fname)
    flood_shp = flood_shp.to_crs('ESRI:54009')

    # We need to calculate the flood area with a method that is robust...
    # ...to admin areas that have zero intersection (ie. zero flooding)
    intersection = gpd.overlay(adm_shp, flood_shp, how='difference')
    intersection = intersection.dissolve(by=adm_grp)
    not_flooded = intersection['geometry'].area
    not_flooded = not_flooded.rename('not_flooded_area')
    output_df_part = pd.merge(adm_shp, not_flooded.to_frame(), left_on=adm_grp, right_index=True)
    output_df_part.loc[:, 'flooded_area'] = output_df_part['adm_area'] - output_df_part['not_flooded_area']
    output_df_part.loc[:, 'date'] = datetime.datetime.strptime(date[:10], '%Y-%m-%d')

    # --
    #flood_extent = intersection['geometry'].area
    #flood_extent = flood_extent.rename('flooded_area')
    #output_df_part = pd.merge(adm_shp, flood_extent.to_frame(), left_on=adm_grp, right_index=True)
    #output_df_part.loc[:, 'date'] = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    return output_df_part


def main(adm, shp_dir, data_dir):
    """
    Calculate the extent of flooding for ADM4 regions using the shapefiles output from GEE script.
    dirname = name of folder with the shapefiles output from
    """
    adm_grp = adm + '_PCODE'  # Need to do by pcode because admin names are not unique
    adm_shp = get_adm_shp(adm, shp_dir)
    dates = get_gee_files(shp_dir)
    river_area = get_river_area(adm, shp_dir)
    output_df = pd.DataFrame()
    # Loop through all shapefiles and calculate the flood extent
    for date in dates:
        print(date)
        df_flood_frac = get_flood_area(adm_grp, adm_shp, date, shp_dir)
        output_df = output_df.append(df_flood_frac)
    # Merge with the river area measurements
    output_df = pd.merge(output_df, river_area, on=adm_grp)
    # Subtract river area from flooded area
    output_df.loc[:, 'non_river_area'] = output_df['adm_area_y'] - output_df['river_area']
    # Calculate the flooded fraction
    output_df.loc[:, 'flood_fraction'] = output_df['flooded_area'] / output_df['non_river_area']
    # Clean the dataframe
    output_df = clean_df(output_df, adm)
    output_df.to_csv(os.path.join(data_dir, f'{adm}_flood_extent_sentinel.csv'), index=False)
    return output_df
