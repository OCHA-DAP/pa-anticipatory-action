from scripts.d03_analysis import FE_fit_function as ff
from scripts import utils
import pandas as pd
from datetime import datetime
import numpy as np
import os
import logging

# This script takes the output file from Generate_flood_frac.py and fits Gaussian
# and polynomial functions to the data, to interpolate between dates without Sentinel-1
# coverage.

# Required inputs are:
# 1) The .csv file output from the Generate_flood_frac.py script, located in 'data_dir'
# 2) The admin level used to calculate the flood fraction (Eg. ADM2, ADM3, ADM4), specified as a command-line argument

# Directory locations for the input and output files should be specified in the 'config.yml' file.

# TODO: handle the potential errors and warnings more specifically
# TODO: fill in zeros across y where the fit doesn't work?

# To raise all warnings so that they are caught in the exception
# Warnings encountered are mathematical:
# 'invalid value encountered in double_scalars'
# 'underflow encountered in exp'
# 'divide by zero encountered in true_divide'
# 'Optimal parameters not found: Number of calls to function has reached maxfev = 5000'
np.seterr(all='raise')

parameters = utils.parse_yaml('config.yml')['DIRS']
output_dir = parameters['data_dir']
logger = logging.getLogger()


def make_data(df, adm_grp):

    # Output dataframes
    dates = pd.DataFrame([])
    flood_extents = pd.DataFrame([])
    no_fit = []  # To output the list of admin units with no interpolated data

    sel_col = adm_grp + '_PCODE'
    for adm in df[sel_col].unique():

        # Fit the data
        df2 = df.loc[df[sel_col] == adm].reset_index()
        x, y = ff.get_xy(df2)[0], ff.get_xy(df2)[1]  # Get the x and y
        x_new = np.linspace(x[0], x[-1], 85)  # Generate new x data (at daily intervals)

        try:
            # New y values using same x data to calc the error
            y_g_old = ff.gauss(x, *ff.gauss_fit(x, y))  # Generate Gaussian fitted y data
            y_p_old = ff.poly_fit(x, x, y, 3)  # Generate polynomial fitted y data - degree 3
        except Exception as e:
            logger.warning(e)
            no_fit.append(adm)
            continue
        # New y values using daily x data to get better peak estimate
        y_g_new = ff.gauss(x_new, *ff.gauss_fit(x, y))  # Generate Gaussian fitted y data
        y_p_new = ff.poly_fit(x_new, x, y, 3)  # Generate polynomial fitted y data - degree 3

        # Calc the rmse to compare poly vs gauss
        rmse_g = ff.rmse(y_g_old, y)
        rmse_p = ff.rmse(y_p_old, y)

        # Get the peak dates
        date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
        date_g = datetime.strptime(ff.get_peak(x_new, y_g_new), "%Y-%m-%d")
        date_p = datetime.strptime(ff.get_peak(x_new, y_p_new), "%Y-%m-%d")

        # Calculate the difference between dates
        act_g = (date_actual - date_g).days
        act_p = (date_actual - date_p).days

        # Get the FWHM from the Gaussian fitting
        # Tells us the length of time that flooding was at at least 50% of peak
        sigma = ff.gauss_fit(x, y)[3]
        fwhm = ff.get_fwhm(sigma)

        # Create dict with the results - flood extent
        flood_extent = pd.DataFrame(
            {'PCODE': adm,
             'DATE': x_new,
             'FLOOD_EXTENT_G': y_g_new,
             'FLOOD_EXTENT_P': y_p_new})
        flood_extents = flood_extents.append(flood_extent, ignore_index=True)

        # Create dict with the results - peak dates
        result = {'PCODE': adm,
                  'RMSE_G': rmse_g,
                  'RMSE_P': rmse_p,
                  'PEAK_ACT': date_actual,
                  'PEAK_G': date_g,
                  'PEAK_P': date_p,
                  'DIFF_ACT_G': act_g,
                  'DIFF_ACT_P': act_p,
                  'FWHM': fwhm}
        dates = dates.append(result, ignore_index=True)

    # Get the maximum flooding extent and add it to the results dataframe
    max_flood_G = flood_extents.groupby('PCODE')['FLOOD_EXTENT_G'].max().reset_index()
    max_flood_P = flood_extents.groupby('PCODE')['FLOOD_EXTENT_P'].max().reset_index()
    dates = dates.merge(max_flood_G, on='PCODE')
    dates = dates.merge(max_flood_P, on='PCODE')
    dates.rename(columns={"FLOOD_EXTENT_G": "MAX_G", "FLOOD_EXTENT_P": "MAX_P"}, inplace=True)

    # Save the files to output directory
    flood_extents['DATE'] = flood_extents['DATE'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%d/%m/%Y'))
    dates.to_csv(os.path.join(output_dir, f'{adm_grp}_flood_summary.csv'), index=False)
    flood_extents.to_csv(os.path.join(output_dir, f'{adm_grp}_flood_extent_interpolated.csv'), index=False)
    pd.DataFrame(no_fit, columns=['No_Fit']).to_csv(os.path.join(output_dir, f'{adm_grp}_no_fit.csv'), index=False)
    logger.info(f'Output files saved to {output_dir}')


if __name__ == "__main__":
    arg = utils.parse_args()
    utils.config_logger()
    # Read in the dataframe output from Generate_flood_frac.py
    try:
        sentinel = pd.read_csv(os.path.join(output_dir, f'{arg.adm_level}_flood_extent_sentinel.csv'))
        # Calculate the fitted data
        make_data(sentinel, arg.adm_level)
    except FileNotFoundError:
        logger.error('Input CSV file not found. Run Generate_flood_frac.py to generate the required file.')
