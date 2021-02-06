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
# 'array must not contain infs or NaNs' - this is from the na values introduced when admin
# area is fully covered by the river
#np.seterr(all='raise')

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

        # Catch the edge cases where there is no flooding
        # And where the admin area is covered by water
        # TODO: Control flow is a bit convoluted here
        if y.mean() == 0:
            y_new = np.ones(85)*0
            cov = None
            rmse_g = None
            date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
            date_g = None
            act_g = None
            fwhm = None
            max_actual = y.max()

        else:
            try:
                # New y values using same x data to calc the error
                y_old = ff.gauss(x, *ff.gauss_fit(x, y)[0])  # Generate Gaussian fitted y data
                # New y values using daily x data to get better peak estimate
                y_new = ff.gauss(x_new, *ff.gauss_fit(x, y)[0])  # Generate Gaussian fitted y data

            except Exception as e:
                logger.warning(e)
                no_fit.append(adm)

                y_new = np.empty(85) * np.nan
                cov = None
                rmse_g = None
                date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
                date_g = None
                act_g = None
                fwhm = None
                max_actual = y.max()

            else:
                # Get one standard deviation errors on the parameters
                # We will focus just on the mean of the Gaussian (ie. date)
                # In some cases this returns a negative value
                # https://stackoverflow.com/questions/28702631/scipy-curve-fit-returns-negative-variance
                # ^ Seems to indicate that it's just done a bad job of fitting
                pcov = ff.gauss_fit(x, y)[1]
                cov = np.sqrt(np.diag(pcov)[2])
                cov = cov / 86400  # Convert from seconds to days
                # Calc the rmse to compare poly vs gauss
                rmse_g = ff.rmse(y_old, y)
                # Get the peak dates
                date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
                date_g = datetime.strptime(ff.get_peak(x_new, y_new), "%Y-%m-%d")
                # Calculate the difference between dates
                act_g = (date_actual - date_g).days
                # Get the FWHM from the Gaussian fitting
                # Tells us the length of time that flooding was at at least 50% of peak
                sigma = ff.gauss_fit(x, y)[0][3]
                fwhm = ff.get_fwhm(sigma)
                max_actual = y.max()

        # Create dict with the results - flood extent
        flood_extent = pd.DataFrame(
            {'PCODE': adm,
             'DATE': x_new,
             'FLOOD_EXTENT': y_new})
        flood_extents = flood_extents.append(flood_extent, ignore_index=True)

        # Create dict with the results - peak dates
        result = {'PCODE': adm,
                  'RMSE': rmse_g,
                  'PEAK_SAT': date_actual,
                  'PEAK_G': date_g,
                  'DIFF_SAT': act_g,
                  'FWHM': fwhm,
                  'COV': cov,
                  'MAX_SAT': max_actual}
        dates = dates.append(result, ignore_index=True)

    # Get the maximum flooding extent and add it to the results dataframe
    max_flood_G = flood_extents.groupby('PCODE')['FLOOD_EXTENT'].max().reset_index()
    dates = dates.merge(max_flood_G, on='PCODE')
    dates.rename(columns={"FLOOD_EXTENT": "MAX_G"}, inplace=True)

    # Save the files to output directory
    flood_extents['DATE'] = flood_extents['DATE'].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d"))
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
