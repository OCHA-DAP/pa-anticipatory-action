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
# np.seterr(all='raise')

dirs = utils.parse_yaml('config.yml')['DIRS']
output_dir = dirs['data_dir']
params = utils.parse_yaml('config.yml')['PARAMS']
ADM = params['adm']

logger = logging.getLogger()


def make_data(df, adm_grp):

    # Initialize the dataframe that will hold the output data
    dates = pd.DataFrame([])
    flood_extents = pd.DataFrame([])
    no_fit = []

    sel_col = adm_grp + '_PCODE'
    for adm in df[sel_col].unique():

        # Get the x (time) and y (flooding fraction) data
        # from the sentinel estimates which we will use to
        # estimate the Gaussian parameters. We'll generate new
        # x values at daily intervals to interpolate the corresponding
        # flooding fraction on a daily basis.
        df2 = df.loc[df[sel_col] == adm].reset_index()
        x, y = ff.get_xy(df2)[0], ff.get_xy(df2)[1]
        x_new = np.linspace(x[0], x[-1], 85)

        # Before attempting to fit to a Gaussian, we need to
        # catch the edge cases where there is no flooding
        # or where the admin area is covered by water.
        if y.mean() == 0:
            y_new = np.ones(85)*0
            cov = None
            rmse = None
            date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
            date_g = None
            act_g = None
            fwhm = None
            max_actual = y.max()

        # But if some flooding has occurred, we will try to
        # fit the time series to a Gaussian distribution.
        else:
            try:
                popt, pcov = ff.gauss_fit(x, y)

            # In case the function doesn't converge to estimate the Gaussian parameters,
            # we will add in empty data to the summary output.
            except Exception as e:
                logger.warning(e)
                no_fit.append(adm)

                y_new = np.empty(85) * np.nan
                cov = None
                rmse = None
                date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
                date_g = None
                act_g = None
                fwhm = None
                max_actual = y.max()

            else:
                y_fit = ff.gauss(x, *popt)
                y_new = ff.gauss(x_new, *popt)
                # Standard deviation of the errors for the mean (x0) parameter
                # Convert from seconds (from unix time) to hours.
                cov = np.sqrt(np.diag(pcov)[1]) / 86400
                rmse = ff.rmse(y_fit, y)
                date_actual = datetime.strptime(ff.get_peak(x, y), "%Y-%m-%d")
                date_g = datetime.strptime(ff.get_peak(x_new, y_new), "%Y-%m-%d")
                act_g = (date_actual - date_g).days
                fwhm = ff.get_fwhm(popt[2])
                max_actual = y.max()

        # Create dictionaries to append to the output dataframes
        flood_extent = pd.DataFrame(
            {'PCODE': adm,
             'DATE': x_new,
             'FLOOD_EXTENT': y_new})
        flood_extents = flood_extents.append(flood_extent, ignore_index=True)
        result = {'PCODE': adm,
                  'RMSE': rmse,
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
    utils.config_logger()
    # Read in the dataframe output from Generate_flood_frac.py
    try:
        sentinel = pd.read_csv(os.path.join(output_dir, f'{ADM}_flood_extent_sentinel.csv'))
        # Calculate the fitted data
        make_data(sentinel, ADM)
    except FileNotFoundError:
        logger.error('Input CSV file not found. Run Generate_flood_frac.py to generate the required file.')
