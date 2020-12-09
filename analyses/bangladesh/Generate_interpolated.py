import pandas as pd
from scripts.d03_analysis import FE_fit_function as ff
from datetime import datetime
from scripts import utils
import numpy as np
import os
import logging

parameters = utils.parse_yaml('config.yml')['DIRS']
shp_dir = parameters['shp_dir']
output_dir = parameters['data_dir']

logger = logging.getLogger()


def make_data(df, adm_grp):

    # Output dataframes
    dates = pd.DataFrame([])
    flood_extents = pd.DataFrame([])
    intensities = pd.DataFrame([])

    sel_col = adm_grp + '_PCODE'

    for adm in df[sel_col].unique():

        # Fit the data
        df2 = df.loc[df[sel_col] == adm].reset_index()
        x, y = ff.get_xy(df2)[0], ff.get_xy(df2)[1]  # Get the x and y
        x_new = np.linspace(x[0], x[-1], 85)  # Generate new x data (at daily intervals)

        # TODO: Some regions don't have observations for every date, not incl when there is 0 flooding
        # Need to break out when there aren't enough original points to fit function
        if len(x) < 10:
            continue

        # New y values using same x data to calc the error
        y_g_old = ff.gauss(x, *ff.gauss_fit(x, y))  # Generate Gaussian fitted y data
        y_p_old = ff.poly_fit(x, x, y, 3)  # Generate polynomial fitted y data - degree 3

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

        # Create dict with the results - fwhm to indicate intensity of the flooding
        intensity = pd.DataFrame({
            'PCODE': adm,
            'FWHM': fwhm
        }, index=[0])

        intensities = intensities.append(intensity, ignore_index=True)

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
                  'DIFF_ACT_P': act_p}
        dates = dates.append(result, ignore_index=True)

    # Save the files to output directory
    flood_extents['DATE'] = flood_extents['DATE'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%d/%m/%Y'))
    intensities.to_csv(os.path.join(output_dir, f'{adm_grp}_flood_intensity.csv'), index=False)
    dates.to_csv(os.path.join(output_dir, f'{adm_grp}_flood_peak.csv'), index=False)
    flood_extents.to_csv(os.path.join(output_dir, f'{adm_grp}_flood_extent_interpolated.csv'), index=False)
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
