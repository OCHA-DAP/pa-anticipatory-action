import numpy as np
import datetime
from scipy.optimize import curve_fit
import time
import operator


def get_xy(df):
    """
    Extract x and y arrays from a grouped, selected dataframe. Converts time to integers
    df: output of FE_clean_data.select_df()
    """

    ts_data = df
    # Create a column with time as numeric
    ts_data['time_int'] = [time.mktime(time.strptime(date, '%Y-%m-%d')) for date in ts_data['date']]
    # Define the x and y
    y = np.array(ts_data['flood_fraction'])
    x = np.array(ts_data['time_int'])
    return x, y


def gauss(x, H, A, x0, sigma):
    """
    Defines a Gaussian function.
    Adapted from https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466
    """

    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_fit(x, y):
    """
    Fits a Gaussian function to input x and y data.
    Adapted from https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466
    """

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], maxfev=5000)
    return popt


def get_fwhm(sigma):
    """
    Computes the Full Width at Half Max (FWHM) for the Gaussian curve.
    This is used as an indication of the length of time that flooding
    in a given area is at or above 50% of its total extent.
    :param sigma: standard deviation of the Gaussian curve
    :return: fwhm, rounded to the nearest day
    """
    return round(sigma * 2.355 / 86400)


def poly_fit(x_pred, x_actual, y_actual, degree=3):
    """
    Fits a polynomial function to input x and y data.
    x_pred: x used to generate fitted y values
    x_actual: actual x values from the data
    y_actual: actual y values from the data
    degree: degree of polynomial function. 3 is default.
    """

    z = np.polyfit(x_actual, y_actual, degree)
    func = np.poly1d(z)
    return func(x_pred)


def rmse(y_pred, y_true):
    """
    Returns RMSE value from predicted and actual y values.
    """

    return np.sqrt(((y_pred - y_true) ** 2).mean())


def get_peak(x, y):
    """
    Get the peak flooding date from input x (date) and y (flood extent) arrays
    """

    index, value = max(enumerate(y), key=operator.itemgetter(1))
    return (datetime
            .datetime
            .fromtimestamp(x[index])
            .strftime('%Y-%m-%d'))
