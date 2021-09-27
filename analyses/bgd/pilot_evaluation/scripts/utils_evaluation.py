import argparse
import datetime
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy.optimize import curve_fit


def get_xy(df):
    """Extract x and y arrays from a grouped, selected dataframe.

    Converts time to integers
    df: output of FE_clean_data.select_df()
    """

    ts_data = df
    # Create a column with time as numeric
    ts_data["time_int"] = [
        time.mktime(time.strptime(date, "%Y-%m-%d"))
        for date in ts_data["date"]
    ]
    # Define the x and y
    y = np.array(ts_data["flooded_fraction"])
    x = np.array(ts_data["time_int"])
    return x, y


def gauss(x, A, x0, sigma):
    """Defines a Gaussian function.

    Adapted from
    https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466 x:
    x values A: amplitude x0: mean sigma: sigma
    """

    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def gauss_fit(x, y):
    """Fits a Gaussian function to input x and y data.

    Adapted from
    https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466
    """

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(
        gauss,
        x,
        y,
        p0=[max(y), mean, sigma],
        bounds=((0, 1591401600, -np.inf), (1, 1598745600, np.inf)),
        maxfev=5000,
    )
    return popt, pcov


def get_fwhm(sigma):
    """Computes the Full Width at Half Max (FWHM) for the Gaussian
    curve. This is used as an indication of the length of time that
    flooding in a given area is at or above 50% of its total extent.

    :param sigma: standard deviation of the Gaussian curve :return:
    fwhm, rounded to the nearest day
    """
    return round(sigma * 2.355 / 86400)


def poly_fit(x_pred, x_actual, y_actual, degree=3):
    """Fits a polynomial function to input x and y data.

    x_pred: x used to generate fitted y values
    x_actual: actual x values from the data
    y_actual: actual y values from the data
    degree: degree of polynomial function. 3 is default.
    """

    z = np.polyfit(x_actual, y_actual, degree)
    func = np.poly1d(z)
    return func(x_pred)


def rmse(y_pred, y_true):
    """Returns RMSE value from predicted and actual y values."""

    return np.sqrt(((y_pred - y_true) ** 2).mean())


def get_peak(x, y):
    """Get the peak flooding date from input x (date) and y (flood
    extent) arrays."""

    index, value = max(enumerate(y), key=operator.itemgetter(1))
    return datetime.datetime.fromtimestamp(x[index]).strftime("%Y-%m-%d")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "adm_level", help="Admin level to calculate flood fraction"
    )
    args = parser.parse_args()
    return args


def parse_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def compare_estimates(data, sel_union):
    """Return a graph for the selected union, comparing Sentinel-1
    estimates with interview estimates.

    data = dataframe with the flooding estimates over time for BGD
    admin-4 regions sel_union = name of ADM4 region
    """

    # Get only the data for the selected union
    df_selected = data.loc[data["ADM4_PCODE"] == sel_union]
    # Select the relevant flooding estimate data
    # Get the various estimates of flood extent by date (in %)
    flood_extent = df_selected[
        ["flood_fraction", "Interview_1", "Interview_2", "Interview_3", "date"]
    ]
    # Melt data to long format for visualization
    flood_extent_long = flood_extent.melt(id_vars=["date"])
    # Colours for the line graph
    col_mapping = {
        "Interview_1": "#520057",
        "Interview_2": "#db00e8",
        "Interview_3": "#d096d4",
        "flood_fraction": "#ff9626",
    }
    # Create simple line plot to compare the satellite estimates and
    # interviewed estimates
    sns.lineplot(
        x="date",
        y="value",
        hue="variable",
        data=flood_extent_long,
        palette=col_mapping,
    )
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Percent flooding estimate")
    plt.ylim([0, 110])
    plt.title("Estimates of flooding in {}, Bangladesh".format(sel_union))
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, 1))
    plt.legend()
    plt.tight_layout()
