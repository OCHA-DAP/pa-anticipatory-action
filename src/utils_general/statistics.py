import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xskillscore as xs
import logging
import xarray as xr

from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_return_periods_dataframe(
    df: pd.DataFrame,
    rp_var: str,
    years: list = None,
    method: str = "analytical",
    show_plots: bool = False,
    extend_factor: int = 1,
) -> pd.DataFrame:
    """
    Function to get the return periods, either empirically or
    analytically See the `glofas/utils.py` to do this with a xarray
    dataset instead of a dataframe
    :param df: Dataframe with data to compute rp on
    :param rp_var: column name to compute return period on
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :param show_plots: If method is analytical, can show the histogram and GEV
    distribution overlaid
    :param extend_factor: If method is analytical, can extend the interpolation
    range to reach higher return periods
    :return: Dataframe with return period years as index and stations as
    columns
    """
    if years is None:
        years = [1.5, 2, 3, 5]
    df_rps = pd.DataFrame(columns=["rp"], index=years)
    if method == "analytical":
        f_rp = get_return_period_function_analytical(
            df_rp=df,
            rp_var=rp_var,
            show_plots=show_plots,
            extend_factor=extend_factor,
        )
    elif method == "empirical":
        f_rp = get_return_period_function_empirical(
            df_rp=df,
            rp_var=rp_var,
        )
    else:
        logger.error(f"{method} is not a valid keyword for method")
        return None
    df_rps["rp"] = np.round(f_rp(years))
    return df_rps


def get_return_period_function_analytical(
    df_rp: pd.DataFrame,
    rp_var: str,
    show_plots: bool = False,
    plot_title: str = "",
    extend_factor: int = 1,
):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column with the quantity to be evaluated
    :param show_plots: Show the histogram with GEV distribution overlaid
    :param plot_title: The title of the plot
    :param extend_factor: Extend the interpolation range in case you want to
    calculate a relatively high return period
    :return: Interpolated function that gives the quantity for a
    given return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values,
        loc=rp_var_values.median(),
        scale=rp_var_values.median() / 2,
    )
    x = np.linspace(
        rp_var_values.min(),
        rp_var_values.max() * extend_factor,
        100 * extend_factor,
    )
    if show_plots:
        fig, ax = plt.subplots()
        ax.hist(rp_var_values, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(plot_title)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)


def get_return_period_function_empirical(df_rp: pd.DataFrame, rp_var: str):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column
    with the quantity to be evaluated
    :return: Interpolated function
    that gives the quantity for a give return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp[rp_var])


def calc_mpe(observations: np.array, forecast: np.array) -> float:
    """
    :param observations: array with the observed values
    :param forecast: array with the forecasted values.
    Should be the same shape as observations
    :return: the mpe across all entries in the arrays
    """
    return (
        ((forecast - observations) / observations).sum()
        / len(observations.time)
        * 100
    )


def calc_crps(
    observations: xr.DataArray,
    forecasts: xr.DataArray,
    normalization: [str, float] = None,
    member_dim: str = "number",
) -> float:
    """
    observations and forecasts must have the same shape in all
    dimensions except the member_dim dimension
    the member_dim should only be present in the forecasts not in the
    observation
    :param observations: datarray with observed values
    :param forecasts: data-array with forecasted values
    :param normalization: (optional) Can be None, a number, 'mean' or 'std',
    reanalysis metric to divide the CRPS
    :param member_dim: (optional) the dimension name which contains
    the ensemble members
    :return: the CRPS
    """
    if isinstance(normalization, (int, float)):
        norm = normalization
    elif normalization == "mean":
        norm = observations.mean().values
    elif normalization == "std":
        norm = observations.std().values
    elif normalization is None:
        norm = 1
    else:
        logger.error(
            f"Normalization method {normalization} has not been implemented"
        )
    crps = (
        xs.crps_ensemble(observations, forecasts, member_dim=member_dim).values
        / norm
    )
    return crps
