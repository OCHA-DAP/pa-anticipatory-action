import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt

def get_return_period_function_analytical(
    df_rp: pd.DataFrame, rp_var: str, show_plots: bool, plot_title = "",
):
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values, loc=rp_var_values.median(), scale=rp_var_values.median() / 2
    )
    x = np.linspace(rp_var_values.min(), rp_var_values.max(), 100)
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
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp[rp_var])