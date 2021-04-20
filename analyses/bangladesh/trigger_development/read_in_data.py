import sys
import os
from pathlib import Path

from scipy.stats import norm
import numpy as np
import pandas as pd
import xarray as xr

path_mod = f"{Path(os.path.dirname(os.path.realpath(''))).parents[1]}/"
sys.path.append(path_mod)

from src.indicators.flooding.glofas import glofas
from src.bangladesh import get_glofas_data as ggd

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
GLOFAS_DIR = DATA_DIR / "processed/bangladesh/GLOFAS_Data"
STATION = "Bahadurabad_glofas"
ffwc_dir = DATA_DIR / 'exploration/bangladesh/FFWC_Data'


def get_glofas_reanalysis():
    glofas_reanalysis = glofas.GlofasReanalysis()
    da_glofas_reanalysis = glofas_reanalysis.read_processed_dataset(
        country_name=ggd.COUNTRY_NAME, country_iso3=ggd.COUNTRY_ISO3
    )[STATION]
    return da_glofas_reanalysis


def get_glofas_forecast():
    glofas_forecast = glofas.GlofasForecast()
    da_glofas_forecast_dict = {
        leadtime_hour: glofas_forecast.read_processed_dataset(
            country_name=ggd.COUNTRY_NAME,
            country_iso3=ggd.COUNTRY_ISO3,
            leadtime_hour=leadtime_hour,
        )[STATION]
        for leadtime_hour in ggd.LEADTIME_HOURS
    }
    da_glofas_forecast_dict = shift_dates(da_glofas_forecast_dict)
    return convert_dict_to_da(da_glofas_forecast_dict)


def get_glofas_reforecast():
    glofas_reforecast = glofas.GlofasReforecast()
    da_glofas_reforecast_dict = {
        leadtime_hour: glofas_reforecast.read_processed_dataset(
            country_name=ggd.COUNTRY_NAME,
            country_iso3=ggd.COUNTRY_ISO3,
            leadtime_hour=leadtime_hour,
        )[STATION]
        for leadtime_hour in ggd.LEADTIME_HOURS
    }
    da_glofas_reforecast_dict = interp_dates(shift_dates(da_glofas_reforecast_dict))
    return convert_dict_to_da(da_glofas_reforecast_dict)


def shift_dates(da_dict):
    return {
        leadtime_hour: da.assign_coords(
            time=da.time.values + np.timedelta64(int(leadtime_hour / 24), "D")
        )
        for leadtime_hour, da in da_dict.items()
    }


def interp_dates(da_dict):
    return {
        leadtime_hour: da.interp(
            time=pd.date_range(da.time.min().values, da.time.max().values),
            method="linear",
        )
        for leadtime_hour, da in da_dict.items()
    }


def convert_dict_to_da(da_glofas_dict):
    # Create time array that accounts for all the shifts
    time = np.arange(
        da_glofas_dict[min(da_glofas_dict.keys())].time.values[0],
        da_glofas_dict[max(da_glofas_dict.keys())].time.values[-1]
        + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    da_glofas_dict = {
        leadtime_hour: da_glofas.reindex({"time": time})
        for leadtime_hour, da_glofas in da_glofas_dict.items()
    }

    data = np.array([da_glofas.values for da_glofas in da_glofas_dict.values()])
    # Create data array with all lead times, as well as ensemble members (number)
    # and timestep
    return xr.DataArray(
        data=data,
        dims=["leadtime_hour", "number", "time"],
        coords=dict(
            number=list(da_glofas_dict.values())[0].number,  # ensemble member number
            time=time,
            leadtime_hour=list(da_glofas_dict.keys()),
        ),
    )


def get_da_glofas_summary(da_glofas):
    nsig_max = 3
    percentile_dict = {
        **{"median": 50.0},
        **{f"{n}sig+": norm.cdf(n) * 100 for n in range(1, nsig_max + 1)},
        **{f"{n}sig-": (1 - norm.cdf(n)) * 100 for n in range(1, nsig_max + 1)},
    }
    coord_names = ["leadtime_hour", "time"]
    data_vars_dict = {
        var_name: (coord_names, np.percentile(da_glofas, percentile_value, axis=1))
        for var_name, percentile_value in percentile_dict.items()
    }

    return xr.Dataset(
        data_vars=data_vars_dict,
        coords=dict(time=da_glofas.time, leadtime_hour=da_glofas.leadtime_hour),
    )


def read_in_ffwc():
    # Read in data from Sazzad that has forecasts
    ffwc_wl_filename = 'Bahadurabad_WL_forecast20172019.xlsx'
    ffwc_leadtime_hours = [24, 48, 72, 96, 120]

    # Need to combine the three sheets
    df_ffwc_wl_dict = pd.read_excel(
        ffwc_dir / ffwc_wl_filename,
        sheet_name=None,
        header=[1], index_col='Date')
    df_ffwc_wl = (df_ffwc_wl_dict['2017']
        .append(df_ffwc_wl_dict['2018'])
        .append(df_ffwc_wl_dict['2019'])
        .rename(columns={
        f'{leadtime_hour} hrs': f'ffwc_{int(leadtime_hour / 24)}day'
        for leadtime_hour in ffwc_leadtime_hours
    })).drop(columns=['Observed WL'])  # drop observed because we will use the mean later
    # Convert date time to just date
    df_ffwc_wl.index = df_ffwc_wl.index.floor('d')

    # Then read in the older data (goes back much futher)
    FFWC_RL_HIS_FILENAME = '2020-06-07 Water level data Bahadurabad Upper danger level.xlsx'
    ffwc_rl_name = '{}/{}'.format(ffwc_dir, FFWC_RL_HIS_FILENAME)
    df_ffwc_wl_old = pd.read_excel(ffwc_rl_name, index_col=0, header=0)
    df_ffwc_wl_old.index = pd.to_datetime(df_ffwc_wl_old.index, format='%d/%m/%y')
    df_ffwc_wl_old
    df_ffwc_wl_old = df_ffwc_wl_old[['WL']].rename(columns={'WL':
                                                                'observed'})[df_ffwc_wl_old.index < df_ffwc_wl.index[0]]
    df_ffwc_wl = pd.concat([df_ffwc_wl_old, df_ffwc_wl])

    # Read in the more recent file from Hassan
    ffwc_full_data_filename = 'SW46.9L_19-11-2020.xls'
    df_ffwc_wl_full = (pd.read_excel(ffwc_dir / ffwc_full_data_filename,
                                     index_col='DateTime')
                       .rename(columns={'WL(m)': 'observed'}))[['observed']]

    # Mutliple observations per day. Find mean and std
    df_ffwc_wl_full['date'] = df_ffwc_wl_full.index.date
    df_ffwc_wl_full = (df_ffwc_wl_full.groupby('date').agg(['mean', 'std'])
                       )['observed'].rename(columns={'mean': 'observed', 'std': 'obs_std'})
    df_ffwc_wl_full.index = pd.to_datetime(df_ffwc_wl_full.index)

    # Combine with first DF

    df_ffwc_wl = pd.merge(df_ffwc_wl_full[['obs_std']], df_ffwc_wl, left_index=True, right_index=True, how='outer')
    df_ffwc_wl.update(df_ffwc_wl_full, overwrite=False)

    return df_ffwc_wl
