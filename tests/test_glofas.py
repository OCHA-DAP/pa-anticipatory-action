# If you run black on this file you will be sorry
import numpy as np
import xarray as xr

from src.indicators.flooding import glofas

FAKE_STATIONS_LON_LAT = {
    'station_east': [1, 0],
    'station_west': [-2, 0],
    'station_north': [0, 3],
    'station_south': [0, -4]
}


def test_get_reanalysis_query():
    glofas_reanalysis = glofas.GlofasReanalysis(FAKE_STATIONS_LON_LAT)
    query = glofas_reanalysis._get_query(year=2000)
    expected_query = {
        'variable': 'river_discharge_in_the_last_24_hours',
        'format': 'grib',
        'dataset': ['consolidated_reanalysis'],
        'hyear': '2000',
        'hmonth': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'hday': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'area': glofas.get_area(FAKE_STATIONS_LON_LAT),
        'system_version': 'version_2_1'
    }
    assert query == expected_query


def test_get_forecast_query():
    glofas_forecast = glofas.GlofasForecast(FAKE_STATIONS_LON_LAT, leadtime_hours=[])
    query = glofas_forecast._get_query(year=2000, month=1, leadtime_hour=240)
    expected_query = {
        'variable': 'river_discharge_in_the_last_24_hours',
        'format': 'grib',
        'product_type': ['control_forecast', 'ensemble_perturbed_forecasts'],
        'year': '2000',
        'month': '01',
        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'area': glofas.get_area(FAKE_STATIONS_LON_LAT),
        'leadtime_hour': '240'
    }
    assert query == expected_query


def test_get_reforecast_query():
    glofas_reforecast = glofas.GlofasReforecast(FAKE_STATIONS_LON_LAT, leadtime_hours=[])
    query = glofas_reforecast._get_query(year=2000, month=1, leadtime_hour=240)
    expected_query = {
        'variable': 'river_discharge_in_the_last_24_hours',
        'format': 'grib',
        'product_type': ['control_reforecast', 'ensemble_perturbed_reforecasts'],
        'hyear': '2000',
        'hmonth': '01',
        'hday': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                 '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'area': glofas.get_area(FAKE_STATIONS_LON_LAT),
        'leadtime_hour': '240',
        'system_version': 'version_2_2'
    }
    assert query == expected_query


def test_get_area():
    assert glofas.get_area(stations_lon_lat=FAKE_STATIONS_LON_LAT, buffer=0.1) == [3.1, -2.1, -4.1, 1.1]


def test_expand_dims():
    rs = np.random.RandomState(12345)
    size_x, size_y = (10, 20)
    ds = xr.Dataset(
        data_vars={'var_a': (('x', 'y'), rs.rand(size_x, size_y))},
        coords={'x': np.arange(size_x),
                'y': np.arange(size_y)}
    )
    ds.coords['z'] = 1
    assert 'z' not in ds.dims.keys()
    ds = glofas.expand_dims(ds=ds, dataset_name='var_a', coord_names=['z', 'x', 'y'], expansion_dim=0)
    assert 'z' in ds.dims.keys()