import numpy as np
import xarray as xr

from src.indicators.flooding import glofas


def test_get_area():
    stations_lon_lat = {
        'station_east': [1, 0],
        'station_west': [-2, 0],
        'station_north': [0, 3],
        'station_south': [0, -4]
    }
    assert glofas.get_area(stations_lon_lat=stations_lon_lat, buffer=0.1) == [3.1, -2.1, -4.1, 1.1]


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
