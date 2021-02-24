import numpy as np
import xarray as xr

from src.indicators.flooding import glofas


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
