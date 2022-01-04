import numpy as np
import pandas as pd
import xarray as xr

from src.indicators.drought.chirps_rainfallobservations import (
    _compute_tercile_bounds,
    _compute_tercile_category,
)


def test_tercile_assignment():
    """Test to ensure terciles are assigned correctly."""
    da = xr.DataArray(
        np.arange(10).reshape(5, 2),
        dims={
            "lat": np.arange(5),
            "lon": np.arange(2),
        },
    )
    da_bounds = _compute_tercile_bounds(da)
    assert np.array_equal(da_bounds.values, [3, 6])
    da_bn, da_no, da_an = _compute_tercile_category(da, da_bounds)
    assert np.array_equal(
        da_bn.values, np.array([[True] * 4 + [False] * 6]).reshape(5, 2)
    )
    assert np.array_equal(
        da_no.values,
        np.array([[False] * 4 + [True] * 2 + [False] * 4]).reshape(5, 2),
    )
    assert np.array_equal(
        da_an.values, np.array([[False] * 6 + [True] * 4]).reshape(5, 2)
    )


def test_tercile_groupby():
    """Test tercile computation when doing a groupby."""
    da = xr.DataArray(
        np.arange(20).reshape(2, 10),
        coords=[
            np.arange(2),
            pd.date_range("1/1/2000", "1/10/2000", freq="D"),
        ],
        dims=["lon", "time"],
    )
    da_bounds = _compute_tercile_bounds(da.groupby(da.time.dt.month))
    assert np.array_equal(da_bounds.squeeze().values, [[3, 6], [13, 16]])
    da_bn, da_no, da_an = _compute_tercile_category(da, da_bounds)
    assert np.array_equal(
        da_bn.squeeze().values, np.array([[True] * 4 + [False] * 6] * 2)
    )
    assert np.array_equal(
        da_no.squeeze().values,
        np.array([[False] * 4 + [True] * 2 + [False] * 4] * 2),
    )
    assert np.array_equal(
        da_an.squeeze().values, np.array([[False] * 6 + [True] * 4] * 2)
    )
