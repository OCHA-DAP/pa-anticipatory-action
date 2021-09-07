import xarray as xr
import pandas as pd
from src.utils_general.raster_manipulation import (
    compute_raster_statistics,
)
from shapely.geometry import Polygon
from pandas._testing import assert_frame_equal
import geopandas as gpd


def test_compute_raster_statistics():
    da = xr.DataArray(
        [[1, 2, 3], [4, 5, 6]],
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
    ).rio.write_crs("EPSG:4326", inplace=True)

    d = {
        "name": ["hi", "bye"],
        "geometry": [
            Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
            Polygon([(2, 0), (2, 2), (3, 2), (3, 0)]),
        ],
    }
    gdf = gpd.GeoDataFrame(d)

    df_stats = compute_raster_statistics(
        gdf,
        "name",
        da,
        stats_list=["min", "max", "mean", "count", "sum"],
        percentile_list=[25, 50],
    )

    df_expected = pd.DataFrame(
        {
            "min_name": {0: 1, 1: 3},
            "max_name": {0: 5, 1: 6},
            "mean_name": {0: 3.0, 1: 4.5},
            "count_name": {0: 4, 1: 2},
            "sum_name": {0: 12, 1: 9},
            # quantile is computed with linear interpolation see
            # https://stackoverflow.com/questions/48799231/numpy-percentiles-with-linear-interpolation-wrong-value/48799350
            "25quant_name": {0: 1.75, 1: 3.75},
            "50quant_name": {0: 3.0, 1: 4.5},
            "name": {0: "hi", 1: "bye"},
        },
    )

    # if check_dtype=True it catches float vs int which
    # is not needed
    assert_frame_equal(df_stats, df_expected, check_dtype=False)

    # only should apply in cases the gdf exactly covers
    # the da, which is with this test data
    sum_gdf = 21
    assert df_stats["sum_name"].sum() == sum_gdf
