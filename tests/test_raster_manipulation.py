import xarray as xr
import pandas as pd
from src.utils_general.raster_manipulation import compute_raster_statistics
from shapely.geometry import Polygon
from pandas._testing import assert_frame_equal
import geopandas as gpd


def test_compute_raster_statistics():
    da = xr.DataArray(
        [[1, 2, 3], [4, 5, 6]],
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
    )

    d = {
        "geometry": [
            Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
            Polygon([(2, 0), (2, 2), (3, 2), (3, 0)]),
        ]
    }
    gdf = gpd.GeoDataFrame(d)

    df_stats = compute_raster_statistics(
        gdf, da, stats_list=["min", "max", "mean", "count", "sum"]
    )
    df_expected = pd.DataFrame(
        {
            "min": {0: 1.0, 1: 3.0},
            "max": {0: 5.0, 1: 6.0},
            "mean": {0: 3.0, 1: 4.5},
            "count": {0: 4, 1: 2},
            "sum": {0: 12.0, 1: 9.0},
        }
    )

    assert_frame_equal(df_stats, df_expected)


# test that sum/count is conserved
