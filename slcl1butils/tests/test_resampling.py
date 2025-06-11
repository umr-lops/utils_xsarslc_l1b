import copy

import numpy as np
import pytest
import xarray as xr

from slcl1butils.coloc.coloc import coloc_tiles_from_l1bgroup_with_raster

l1b = xr.DataArray(
    np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]]),
    dims=("tile_line", "tile_sample"),
    coords={"tile_sample": [1, 2, 3, 4], "tile_line": [5, 6]},
)
lat = xr.DataArray(
    np.array([[40.0, 41.0, 42.0, 43.0], [40.5, 41.5, 42.5, 43.5]]),
    dims=("tile_line", "tile_sample"),
    coords={"tile_sample": [1, 2, 3, 4], "tile_line": [5, 6]},
)
lon = xr.DataArray(
    np.array([[120.0, 121.0, 122.0, 123.0], [120.6, 121.7, 122.8, 123.9]]),
    dims=("tile_line", "tile_sample"),
    coords={"tile_sample": [1, 2, 3, 4], "tile_line": [5, 6]},
)
latwithNan = copy.copy(lat)
lonwithNan = copy.copy(lon)
latwithNan[1, 2] = np.nan
lonwithNan[1, 2] = np.nan
# l1b = l1b.assign_coords({'lon':lon, "latitude":lat}).rename('plop'))
#
#
N, M = 100, 100
field_ww3 = np.random.rand(N, M)
field_ww3_with_nan = copy.copy(field_ww3)
field_ww3_with_nan[1, 2] = np.nan
xa_ww3 = xr.DataArray(
    field_ww3,
    dims=("x", "y"),
    coords={"y": np.linspace(20, 50, N), "x": np.linspace(100, 150, M)},
)
xa_ww3_with_nan = xr.DataArray(
    field_ww3,
    dims=("x", "y"),
    coords={"y": np.linspace(20, 50, N), "x": np.linspace(100, 150, M)},
)
ds_ww3 = xr.Dataset({"hs": xa_ww3, "wl": xa_ww3}, attrs={"name": "xa_ww3"})
ds_ww3_with_nan = xr.Dataset(
    {"hs": xa_ww3_with_nan, "wl": xa_ww3_with_nan}, attrs={"name": "xa_ww3"}
)
# ww3 = xr.DataArray(np.random.rand(N,M), dims=('lons','lats'), coords={'lats':np.linspace(20,50,N), 'lons':np.linspace(100,150,M)})
# ww3


@pytest.mark.parametrize(
    ["l1b", "ww3", "mergeflag"],
    (
        pytest.param(
            l1b.assign_coords({"longitude": lon, "latitude": lat}).rename("plop"),
            ds_ww3,
            True,
            id="l1bwithoutNaN-ww3withoutNaN-merged",
        ),
        pytest.param(
            l1b.assign_coords({"longitude": lon, "latitude": lat}).rename("plop"),
            ds_ww3,
            False,
            id="l1bwithoutNaN-ww3withoutNaN-nomerge",
        ),
        pytest.param(
            l1b.assign_coords({"longitude": lon, "latitude": lat}).rename("plop"),
            ds_ww3_with_nan,
            False,
            id="l1bwithoutNaN-ww3withNaN-nomerge",
        ),
        pytest.param(
            l1b.assign_coords({"longitude": lonwithNan, "latitude": latwithNan}).rename(
                "plop"
            ),
            ds_ww3_with_nan,
            False,
            id="l1bwithNaN-ww3withNaN-nomerge",
        ),
    ),
)
def test_interpolation(l1b, ww3, mergeflag):
    actual = coloc_tiles_from_l1bgroup_with_raster(
        l1b_ds=l1b, raster_bb_ds=ww3, apply_merging=mergeflag
    )

    assert np.isnan(actual["hs"].values).any()
