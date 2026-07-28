"""
Unit tests for slcl1butils.coloc.coloc_scatterometer
"""

import os
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from slcl1butils.coloc import coloc_scatterometer as coloc


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_safe_name():
    return "S1A_IW_XSP__1SDV_20250321T164835_20250321T164900_012345_006789_6672_A23.SAFE"


@pytest.fixture
def mock_scat_ds():
    """Create a dummy scatterometer xarray Dataset with coordinates around Europe."""
    nrows, ncols = 10, 10
    # Lon between -30 and -20, lat between 40 and 50 (close to SAR fixture)
    lon = np.linspace(-30, -20, nrows * ncols).reshape(nrows, ncols)
    lat = np.linspace(40, 50, nrows * ncols).reshape(nrows, ncols)
    wind_speed = np.full((nrows, ncols), 5.0)
    wind_direction = np.full((nrows, ncols), 90.0)
    time = np.full((nrows, ncols), np.datetime64("2025-03-21T16:50:00"), dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "lon": (["row", "cell"], lon),
            "lat": (["row", "cell"], lat),
            "wind_speed": (["row", "cell"], wind_speed),
            "wind_direction": (["row", "cell"], wind_direction),
            "time": (["row", "cell"], time),
        }
    )
    return ds


@pytest.fixture
def mock_scat_ds_with_dir():
    """Same as mock_scat_ds but using 'wind_dir' instead of 'wind_direction' for read_scat_netcdf test."""
    nrows, ncols = 10, 10
    lon = np.linspace(-30, -20, nrows * ncols).reshape(nrows, ncols)
    lat = np.linspace(40, 50, nrows * ncols).reshape(nrows, ncols)
    wind_speed = np.full((nrows, ncols), 5.0)
    wind_dir = np.full((nrows, ncols), 90.0)
    time = np.full((nrows, ncols), np.datetime64("2025-03-21T16:50:00"), dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "lon": (["row", "cell"], lon),
            "lat": (["row", "cell"], lat),
            "wind_speed": (["row", "cell"], wind_speed),
            "wind_dir": (["row", "cell"], wind_dir),
            "time": (["row", "cell"], time),
        }
    )
    return ds


@pytest.fixture
def mock_sar_ds_2d():
    """Create a 2D SAR dataset with coordinates in the same region as SCAT."""
    nline, nsample = 5, 4
    # Lon between -28 and -22, lat between 42 and 48 (close to SCAT grid)
    lon = np.linspace(-28, -22, nline * nsample).reshape(nline, nsample)
    lat = np.linspace(42, 48, nline * nsample).reshape(nline, nsample)
    ds = xr.Dataset(
        {
            "longitude": (["tile_line", "tile_sample"], lon),
            "latitude": (["tile_line", "tile_sample"], lat),
        },
        coords={
            "tile_line": np.arange(nline),
            "tile_sample": np.arange(nsample),
        },
        attrs={"safe": "S1A_IW_XSP__1SDV_20250321T164835_20250321T164900_012345_006789_6672_A23.SAFE"},
    )
    return ds


@pytest.fixture
def mock_parquet_dataframe():
    """Create a dummy pandas DataFrame with a single exact match."""
    df = pd.DataFrame(
        {
            "SCAT Path": ["/path/to/scat.nc"],
            "SAR File Name": ["S1A_IW_XSP__1SDV_20250321T164835_...SAFE"],
            "SAR Start Acquisition datetime": [pd.Timestamp("2025-03-21T16:48:35")],
        }
    )
    return df


# ------------------------------------------------------------------------------
# Tests for extract_sar_datetime
# ------------------------------------------------------------------------------

def test_extract_sar_datetime_valid(mock_safe_name):
    dt = coloc.extract_sar_datetime(mock_safe_name)
    assert dt == datetime(2025, 3, 21, 16, 48, 35)


def test_extract_sar_datetime_invalid():
    assert coloc.extract_sar_datetime("invalid_name") is None


# ------------------------------------------------------------------------------
# Tests for normalize_lon
# ------------------------------------------------------------------------------

def test_normalize_lon_center_0():
    assert coloc.normalize_lon(190, center=0) == -170
    assert coloc.normalize_lon(-10, center=0) == -10
    assert coloc.normalize_lon(10, center=0) == 10


def test_normalize_lon_center_360():
    assert coloc.normalize_lon(190, center=360) == 190
    assert coloc.normalize_lon(-10, center=360) == 350


# ------------------------------------------------------------------------------
# Tests for haversine
# ------------------------------------------------------------------------------

def test_haversine_known_distance():
    # Distance from Paris (2.35°E, 48.86°N) to London (0.13°W, 51.51°N) ~ 344 km
    lon1, lat1 = 2.35, 48.86
    lon2, lat2 = -0.13, 51.51
    dist = coloc.haversine(lon1, lat1, lon2, lat2)
    assert 330 < dist < 360  # approx


def test_haversine_zero():
    lon, lat = 10, 20
    assert coloc.haversine(lon, lat, lon, lat) == 0.0


# ------------------------------------------------------------------------------
# Tests for resolve_scat_path (with mocking)
# ------------------------------------------------------------------------------

def test_resolve_scat_path_exists(monkeypatch):
    def mock_exists(path):
        return True
    monkeypatch.setattr(os.path, "exists", mock_exists)
    path = "/some/path/file.nc"
    assert coloc.resolve_scat_path(path) == path


def test_resolve_scat_path_not_found_but_replacement(monkeypatch):
    def mock_exists(path):
        if path == "/home/datawork-cersat-public/file.nc":
            return False
        if path == "/legacy/project/cersat/public/file.nc":
            return True
        return False
    monkeypatch.setattr(os.path, "exists", mock_exists)
    path = "/home/datawork-cersat-public/file.nc"
    expected = "/legacy/project/cersat/public/file.nc"
    assert coloc.resolve_scat_path(path) == expected


def test_resolve_scat_path_still_not_found(monkeypatch, caplog):
    def mock_exists(path):
        return False
    monkeypatch.setattr(os.path, "exists", mock_exists)
    path = "/some/path/file.nc"
    with caplog.at_level("WARNING"):
        result = coloc.resolve_scat_path(path)
    assert result == path
    assert "SCAT file not found" in caplog.text


# ------------------------------------------------------------------------------
# Tests for find_matchup_file (with mocking)
# ------------------------------------------------------------------------------

def test_find_matchup_file_parquet_not_exist(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda x: False)
    result = coloc.find_matchup_file("dummy", "/no/parquet.parquet")
    assert result is None


def test_find_matchup_file_exact_match(monkeypatch, mock_parquet_dataframe, mock_safe_name):
    def mock_exists(path):
        return True
    def mock_read_parquet(path):
        return mock_parquet_dataframe
    monkeypatch.setattr(os.path, "exists", mock_exists)
    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

    def fake_apply(df, scat_filter, identifier):
        return df
    monkeypatch.setattr(coloc, "_apply_filters", fake_apply)

    result = coloc.find_matchup_file(mock_safe_name, "/dummy.parquet")
    assert result is not None
    assert len(result) == 1  # now a single exact match
    assert result.iloc[0]["SAR File Name"].startswith("S1A_IW_XSP")


def test_find_matchup_file_time_proximity(monkeypatch, mock_safe_name):
    # Create a DataFrame with a file that has a close time but not exact filename
    df = pd.DataFrame({
        "SCAT Path": ["/path/to/scat.nc"],
        "SAR File Name": ["S1A_IW_GRDH_1SDV_20250321T164835_...SAFE"],
        "SAR Start Acquisition datetime": [pd.Timestamp("2025-03-21T16:48:35")],
    })
    def mock_exists(path):
        return True
    def mock_read_parquet(path):
        return df
    monkeypatch.setattr(os.path, "exists", mock_exists)
    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

    def fake_apply(df, scat_filter, identifier):
        return df
    monkeypatch.setattr(coloc, "_apply_filters", fake_apply)

    result = coloc.find_matchup_file(mock_safe_name, "/dummy.parquet")
    assert result is not None
    assert len(result) == 1


# ------------------------------------------------------------------------------
# Tests for read_scat_netcdf (mock xr.open_dataset)
# ------------------------------------------------------------------------------

def test_read_scat_netcdf(monkeypatch, mock_scat_ds_with_dir):
    # Mock xr.open_dataset to return a dataset with 'wind_dir' and 'wind_speed'
    def mock_open_dataset(path):
        return mock_scat_ds_with_dir
    monkeypatch.setattr(xr, "open_dataset", mock_open_dataset)

    ds = coloc.read_scat_netcdf("dummy.nc")
    assert "lon" in ds
    assert "lat" in ds
    assert "wind_speed" in ds
    assert "wind_direction" in ds  # the function renames it to 'wind_direction'
    assert "time" in ds
    # Check that lon values are within [-180, 180]
    assert np.all((ds.lon.values >= -180) & (ds.lon.values <= 180))


# ------------------------------------------------------------------------------
# Tests for associate_sar_tiles_with_scat
# ------------------------------------------------------------------------------

def test_associate_sar_tiles_with_scat_2d(mock_sar_ds_2d, mock_scat_ds):
    counter = defaultdict(int)
    ds_out, counter = coloc.associate_sar_tiles_with_scat(
        mock_sar_ds_2d, mock_scat_ds, counter, max_distance_km=5000  # large enough to match all
    )
    expected_vars = [
        "scat_lon", "scat_lat", "scat_wind_speed", "scat_wind_direction",
        "scat_time", "scat_distance_km", "scat_time_delta_seconds"
    ]
    for var in expected_vars:
        assert var in ds_out
    assert ds_out.scat_lon.dims == ("tile_line", "tile_sample")
    assert ds_out.scat_lon.shape == (5, 4)
    valid_mask = np.isfinite(ds_out.scat_lon.values)
    assert valid_mask.sum() > 0
    assert counter["tile_with_scat_data"] > 0


def test_associate_sar_tiles_with_scat_no_valid_scat(mock_sar_ds_2d, mock_scat_ds):
    # Create a scat_ds with all invalid wind (NaNs)
    scat_ds_bad = mock_scat_ds.copy()
    scat_ds_bad["wind_speed"].values[:] = np.nan
    scat_ds_bad["wind_direction"].values[:] = np.nan
    counter = defaultdict(int)
    ds_out, counter = coloc.associate_sar_tiles_with_scat(
        mock_sar_ds_2d, scat_ds_bad, counter, max_distance_km=5000
    )
    # The function should return the original ds (no changes)
    assert "scat_lon" not in ds_out  # not added
    assert counter["tile_with_scat_data"] == 0


def test_associate_sar_tiles_with_scat_time_delta(mock_sar_ds_2d, mock_scat_ds):
    ds = mock_sar_ds_2d.copy()
    nline, nsample = 5, 4
    sensing_time = np.full((nline, nsample), np.datetime64("2025-03-21T16:48:35"), dtype="datetime64[ns]")
    ds["sensing_time"] = (["tile_line", "tile_sample"], sensing_time)
    counter = defaultdict(int)
    ds_out, counter = coloc.associate_sar_tiles_with_scat(
        ds, mock_scat_ds, counter, max_distance_km=5000
    )
    assert "scat_time_delta_seconds" in ds_out
    # At least some deltas should be finite (since times are close)
    finite_deltas = np.isfinite(ds_out.scat_time_delta_seconds.values)
    assert finite_deltas.sum() > 0


# ------------------------------------------------------------------------------
# Tests for get_scat_footprint_polygon
# ------------------------------------------------------------------------------

def test_get_scat_footprint_polygon(mock_scat_ds):
    poly = coloc.get_scat_footprint_polygon(mock_scat_ds, buffer_deg=1.0)
    assert isinstance(poly, Polygon)
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    assert min_lon >= -31
    assert max_lon <= -19


# ------------------------------------------------------------------------------
# Tests for parse_tile_indices
# ------------------------------------------------------------------------------

def test_parse_tile_indices_2d(mock_sar_ds_2d):
    idx = coloc.parse_tile_indices("2,1", mock_sar_ds_2d)
    assert idx == (2, 1)


def test_parse_tile_indices_3d():
    # Create a mock 3D dataset
    ds = xr.Dataset(
        {"longitude": (["burst", "tile_line", "tile_sample"], np.zeros((2,3,4)))},
        coords={"burst": [0,1], "tile_line": [0,1,2], "tile_sample": [0,1,2,3]}
    )
    idx = coloc.parse_tile_indices("1,2,3", ds)
    assert idx == (1, 2, 3)


def test_parse_tile_indices_invalid(mock_sar_ds_2d):
    with pytest.raises(ValueError, match="Dataset has 2D dimensions, need 2 indices"):
        coloc.parse_tile_indices("1,2,3", mock_sar_ds_2d)
    with pytest.raises(ValueError, match="tile_line index 5 >= 5"):
        coloc.parse_tile_indices("5,1", mock_sar_ds_2d)