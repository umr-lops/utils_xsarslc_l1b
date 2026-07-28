#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This module provides functions to find scatterometer matchups from a Parquet file,
read scatterometer netCDF data, and associate wind information to SAR tiles.
"""

import os
import re
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

__all__ = [
    'find_matchup_file',
    'read_scat_netcdf',
    'haversine',
    'normalize_lon',
    'find_closest_scat_point',
    'associate_sar_tiles_with_scat',
    'extract_sar_datetime',
]

def get_tile_polygon(ds_sar, tile_idx):
    """
    Create a Shapely polygon for a specific tile using corner coordinates.
    
    Args:
        ds_sar (xr.Dataset): SAR dataset with corner_longitude and corner_latitude variables.
        tile_idx (tuple): Indices (burst, tile_line, tile_sample) or (tile_line, tile_sample).
    
    Returns:
        shapely.geometry.Polygon or None: Polygon of the tile, or None if not available.
    """
    
    
    # Determine dimensions
    if 'burst' in ds_sar.corner_longitude.dims:
        # 3D: (burst, tile_line, tile_sample, c_sample, c_line)
        burst, tl, ts = tile_idx
        lon_corners = ds_sar.corner_longitude.isel(burst=burst, tile_line=tl, tile_sample=ts).values
        lat_corners = ds_sar.corner_latitude.isel(burst=burst, tile_line=tl, tile_sample=ts).values
    else:
        # 2D: (tile_line, tile_sample, c_sample, c_line)
        tl, ts = tile_idx
        lon_corners = ds_sar.corner_longitude.isel(tile_line=tl, tile_sample=ts).values
        lat_corners = ds_sar.corner_latitude.isel(tile_line=tl, tile_sample=ts).values
    
    # Reorder corners: [0, 1, 3, 2] for proper polygon (bottom-left, bottom-right, top-right, top-left)
    # c_sample dimension (0,1) and c_line dimension (0,1)
    # c_sample=0, c_line=0 -> corner 0 (bottom-left)
    # c_sample=1, c_line=0 -> corner 1 (bottom-right)
    # c_sample=1, c_line=1 -> corner 2 (top-right)
    # c_sample=0, c_line=1 -> corner 3 (top-left)
    # We need order: [0, 1, 3, 2]
    reorder = [0, 1, 3, 2]
    lon_ordered = lon_corners.flatten()[reorder]
    lat_ordered = lat_corners.flatten()[reorder]
    
    # Create polygon
    points = list(zip(lon_ordered, lat_ordered))
    return Polygon(points)
# -----------------------------------------------------------------------------
# 1. Matchup file search (Parquet)
# -----------------------------------------------------------------------------

def extract_sar_datetime(safename_l1bxsp):
    """Extract the start acquisition datetime from the SAFE name."""
    match = re.search(r"_(\d{8}T\d{6})_", safename_l1bxsp)
    if not match:
        logger.warning("Unable to extract datetime from safename: %s", safename_l1bxsp)
        return None
    datetime_str = match.group(1)
    return datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")


def extract_scat_info_from_path(scat_path_series):
    """Extract scatterometer type, satellite, and resolution from SCAT Path."""
    def _extract(path):
        satellite = None
        scat_type = None
        resolution = None
        parts = path.split('/')
        for part in parts:
            if part in ['hy-2b', 'hy-2c', 'metop-b', 'metop-c']:
                satellite = part
            elif part in ['hscat', 'ascat']:
                scat_type = part
            elif part in ['5.7km', '12.5km', '25km']:
                resolution = part
        return satellite, scat_type, resolution

    extracted = scat_path_series.apply(_extract)
    df_new = pd.DataFrame(
        extracted.tolist(),
        columns=['scat_satellite', 'scat_type', 'scat_resolution'],
        index=scat_path_series.index
    )
    return df_new


def _add_scat_info(df):
    """Add scatterometer info columns from SCAT Path."""
    if "SCAT Path" in df.columns:
        info_df = extract_scat_info_from_path(df["SCAT Path"])
        df = pd.concat([df, info_df], axis=1)
    else:
        logger.warning("Column 'SCAT Path' not found, cannot extract scatterometer info.")
    return df


def _apply_filters(df, scat_filter, identifier):
    """Apply extra filters to the DataFrame."""
    if scat_filter:
        col_mapping = {
            'type': 'scat_type',
            'satellite': 'scat_satellite',
            'resolution': 'scat_resolution'
        }
        for key, value in scat_filter.items():
            col_name = col_mapping.get(key, key)
            if col_name in df.columns:
                df = df[df[col_name] == value]
            else:
                logger.warning("Column '%s' not found, ignoring filter.", col_name)
    if df.empty:
        logger.debug("No matchup after applying filters %s for %s.", scat_filter, identifier)
        return None
    logger.info("Matchups found: %d rows for %s", len(df), identifier)
    return df


def find_matchup_by_time(sar_datetime, df, scat_filter=None, time_window_seconds=3):
    """Find matchups by time proximity."""
    if "SAR Start Acquisition datetime" not in df.columns:
        logger.error("Column 'SAR Start Acquisition datetime' missing.")
        return None

    if not pd.api.types.is_datetime64_any_dtype(df["SAR Start Acquisition datetime"]):
        df["SAR Start Acquisition datetime"] = pd.to_datetime(df["SAR Start Acquisition datetime"])

    time_diff_sec = (df["SAR Start Acquisition datetime"] - sar_datetime).abs().dt.total_seconds()
    mask_time = time_diff_sec <= time_window_seconds
    if not mask_time.any():
        logger.debug("No matchup within %.1f seconds.", time_window_seconds)
        return None

    filtered_df = df[mask_time].copy()
    filtered_df = filtered_df.assign(time_diff_sec=time_diff_sec[mask_time])
    filtered_df = filtered_df.sort_values("time_diff_sec")

    logger.info("Found %d matchups within time window (closest at %.2f s).",
                len(filtered_df), filtered_df.iloc[0]["time_diff_sec"])

    preview_cols = ["SAR File Name", "SAR Start Acquisition datetime", "SCAT Path", "time_diff_sec"]
    available = [c for c in preview_cols if c in filtered_df.columns]
    if available:
        logger.debug("Closest matchups (before filtering):\n%s",
                     filtered_df.head(3)[available].to_string(index=False))

    filtered_df = _add_scat_info(filtered_df)
    return _apply_filters(filtered_df, scat_filter, sar_datetime)


def find_matchup_file(safename_l1bxsp, parquet_file, scat_filter=None, time_window_seconds=3):
    """Search for matchups in a Parquet file for a given SAFE."""
    if not os.path.exists(parquet_file):
        logger.error("Parquet file not found: %s", parquet_file)
        return None
    
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        logger.error("Cannot read Parquet: %s", e)
        return None
    
    sar_datetime = extract_sar_datetime(safename_l1bxsp)
    if sar_datetime is None:
        return None

    # Exact match by file name
    if "SAR File Name" in df.columns:
        mask_exact = df["SAR File Name"].str.contains(
            sar_datetime.strftime("%Y%m%dT%H%M%S"), na=False
        )
        
        if mask_exact.any():
            filtered_df = df[mask_exact].copy()
            logger.debug("Matchup found by exact file name.")
            filtered_df = _add_scat_info(filtered_df)
            return _apply_filters(filtered_df, scat_filter, safename_l1bxsp)

    # Fallback: time proximity
    logger.debug("No exact match; trying time proximity with window of %.1f s.", time_window_seconds)
    return find_matchup_by_time(sar_datetime, df, scat_filter, time_window_seconds)


# -----------------------------------------------------------------------------
# 2. Scatterometer netCDF reading and colocation
# -----------------------------------------------------------------------------

def read_scat_netcdf(scat_path):
    """
    Read HY‑2 (or ASCAT) scatterometer netCDF file and return an xarray Dataset
    with unpacked variables (lon, lat, wind_speed, wind_direction, time).
    
    Longitudes are converted from 0-360° to -180-180° for easier plotting.
    """
    ds = xr.open_dataset(scat_path)

    # Unpack lon/lat with scale_factor=1e-5, add_offset=0
    # Les valeurs sont en micro-degrés (1e-5 degrés)
    lon = ds.lon #* 1e-5
    lat = ds.lat #* 1e-5
    
    # Convertir les longitudes de 0-360° vers -180-180°
    # normalize_lon est importé plus haut dans le module
    lon_normalized = normalize_lon(lon, center=0)

    # Find wind speed variable
    speed_var = 'wind_speed'
    # for var in ds.data_vars:
    #     if 'speed' in var.lower() and ('wind' in var.lower() or var == 'wind_speed'):
    #         speed_var = var
    #         break

    # Find wind direction variable (could be 'wind_dir' or 'wind_direction')
    dir_var = 'wind_dir'
    # for var in ds.data_vars:
    #     if 'dir' in var.lower() and ('wind' in var.lower() or var == 'wind_dir' or var == 'wind_direction'):
    #         dir_var = var
    #         break

    if speed_var is None or dir_var is None:
        raise ValueError(f"Could not find wind speed/direction in {scat_path}")

    # Apply scale_factor and add_offset if present
    wind_speed = ds[speed_var]
    if hasattr(ds[speed_var], 'scale_factor') and hasattr(ds[speed_var], 'add_offset'):
        wind_speed = wind_speed * ds[speed_var].scale_factor + ds[speed_var].add_offset
    elif hasattr(ds[speed_var], 'scale_factor'):
        wind_speed = wind_speed * ds[speed_var].scale_factor

    wind_direction = ds[dir_var]
    if hasattr(ds[dir_var], 'scale_factor') and hasattr(ds[dir_var], 'add_offset'):
        wind_direction = wind_direction * ds[dir_var].scale_factor + ds[dir_var].add_offset
    elif hasattr(ds[dir_var], 'scale_factor'):
        wind_direction = wind_direction * ds[dir_var].scale_factor

    # Time (usually in seconds since 1990-01-01, keep as numeric)
    time = ds.time
    scat_ds = xr.Dataset({
        'lon': (['row', 'cell'], lon_normalized.values),
        'lat': (['row', 'cell'], lat.values),
        'wind_speed': (['row', 'cell'], wind_speed.values),
        'wind_direction': (['row', 'cell'], wind_direction.values),
        'time': (['row', 'cell'], time.values),
    })
    return scat_ds


def haversine(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def normalize_lon(lon, center=0):
    """Normalize longitude to -180..180 or 0..360."""
    if center == 0:
        return ((lon + 180) % 360) - 180
    else:
        return lon % 360


# def find_closest_scat_point(scat_ds, target_lon, target_lat, max_distance_km=50):
#     """
#     Find closest scatterometer cell with valid wind data within max_distance_km.
#     Returns None if no valid point within distance.
#     """
#     # Mask valid data
#     mask = np.isfinite(scat_ds.wind_speed) & np.isfinite(scat_ds.wind_direction)
#     if not np.any(mask):
#         return None

#     lon_flat = scat_ds.lon.values[mask]
#     lat_flat = scat_ds.lat.values[mask]
#     speed_flat = scat_ds.wind_speed.values[mask]
#     dir_flat = scat_ds.wind_direction.values[mask]
#     time_flat = scat_ds.time.values[mask]

#     # Normalize longitudes to avoid anti‑meridian issues
#     target_lon_norm = normalize_lon(target_lon, center=0)
#     scat_lon_norm = normalize_lon(lon_flat, center=0)

#     # Compute distances
#     dist = haversine(target_lon_norm, target_lat, scat_lon_norm, lat_flat)
#     min_idx = np.argmin(dist)
#     min_dist = dist[min_idx]

#     if min_dist > max_distance_km:
#         return None

#     return {
#         'lon': lon_flat[min_idx],
#         'lat': lat_flat[min_idx],
#         'wind_speed': speed_flat[min_idx],
#         'wind_direction': dir_flat[min_idx],
#         'time': time_flat[min_idx],
#         'distance_km': min_dist,
#     }


def associate_sar_tiles_with_scat(ds_sar, scat_ds, counter, max_distance_km=50):
    """
    For each SAR tile centroid, find the closest scatterometer point.
    ds_sar must have variables 'longitude' and 'latitude' with either
    dimensions (burst, tile_line, tile_sample) or (tile_line, tile_sample).
    Adds new variables with the same dimensions.

    Returns:
        ds_sar (xr.Dataset): updated dataset SAR with SCAT informations
        counter (defaultdict): count the stats of association SCAT-SAR
    """
    import pandas as pd
    import re
    from datetime import datetime

    # ---- Récupération du temps SAR (sensing_time) ----
    lon_dims = ds_sar.longitude.dims
    lat_dims = ds_sar.latitude.dims
    if lon_dims != lat_dims:
        logger.warning("longitude and latitude have different dims: %s vs %s", lon_dims, lat_dims)

    if len(lon_dims) == 3 and 'burst' in lon_dims and 'tile_line' in lon_dims and 'tile_sample' in lon_dims:
        dims = ('burst', 'tile_line', 'tile_sample')
        shape = (ds_sar.dims['burst'], ds_sar.dims['tile_line'], ds_sar.dims['tile_sample'])
        lon_sar = ds_sar.longitude.values
        lat_sar = ds_sar.latitude.values
        if 'sensing_time' in ds_sar:
            sensing_time = ds_sar.sensing_time.values
        else:
            # Fallback: extraire la date du nom SAFE
            safe_name = ds_sar.attrs.get('safe', '')
            match = re.search(r'_(\d{8}T\d{6})_', safe_name)
            if match:
                start_date_str = match.group(1)
                dt = datetime.strptime(start_date_str, "%Y%m%dT%H%M%S")
                sensing_time = np.full(shape, np.datetime64(dt), dtype='datetime64[ns]')
            else:
                sensing_time = None
                logger.warning("sensing_time not available, cannot compute time delta.")
    elif len(lon_dims) == 2 and 'tile_line' in lon_dims and 'tile_sample' in lon_dims:
        dims = ('tile_line', 'tile_sample')
        shape = (ds_sar.dims['tile_line'], ds_sar.dims['tile_sample'])
        lon_sar = ds_sar.longitude.values
        lat_sar = ds_sar.latitude.values
        if 'sensing_time' in ds_sar:
            sensing_time = ds_sar.sensing_time.values
        else:
            safe_name = ds_sar.attrs.get('safe', '')
            match = re.search(r'_(\d{8}T\d{6})_', safe_name)
            if match:
                start_date_str = match.group(1)
                dt = datetime.strptime(start_date_str, "%Y%m%dT%H%M%S")
                sensing_time = np.full(shape, np.datetime64(dt), dtype='datetime64[ns]')
            else:
                sensing_time = None
                logger.warning("sensing_time not available, cannot compute time delta.")
    else:
        raise ValueError(f"Unsupported dimensions for longitude/latitude: {lon_dims}")

    # Flatten centroids
    lon_flat = lon_sar.ravel()
    lat_flat = lat_sar.ravel()
    if sensing_time is not None:
        sensing_time_flat = sensing_time.ravel()
    else:
        sensing_time_flat = None

    # Pre‑allocate output arrays with same shape
    scat_lon_out = np.full(shape, np.nan)
    scat_lat_out = np.full(shape, np.nan)
    scat_speed_out = np.full(shape, np.nan)
    scat_dir_out = np.full(shape, np.nan)
    scat_time_out = np.full(shape, np.datetime64('NaT'), dtype='datetime64[ns]')
    scat_dist_out = np.full(shape, np.nan)
    scat_time_delta_out = np.full(shape, np.nan)  # delta en secondes

    # ---- Filtrage des données SCAT valides (vent + temps) ----
    # On utilise pd.isnull pour détecter les NaT
    import pandas as pd
    mask_wind = np.isfinite(scat_ds.wind_speed) & np.isfinite(scat_ds.wind_direction)
    # Le temps peut être NaT, on le filtre
    time_valid = ~pd.isnull(scat_ds.time.values)
    mask = mask_wind & time_valid

    if not np.any(mask):
        logger.warning("No valid scatterometer data (wind + time).")
        return ds_sar, counter

    scat_lon_arr = scat_ds.lon.values[mask]
    scat_lat_arr = scat_ds.lat.values[mask]
    scat_speed_arr = scat_ds.wind_speed.values[mask]
    scat_dir_arr = scat_ds.wind_direction.values[mask]
    scat_time_arr = scat_ds.time.values[mask]  # déjà des datetime64

    # Normalize scat longitudes once
    scat_lon_norm = normalize_lon(scat_lon_arr, center=0)

    # ---- Loop over all tiles ----
    for idx in range(len(lon_flat)):
        target_lon = lon_flat[idx]
        target_lat = lat_flat[idx]
        target_lon_norm = normalize_lon(target_lon, center=0)

        # Compute distances to all scat points
        dist = haversine(target_lon_norm, target_lat, scat_lon_norm, scat_lat_arr)
        min_idx = np.argmin(dist)
        min_dist = dist[min_idx]

        if min_dist <= max_distance_km:
            counter['tile_with_scat_data'] += 1
            scat_lon_out.flat[idx] = scat_lon_arr[min_idx]
            scat_lat_out.flat[idx] = scat_lat_arr[min_idx]
            scat_speed_out.flat[idx] = scat_speed_arr[min_idx]
            scat_dir_out.flat[idx] = scat_dir_arr[min_idx]
            scat_time_out.flat[idx] = scat_time_arr[min_idx]
            scat_dist_out.flat[idx] = min_dist

            # ---- Calcul du delta temps (en secondes) ----
            if sensing_time_flat is not None:
                sar_dt = sensing_time_flat[idx]
                scat_dt = scat_time_arr[min_idx]
                # Soustraire les deux datetime64
                delta_seconds = (sar_dt - scat_dt).astype('timedelta64[s]').astype(float)
                # Vérifier que la valeur est raisonnable (par exemple < 1 heure)
                if np.abs(delta_seconds) < 3600:
                    scat_time_delta_out.flat[idx] = delta_seconds
                else:
                    # Delta trop grand -> on met NaN (probablement un décalage horaire)
                    scat_time_delta_out.flat[idx] = np.nan
        else:
            counter['tile_without_scat_data'] += 1

    # ---- Add to dataset ----
    ds_sar = ds_sar.assign(
        scat_lon=(dims, scat_lon_out),
        scat_lat=(dims, scat_lat_out),
        scat_wind_speed=(dims, scat_speed_out),
        scat_wind_direction=(dims, scat_dir_out),
        scat_time=(dims, scat_time_out),
        scat_distance_km=(dims, scat_dist_out),
        scat_time_delta_seconds=(dims, scat_time_delta_out),
    )
    ds_sar.attrs['matchup_parquet_file'] = scat_ds.attrs.get('matchup_parquet_file', '')
    ds_sar.attrs['scat_path'] = scat_ds.attrs.get('scat_path', '')
    return ds_sar, counter


# -----------------------------------------------------------------------------
# 3. Path helper for scatterometer files
# -----------------------------------------------------------------------------

def resolve_scat_path(path):
    """
    Attempt to resolve the scatterometer file path by trying common
    storage prefixes. If the file does not exist, tries to replace
    known prefixes.
    """
    if os.path.exists(path):
        return path

    # List of (old_prefix, new_prefix) pairs to try
    replacements = [
        ('/home/datawork-cersat-public/', '/legacy/project/cersat/public/'),
        ('/legacy/project/cersat/public/', '/home/datawork-cersat-public/'),
    ]
    for old, new in replacements:
        if old in path:
            new_path = path.replace(old, new)
            if os.path.exists(new_path):
                logger.info("Resolved SCAT path from %s to %s", path, new_path)
                return new_path

    logger.warning("SCAT file not found: %s", path)
    return path

def get_scat_footprint_polygon(scat_ds, buffer_deg=0.5):
    """
    Return a simple rectangle polygon from min/max of scatterometer lon/lat.
    Optionally add a buffer.
    """
    lon_flat = scat_ds.lon.values.ravel()
    lat_flat = scat_ds.lat.values.ravel()
    mask = np.isfinite(lon_flat) & np.isfinite(lat_flat)
    if not np.any(mask):
        return None
    min_lon, max_lon = np.min(lon_flat[mask]), np.max(lon_flat[mask])
    min_lat, max_lat = np.min(lat_flat[mask]), np.max(lat_flat[mask])
    # Add buffer
    min_lon -= buffer_deg
    max_lon += buffer_deg
    min_lat -= buffer_deg
    max_lat += buffer_deg
    polygon = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat)
    ])
    return polygon


def parse_tile_indices(tile_str, ds_sar):
    """
    Parse tile indices from string and validate against dataset dimensions.
    
    Args:
        tile_str (str): String like '0,3,1' or '3,1'
        ds_sar (xr.Dataset): Dataset to validate dimensions against.
    
    Returns:
        tuple: (burst_idx, tile_line_idx, tile_sample_idx) or (tile_line_idx, tile_sample_idx)
    """
    parts = [int(x) for x in tile_str.split(',')]
    
    # Check if 3D or 2D
    if 'burst' in ds_sar.longitude.dims:
        if len(parts) != 3:
            raise ValueError(f"Dataset has burst dimension, need 3 indices, got {len(parts)}")
        b, tl, ts = parts
        # Validate
        if b >= ds_sar.dims['burst']:
            raise ValueError(f"burst index {b} >= {ds_sar.dims['burst']}")
        if tl >= ds_sar.dims['tile_line']:
            raise ValueError(f"tile_line index {tl} >= {ds_sar.dims['tile_line']}")
        if ts >= ds_sar.dims['tile_sample']:
            raise ValueError(f"tile_sample index {ts} >= {ds_sar.dims['tile_sample']}")
        return (b, tl, ts)
    else:
        if len(parts) != 2:
            raise ValueError(f"Dataset has 2D dimensions, need 2 indices, got {len(parts)}")
        tl, ts = parts
        if tl >= ds_sar.dims['tile_line']:
            raise ValueError(f"tile_line index {tl} >= {ds_sar.dims['tile_line']}")
        if ts >= ds_sar.dims['tile_sample']:
            raise ValueError(f"tile_sample index {ts} >= {ds_sar.dims['tile_sample']}")
        return (tl, ts)

