#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Plot a single SAR tile with surrounding SCAT points.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.geodesic import Geodesic
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from shapely.geometry import Point


def plot_tile_with_scat_points(
    ds_sar,
    scat_ds,
    tile_idx,
    radius_km=50,
    sar_polygon=None,
    scat_polygon=None,
    output_path=None,
):
    """
    Plot a single SAR tile with:
    - Tile polygon (from corner coordinates)
    - Tile center
    - All SCAT points in the area
    - SCAT points within radius_km of tile center
    - The closest SCAT point (selected by colocation)
    - A magenta dashed circle showing the radius limit
    - A black line between tile center and selected SCAT point
    - A grey dashed line from the midpoint to the distance annotation

    Args:
        ds_sar (xr.Dataset): SAR dataset with scat_* variables and corner_* variables.
        scat_ds (xr.Dataset): Scatterometer dataset.
        tile_idx (tuple): Indices of the tile to plot.
            For 3D: (burst, tile_line, tile_sample)
            For 2D: (tile_line, tile_sample)
        radius_km (float): Radius around tile center to highlight SCAT points.
        sar_polygon (shapely.geometry.Polygon, optional): Full SAR footprint.
        scat_polygon (shapely.geometry.Polygon, optional): Full SCAT footprint.
        output_path (str, optional): Save figure to this path.
    """
    from slcl1butils.coloc.coloc_scatterometer import (
        get_tile_polygon,
        haversine,
        normalize_lon,
    )

    # ---- Extract tile data ----
    # Get tile center coordinates
    if "burst" in ds_sar.longitude.dims:
        b, tl, ts = tile_idx
        center_lon = ds_sar.longitude.isel(burst=b, tile_line=tl, tile_sample=ts).values
        center_lat = ds_sar.latitude.isel(burst=b, tile_line=tl, tile_sample=ts).values
        # Get SCAT info for this tile
        scat_lon = ds_sar.scat_lon.isel(burst=b, tile_line=tl, tile_sample=ts).values
        scat_lat = ds_sar.scat_lat.isel(burst=b, tile_line=tl, tile_sample=ts).values
        scat_speed = ds_sar.scat_wind_speed.isel(
            burst=b, tile_line=tl, tile_sample=ts
        ).values
        scat_dir = ds_sar.scat_wind_direction.isel(
            burst=b, tile_line=tl, tile_sample=ts
        ).values
        scat_dist = ds_sar.scat_distance_km.isel(
            burst=b, tile_line=tl, tile_sample=ts
        ).values
    else:
        tl, ts = tile_idx
        center_lon = ds_sar.longitude.isel(tile_line=tl, tile_sample=ts).values
        center_lat = ds_sar.latitude.isel(tile_line=tl, tile_sample=ts).values
        scat_lon = ds_sar.scat_lon.isel(tile_line=tl, tile_sample=ts).values
        scat_lat = ds_sar.scat_lat.isel(tile_line=tl, tile_sample=ts).values
        scat_speed = ds_sar.scat_wind_speed.isel(tile_line=tl, tile_sample=ts).values
        scat_dir = ds_sar.scat_wind_direction.isel(tile_line=tl, tile_sample=ts).values
        scat_dist = ds_sar.scat_distance_km.isel(tile_line=tl, tile_sample=ts).values

    has_valid_scat = np.isfinite(scat_lon) & np.isfinite(scat_lat)

    # ---- Get tile polygon from corner coordinates ----
    tile_polygon = get_tile_polygon(ds_sar, tile_idx)

    # ---- Filter SCAT points ----
    # All valid SCAT points
    mask_scat = np.isfinite(scat_ds.wind_speed) & np.isfinite(scat_ds.wind_direction)
    lon_scat_all = scat_ds.lon.values[mask_scat]
    lat_scat_all = scat_ds.lat.values[mask_scat]

    # SCAT points within radius (using haversine)
    distances = haversine(center_lon, center_lat, lon_scat_all, lat_scat_all)
    mask_radius = distances <= radius_km
    lon_scat_radius = lon_scat_all[mask_radius]
    lat_scat_radius = lat_scat_all[mask_radius]
    dist_scat_radius = distances[mask_radius]

    # ---- Calculate bounding box with margin ----
    margin_deg = 0.8  # ~50 km at mid-latitudes
    lon_min = center_lon - margin_deg
    lon_max = center_lon + margin_deg
    lat_min = center_lat - margin_deg
    lat_max = center_lat + margin_deg

    # Also consider tile polygon bounds if available
    if tile_polygon is not None:
        bounds = tile_polygon.bounds
        lon_min = min(lon_min, bounds[0] - 0.2)
        lon_max = max(lon_max, bounds[2] + 0.2)
        lat_min = min(lat_min, bounds[1] - 0.2)
        lat_max = max(lat_max, bounds[3] + 0.2)

    # ---- Create figure ----
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Background
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)

    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # ---- Plot magenta circle for radius limit ----
    geod = Geodesic()
    # radius_km in km -> convert to meters for geodesic.circle
    circle_points = geod.circle(
        lon=center_lon, lat=center_lat, radius=radius_km * 1000, n_samples=100  # meters
    )
    circle_lons = [p[0] for p in circle_points]
    circle_lats = [p[1] for p in circle_points]
    ax.plot(
        circle_lons,
        circle_lats,
        color="magenta",
        linestyle="--",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label=f"{radius_km} km radius",
    )

    # ---- Plot all SCAT points ----
    ax.scatter(
        lon_scat_all,
        lat_scat_all,
        s=15,
        color="gray",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        label="All SCAT points",
    )

    # ---- Plot SCAT points within radius ----
    if len(lon_scat_radius) > 0:
        ax.scatter(
            lon_scat_radius,
            lat_scat_radius,
            s=15,
            color="blue",
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            label=f"SCAT within {radius_km} km",
        )

    # ---- Plot SAR tile polygon ----
    if tile_polygon is not None:
        ax.add_geometries(
            [tile_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="green",
            linewidth=3,
            label="SAR tile",
        )

    # ---- Plot tile center ----
    ax.scatter(
        center_lon,
        center_lat,
        s=100,
        color="green",
        marker="*",
        edgecolors="black",
        linewidth=1,
        transform=ccrs.PlateCarree(),
        label="Tile center",
        zorder=10,
    )

    # ---- Plot selected SCAT point and distance line ----
    if has_valid_scat:
        # Black solid line between tile center and selected SCAT point
        ax.plot(
            [center_lon, scat_lon],
            [center_lat, scat_lat],
            color="black",
            linewidth=1.5,
            transform=ccrs.PlateCarree(),
            label="Distance line",
        )

        # Midpoint coordinates
        lon_mid = (center_lon + scat_lon) / 2
        lat_mid = (center_lat + scat_lat) / 2

        # Offset for annotation placement (adjust as needed)
        offset_lon = 0.15
        offset_lat = 0.10

        # Grey dashed line from midpoint to annotation
        ax.plot(
            [lon_mid, lon_mid + offset_lon],
            [lat_mid, lat_mid + offset_lat],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            transform=ccrs.PlateCarree(),
        )

        # Annotate distance at the end of the grey line
        ax.text(
            lon_mid + offset_lon + 0.02,
            lat_mid + offset_lat + 0.02,
            f"{scat_dist:.1f} km",
            color="black",
            fontsize=10,
            weight="bold",
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Red star for selected SCAT point
        ax.scatter(
            scat_lon,
            scat_lat,
            s=120,
            color="red",
            marker="*",
            edgecolors="black",
            linewidth=1,
            transform=ccrs.PlateCarree(),
            label="Selected SCAT",
            zorder=10,
        )

    # ---- Plot full footprints if provided ----
    if sar_polygon is not None:
        ax.add_geometries(
            [sar_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="blue",
            linewidth=1.5,
            alpha=0.5,
            label="SAR footprint",
        )
    if scat_polygon is not None:
        ax.add_geometries(
            [scat_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            alpha=0.5,
            label="SCAT footprint",
        )

    # ---- Legend and title ----
    ax.legend(loc="best", fontsize=9)

    # Title with tile info
    title_str = f"Tile {tile_idx}"
    if has_valid_scat:
        title_str += f" | Wind: {scat_speed:.1f} m/s @ {scat_dir:.0f}°"
    else:
        title_str += " | No valid wind"
    ax.set_title(title_str, fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Tile plot saved to {output_path}")
    else:
        plt.show()
