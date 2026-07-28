#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Plotting functions for scatterometer colocation results.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


def plot_colocation(
    ds_sar,
    scat_ds,
    title="Scatterometer colocation",
    figsize=(14, 6),
    sar_polygon=None,
    scat_polygon=None,
    output_path=None,
):
    """
    Create two maps side by side:4E
    - Left: SAR tile centroids colored by validity of scat wind (green=valid, gray=invalid)
            plus scatterometer grid points (small gray dots)
    - Right: Wind vectors (quiver) at SAR centroids with valid wind,
             gray dots for invalid tiles. Vector colors depend on wind speed.

    Args:
        ds_sar (xr.Dataset): SAR dataset with added scat_* variables.
        scat_ds (xr.Dataset): Scatterometer dataset (for grid points).
        title (str): Overall title.
        figsize (tuple): Figure size.
        sar_polygon (shapely.geometry.Polygon, optional): Polygon of SAR footprint.
        scat_polygon (shapely.geometry.Polygon, optional): Polygon of scatterometer footprint.
        output_path (str, optional): If provided, save the figure to this path.
    """
    # Extract data
    lon_sar = ds_sar.longitude.values.ravel()
    lat_sar = ds_sar.latitude.values.ravel()
    scat_lon = ds_sar.scat_lon.values.ravel()
    scat_lat = ds_sar.scat_lat.values.ravel()
    scat_speed = ds_sar.scat_wind_speed.values.ravel()
    scat_dir = ds_sar.scat_wind_direction.values.ravel()
    valid = np.isfinite(scat_speed) & np.isfinite(scat_dir)

    # Scatterometer grid points (valid winds)
    mask_scat = np.isfinite(scat_ds.wind_speed) & np.isfinite(scat_ds.wind_direction)
    lon_scat = scat_ds.lon.values[mask_scat]
    lat_scat = scat_ds.lat.values[mask_scat]

    # Calculate SAR bounding box with margin
    lon_sar_valid = lon_sar[np.isfinite(lon_sar)]
    lat_sar_valid = lat_sar[np.isfinite(lat_sar)]
    if len(lon_sar_valid) > 0 and len(lat_sar_valid) > 0:
        margin = 5.0  # degrees
        lon_min = np.min(lon_sar_valid) - margin
        lon_max = np.max(lon_sar_valid) + margin
        lat_min = np.min(lat_sar_valid) - margin
        lat_max = np.max(lat_sar_valid) + margin
    else:
        # Fallback: use SAR polygon or global extent
        if sar_polygon is not None:
            bounds = sar_polygon.bounds
            lon_min, lat_min, lon_max, lat_max = bounds
            lon_min -= margin
            lon_max += margin
            lat_min -= margin
            lat_max += margin
        else:
            lon_min, lon_max = -180, 180
            lat_min, lat_max = -90, 90

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Common settings for both axes
    for ax in (ax1, ax2):
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        # Set extent to SAR bounding box with margin
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    # --- Left map: validity and scat grid ---
    ax1.scatter(
        lon_scat,
        lat_scat,
        s=1,
        color="gray",
        alpha=0.3,
        transform=ccrs.PlateCarree(),
        label="Scat grid",
    )
    colors = np.where(valid, "green", "gray")
    ax1.scatter(
        lon_sar,
        lat_sar,
        s=8,
        c=colors,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        label="SAR tiles",
    )

    if sar_polygon is not None:
        ax1.add_geometries(
            [sar_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
            label="SAR footprint",
        )
    if scat_polygon is not None:
        ax1.add_geometries(
            [scat_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            label="SCAT footprint",
        )

    ax1.set_title("Validity of matched wind")
    ax1.legend(loc="best")

    # --- Right map: wind vectors with color by speed ---
    invalid_mask = ~valid
    if np.any(invalid_mask):
        ax2.scatter(
            lon_sar[invalid_mask],
            lat_sar[invalid_mask],
            s=8,
            color="gray",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            label="No valid wind",
        )

    if np.any(valid):
        # Convert wind direction to u/v components
        dir_rad = np.radians(scat_dir[valid])
        u = -scat_speed[valid] * np.sin(dir_rad)
        v = -scat_speed[valid] * np.cos(dir_rad)

        # Create colorbar based on wind speed (fixed 0-20 m/s)
        speeds = scat_speed[valid]

        # Use quiver with colormap and fixed norm
        norm = plt.Normalize(vmin=0, vmax=20)
        q = ax2.quiver(
            lon_sar[valid],
            lat_sar[valid],
            u,
            v,
            speeds,  # color by wind speed
            scale=300,
            scale_units="width",
            cmap="viridis",
            norm=norm,
            width=0.003,
            headwidth=3,
            transform=ccrs.PlateCarree(),
        )

        # Add colorbar with fixed limits
        cbar = plt.colorbar(q, ax=ax2, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label("Wind speed (m/s)")
        cbar.set_ticks(np.arange(0, 21, 5))  # ticks at 0, 5, 10, 15, 20
    else:
        ax2.scatter(
            lon_sar,
            lat_sar,
            s=8,
            color="gray",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            label="SAR tiles (no wind)",
        )

    if sar_polygon is not None:
        ax2.add_geometries(
            [sar_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
            label="SAR footprint",
        )
    if scat_polygon is not None:
        ax2.add_geometries(
            [scat_polygon],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            label="SCAT footprint",
        )

    ax2.set_title("Wind vectors (color = speed)")
    ax2.legend(loc="best")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
