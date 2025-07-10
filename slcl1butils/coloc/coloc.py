import numpy as np
import xarray as xr


def raster_cropping_in_polygon_bounding_box(poly_tile, raster_ds, enlarge=True, step=1):
    """

    Parameters
    ----------
    poly_tile
    raster_ds
    enlarge
    step

    Returns
    -------

    """

    lon1, lat1, lon2, lat2 = poly_tile.exterior.bounds
    lon_range = [lon1, lon2]
    lat_range = [lat1, lat2]

    # ensure dims ordering
    raster_ds = raster_ds.transpose("y", "x")

    # ensure coords are increasing ( for RectBiVariateSpline )
    for coord in ["x", "y"]:
        if raster_ds[coord].values[-1] < raster_ds[coord].values[0]:
            raster_ds = raster_ds.reindex({coord: raster_ds[coord][::-1]})

    # from lon/lat box in xsar dataset, get the corresponding box in raster_ds (by index)
    ilon_range = [
        np.searchsorted(raster_ds.x.values, lon_range[0]),
        np.searchsorted(raster_ds.x.values, lon_range[1]),
    ]
    ilat_range = [
        np.searchsorted(raster_ds.y.values, lat_range[0]),
        np.searchsorted(raster_ds.y.values, lat_range[1]),
    ]
    # enlarge the raster selection range, for correct interpolation
    if enlarge:
        ilon_range, ilat_range = [
            [rg[0] - step, rg[1] + step] for rg in (ilon_range, ilat_range)
        ]

    # select the xsar box in the raster
    raster_ds = raster_ds.isel(x=slice(*ilon_range), y=slice(*ilat_range))

    return raster_ds


def coloc_tiles_from_l1bgroup_with_raster(l1b_ds, raster_bb_ds, apply_merging=True):
    """

    Args:
        l1b_ds:
        raster_bb_ds:
        apply_merging:
        method:

    Returns:

    """
    latsar = l1b_ds.latitude
    lonsar = l1b_ds.longitude
    mapped_ds_list = []
    for var in raster_bb_ds:
        if var not in ["forecast_hour"]:
            raster_da = raster_bb_ds[var]
            upscaled_da = raster_da
            upscaled_da.name = var
            upscaled_da = upscaled_da.astype(
                float
            )  # added by agrouaze to fix TypeError: No matching signature found at interpolation
            projected_field = upscaled_da.interp(
                x=lonsar, y=latsar, assume_sorted=False
            ).drop_vars(["x", "y"])
            projected_field.attrs["source"] = raster_bb_ds.attrs.get("name", "unknown")
            mapped_ds_list.append(projected_field)
    raster_mapped = xr.merge(mapped_ds_list)

    if apply_merging:
        merged_raster_mapped = xr.merge([l1b_ds, raster_mapped])
        return merged_raster_mapped
    else:
        return raster_mapped
