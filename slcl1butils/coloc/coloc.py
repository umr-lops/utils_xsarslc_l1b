import pdb
from slcl1butils.raster_readers import ecmwf_0100_1h
from slcl1butils.raster_readers import ww3_global_yearly_3h
from slcl1butils.raster_readers import resource_strftime
import sys, os
from slcl1butils.get_polygons_from_l1b import get_swath_tiles_polygons_from_l1bgroup
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import xarray as xr
from datatree import DataTree
import logging


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
    raster_ds = raster_ds.transpose('y', 'x')

    # ensure coords are increasing ( for RectBiVariateSpline )
    for coord in ['x', 'y']:
        if raster_ds[coord].values[-1] < raster_ds[coord].values[0]:
            raster_ds = raster_ds.reindex({coord: raster_ds[coord][::-1]})

    # from lon/lat box in xsar dataset, get the corresponding box in raster_ds (by index)
    ilon_range = [
        np.searchsorted(raster_ds.x.values, lon_range[0]),
        np.searchsorted(raster_ds.x.values, lon_range[1])
    ]
    ilat_range = [
        np.searchsorted(raster_ds.y.values, lat_range[0]),
        np.searchsorted(raster_ds.y.values, lat_range[1])
    ]
    # enlarge the raster selection range, for correct interpolation
    if (enlarge):
        ilon_range, ilat_range = [[rg[0] - step, rg[1] + step] for rg in (ilon_range, ilat_range)]

    # select the xsar box in the raster
    raster_ds = raster_ds.isel(x=slice(*ilon_range), y=slice(*ilat_range))

    return raster_ds


def coloc_tiles_from_l1bgroup_with_raster(l1b_ds, raster_bb_ds, apply_merging=True, method=1):
    """

    Args:
        l1b_ds:
        raster_bb_ds:
        apply_merging:
        method:

    Returns:

    """

    if method == 0:
        #
        # Get the tile polygons and corresponding coordinates
        polygons, coordinates = get_swath_tiles_polygons_from_l1bgroup(l1b_ds, swath_only=False)
        poly_tiles = polygons['tiles']
        # For each tile find the ancillary data
        anc_ds = []
        for cpt, poly_tile in enumerate(poly_tiles):
            #
            # Extract the raster in a boundix box 
            raster_bb_tile_ds = raster_cropping_in_polygon_bounding_box(poly_tile, raster_bb_ds, enlarge=True, step=1)
            # Compute the mean wind component in the tile
            U10 = np.nanmean(raster_bb_tile_ds['U10'].values)
            V10 = np.nanmean(raster_bb_tile_ds['V10'].values)
            #
            _anc_ds = xr.Dataset(data_vars={'U10': U10, 'V10': V10}, coords={'burst': coordinates['ibursts'][cpt],
                                                                             'tile_sample':
                                                                                 coordinates['itile_samples'][cpt],
                                                                             'tile_line': coordinates['itile_lines'][
                                                                                 cpt]}).expand_dims(
                ['burst', 'tile_sample', 'tile_line'])
            anc_ds.append(_anc_ds)
        raster_mapped = xr.combine_by_coords(anc_ds)
        merged_raster_mapped = xr.merge([l1b_ds, raster_mapped])

    if method == 1:
        lons = l1b_ds.longitude.values.flatten()
        lats = l1b_ds.latitude.values.flatten()

        mapped_ds_list = []
        for var in raster_bb_ds:
            raster_da = raster_bb_ds[var]  # .chunk(raster_bb_ds[var].shape)
            # upscale in original projection using interpolation
            # in most cases, RectBiVariateSpline give better results, but can't handle Nans
            # if np.any(np.isnan(raster_da)):
            upscaled_da = raster_da.interp(x=lons, y=lats)

            # else:
            #    upscaled_da = map_blocks_coords(
            #        xr.DataArray(dims=['y', 'x'], coords={'x': lons, 'y': lats}).chunk(1000),
            #        RectBivariateSpline(
            #            raster_da.y.values,
            #            raster_da.x.values,
            #            raster_da.values,
            #            kx=3, ky=3
            #        )
            #    )
            upscaled_da.name = var
            # interp upscaled_da on sar grid
            # mapped_ds_list.append(
            #     upscaled_da.interp(
            #         x=l1b_ds.longitude,
            #         y=l1b_ds.latitude
            #     ).drop_vars(['x', 'y'])
            # )
            # fix double interpolation 11 april 2023
            mapped_ds_list.append(
                upscaled_da.drop_vars(['x', 'y'])
            )
        raster_mapped = xr.merge(mapped_ds_list)
        merged_raster_mapped = xr.merge([l1b_ds, raster_mapped])

    if apply_merging:
        return merged_raster_mapped
    else:
        return raster_mapped


def do_coloc_L1B_with_raster_SAFE(full_safe_file, ancillary_list, skip=True):
    """

    Args:
        full_safe_file: str
        ancillary_list: list
        skip: bool [optional]

    Returns:

    """
    # TODO: see whether we keep or not this method
    safe_file = os.path.basename(full_safe_file)
    safe_path = os.path.dirname(full_safe_file) + '/'
    sth = 1

    files = glob(os.path.join(safe_path, safe_file, '*_L1B_*nc'))
    for _file in files:

        # ====================
        # FILE OUT
        # ====================

        if sth > 0:
            #
            #
            # Output file directory
            pathout_root = safe_path.replace('l1b', 'l1e')
            # print(~os.path.isdir(pathout_root))
            if (os.path.isdir(pathout_root) == False):
                os.system('mkdir ' + pathout_root)
            pathout = pathout_root + safe_file + '/'
            # print(os.path.isdir(pathout))
            if (os.path.isdir(pathout) == False):
                os.system('mkdir ' + pathout)
            #
            # Ouput filename
            fileout = os.path.basename(_file).replace('L1B', 'L1D')
            if skip and os.path.isfile(pathout + fileout):
                print('File already created and available on disk')
                print('Here: ' + pathout + fileout)
                continue
            logging.info('File out: %s', pathout + fileout)

        ds_inter = []
        ds_intra = []
        for ancillary in ancillary_list:

            print(ancillary['name'])

            # For each L1B
            burst_type = 'intra'
            l1b_ds = xr.open_dataset(_file, group=burst_type + 'burst')
            ## Check if the ancillary data can be found
            sar_date = datetime.strptime(str.split(l1b_ds.attrs['start_date'], '.')[0], '%Y-%m-%d %H:%M:%S')
            closest_date, filename = resource_strftime(ancillary['resource'], step=ancillary['step'], date=sar_date)
            if (len(glob(filename)) != 1):
                continue
            # Getting the raster from anxillary data
            if (ancillary['name'] == 'ecmwf_0100_1h'):
                raster_ds = ecmwf_0100_1h(filename)
            if (ancillary['name'] == 'ww3_global_yearly_3h'):
                raster_ds = ww3_global_yearly_3h(filename, closest_date)

            # Get the polygons of the swath data
            polygons = get_swath_tiles_polygons_from_l1bgroup(l1b_ds, swath_only=True)
            # Crop the raster to the swath bounding box limit
            raster_bb_ds = raster_cropping_in_polygon_bounding_box(polygons['swath'][0], raster_ds)

            # Loop on the grid in the product
            burst_types = ['intra', 'inter']
            for burst_type in burst_types:
                # Define the dataset to work on
                # get the mapped raster onto swath grid for each tile
                if (burst_type == 'intra'):
                    l1b_ds_intra = xr.open_dataset(_file, group=burst_type + 'burst')
                    _ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_intra, raster_bb_ds, apply_merging=False)
                    ds_intra.append(_ds)
                else:
                    l1b_ds_inter = xr.open_dataset(_file, group=burst_type + 'burst')
                    _ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_inter, raster_bb_ds, apply_merging=False)
                    ds_inter.append(_ds)

        # return  ds_inter, l1b_ds_inter
        # Merging the datasets
        ds_intra = xr.merge([l1b_ds_intra, xr.merge(ds_intra)])
        ds_inter = xr.merge([l1b_ds_inter, xr.merge(ds_inter)])
        #  Building the output datatree
        # Data Tree for outputs
        dt = DataTree()
        dt['intraburst'] = DataTree(data=ds_intra)
        dt['interburst'] = DataTree(data=ds_inter)
        # return dt, ds_intra, ds_inter

        # Saving the results in netCDF
        # ====================
        # FILE OUT
        # ====================
        sth = 1
        if (sth > 0):
            # Saving the results in netCDF
            dt.to_netcdf(pathout + fileout)

    return 0
