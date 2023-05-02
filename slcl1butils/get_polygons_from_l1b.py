#!/usr/bin/env python
import numpy as np
from shapely import geometry
from shapely import wkt
import xarray as xr
import logging

polygons_varnames = ['swath', 'tiles', 'bursts']
def get_swath_tiles_polygons_from_l1bgroup(l1b_ds, swath_only=False, ik=0,burst_type='intra', **kwargs):
    """
    get polygons for a given group of L1B SAR IFREMER product
    Args:
        l1b_ds: xarray.Dataset
        swath_only: bool [optional]
        ik: int [optional], index of wave number selected for variable displayed such as 'cwave' or 'imacs'
        burst_type: str intra or inter [default='intra']
    Keyword Args:
        kwargs (dict): optional keyword arguments : variable_names (list), is valid entry
    Returns:

    """

    polygons = {}
    coordinates = {}
    variables = {}
    poly_tiles = []
    ibursts = []
    poly_bursts = []
    itile_samples = []
    itile_lines = []
    variable_names = kwargs.get('variable_names', None)
    if variable_names:
        if variable_names[0] is not None:
            for variable_name in variable_names:
                variables[variable_name] = []

    # Swath_polynom
    poly_swath = wkt.loads(l1b_ds.attrs['footprint'])
    polygons['swath'] = [poly_swath]
    if swath_only:
        return polygons, coordinates, variables
    logging.debug('l1b_ds : %s',l1b_ds)
    # Tiles polynom & variables
    for iburst in l1b_ds['burst'].values:
        if burst_type == 'intra':
            if 'burst_corner_longitude' in l1b_ds:
                burst_lon_corners = l1b_ds['burst_corner_longitude'].sel(burst=iburst).values.flatten().tolist()
                burst_lat_corners = l1b_ds['burst_corner_latitude'].sel(burst=iburst).values.flatten().tolist()
                if np.sum((~np.isfinite(burst_lon_corners))) == 0:
                    # Define the burst polygons
                    order = [0, 1, 3, 2, 0]
                    poly_burst = geometry.Polygon(np.stack([np.array(burst_lon_corners)[order], np.array(burst_lat_corners)[order]]).T)
                    #pts_burst = [geometry.Point(burst_lon_corners[cpt],burst_lat_corners[cpt]) for cpt,_ in enumerate(burst_lon_corners)]
                    #pts_burst = [pts_burst[0], pts_burst[1], pts_burst[3], pts_burst[2]]
                    #poly_burst = geometry.Polygon(pts_burst)
                    poly_bursts.append(poly_burst)
        for itile_sample in l1b_ds['tile_sample'].values:
            for itile_line in l1b_ds['tile_line'].values:
                # Find lon/lat tile corners
                lon_corners = l1b_ds['corner_longitude'].sel(burst=iburst, tile_sample=itile_sample,
                                                             tile_line=itile_line).values.flatten().tolist()
                lat_corners = l1b_ds['corner_latitude'].sel(burst=iburst, tile_sample=itile_sample,
                                                            tile_line=itile_line).values.flatten().tolist()
                # Check on the lon/lat corners validity
                # print(np.sum((~np.isfinite(lon_corners))))
                if np.sum((~np.isfinite(lon_corners))) == 0:
                    # Define the tile polygons
                    pts = [geometry.Point(lon_corners[cpt], lat_corners[cpt]) for cpt, _ in enumerate(lon_corners)]
                    pts = [pts[0], pts[1], pts[3], pts[2],pts[0]]
                    logging.debug('pts: %s',pts)
                    logging.debug('one pt : %s %s',pts[0],type(pts[0]))
                    order = [0,1,3,2,0]
                    poly_tile = geometry.Polygon(np.stack([np.array(lon_corners)[order],np.array(lat_corners)[order]]).T)
                    #poly_tile = geometry.Polygon(pts)
                    poly_tiles.append(poly_tile)
                    # Coordinates
                    ibursts.append(iburst)
                    itile_samples.append(itile_sample)
                    itile_lines.append(itile_line)
                    if variable_names:
                        if (variable_names[0] is not None):
                            for variable_name in variable_names:
                                if (variable_name == 'macs_Im'):
                                    variables[variable_name].append(float(
                                        l1b_ds[variable_name].sel(burst=iburst, tile_sample=itile_sample,
                                                                  tile_line=itile_line).values[ik]))
                                else:
                                    variables[variable_name].append(float(
                                        l1b_ds[variable_name].sel(burst=iburst, tile_sample=itile_sample,
                                                                  tile_line=itile_line).values))

    polygons['tiles'] = poly_tiles
    polygons['bursts'] = poly_bursts
    #
    coordinates['ibursts'] = ibursts
    coordinates['itile_samples'] = itile_samples
    coordinates['itile_lines'] = itile_lines

    return polygons, coordinates, variables


def get_swath_tiles_polygons_from_l1bfile(l1b_file, ik=0, **kwargs):
    """
     get polygons for all the groups in a L1B SAR IFREMER file
     Args:
         l1b_file: str full path of L1B product .nc
         ik: int [optional], index of wave number selected for variable displayed such as 'cwave' or 'imacs'
     Keyword Args:
         kwargs (dict): optional keyword arguments : variable_names (list), is valid entry
     Returns:

     """
    swath_only = False
    # Initialisation of output structures
    _variables = None
    polygons = {}
    coordinates = {}
    variables = {}
    burst_types = ['intra', 'inter']

    coordinates_varnames = ['ibursts', 'itile_samples', 'itile_lines']
    variable_names = kwargs.get('variable_names', None)
    for burst_type in burst_types:
        # polygons
        polygons[burst_type] = {}
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = []
        # coordinates
        coordinates[burst_type] = {}
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = []
        # variables
        if variable_names:
            if variable_names[0] is not None:
                variables[burst_type] = {}
                for variable_name in variable_names:
                    variables[burst_type][variable_name] = []

    burst_types = ['intra', 'inter']
    for burst_type in burst_types:
        l1b_ds = xr.open_dataset(l1b_file, group=burst_type + 'burst')
        if variable_names:
            if variable_names[0] is not None:
                _polygons, _coordinates, _variables = get_swath_tiles_polygons_from_l1bgroup(l1b_ds,
                                                                                             variable_names=variable_names,
                                                                                             ik=ik)
        else:
            _polygons, _coordinates, _variables  = get_swath_tiles_polygons_from_l1bgroup(l1b_ds)

        # polygons
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = polygons[burst_type][polygons_varname] + _polygons[
                polygons_varname]
        # coordinates
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = coordinates[burst_type][coordinates_varname] + _coordinates[
                coordinates_varname]
        # variables
        if variable_names:
            if variable_names[0] is not None:
                for variable_name in variable_names:
                    variables[burst_type][variable_name] = variables[burst_type][variable_name] + _variables[variable_name]


    return polygons, coordinates, variables


def get_swath_tiles_polygons_from_l1bfiles(l1b_files, ik=0, **kwargs):
    """
         get polygons for all the groups in a set of L1B SAR IFREMER files
         Args:
             l1b_file: str full path of L1B product .nc
             ik: int [optional], index of wave number selected for variable displayed such as 'cwave' or 'imacs'
         Keyword Args:
             kwargs (dict): optional keyword arguments : variable_names (list), is valid entry
         Returns:

         """
    # Initialisation of output structures
    polygons = {}
    coordinates = {}
    variables = {}
    _polygons = {}
    _coordinates = None
    _variables = None
    burst_types = ['intra', 'inter']
    coordinates_varnames = ['ibursts', 'itile_samples', 'itile_lines']
    variable_names = kwargs.get('variable_names', None)
    for burst_type in burst_types:
        # polygons
        polygons[burst_type] = {}
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = []
        # coordinates
        coordinates[burst_type] = {}
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = []
        # variables
        if variable_names:
            if variable_names[0] is not None:
                variables[burst_type] = {}
                for variable_name in variable_names:
                    variables[burst_type][variable_name] = []

    for l1b_file in l1b_files:
        # Read the file
        if variable_names:
            if variable_names[0] is not None:
                _polygons, _coordinates, _variables = get_swath_tiles_polygons_from_l1bfile(l1b_file,
                                                                                            variable_names=variable_names,
                                                                                            ik=ik)
        else:
            _polygons, _coordinates, _variables = get_swath_tiles_polygons_from_l1bfile(l1b_file)

        # Fill the output for each burst_type
        for burst_type in burst_types:
            # polygons
            for polygons_varname in polygons_varnames:
                polygons[burst_type][polygons_varname] = polygons[burst_type][polygons_varname] + _polygons[burst_type][
                    polygons_varname]
            # coordinates
            for coordinates_varname in coordinates_varnames:
                coordinates[burst_type][coordinates_varname] = coordinates[burst_type][coordinates_varname] + \
                                                               _coordinates[burst_type][coordinates_varname]
            # variables
            if variable_names:
                if variable_names[0] is not None:
                    for variable_name in variable_names:
                        variables[burst_type][variable_name] = variables[burst_type][variable_name] + \
                                                               _variables[burst_type][variable_name]


    return polygons, coordinates, variables
