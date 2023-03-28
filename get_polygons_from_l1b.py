# +
import datatree
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from shapely import geometry
from shapely import wkt
import time
import xarray as xr

# from xsarslc.tools import xndindex

# +
def get_swath_tiles_polygons_from_l1bgroup(l1b_ds, swath_only=False, variable_names = [None],ik=0):

    polygons = {} ; coordinates = {}; variables = {}
    poly_tiles = []
    ibursts = []
    itile_samples = []
    itile_lines = []
    if (variable_names[0] is not None):
        for variable_name in variable_names:
            variables[variable_name] = []
    
    # Swath_polynom
    poly_swath = wkt.loads(l1b_ds.attrs['footprint'])
    polygons['swath'] = [poly_swath]
    if (swath_only):
        return polygons
    
    # Tiles polynom & variables
    for iburst in l1b_ds['burst'].values:
        for itile_sample in l1b_ds['tile_sample'].values:
            for itile_line in l1b_ds['tile_line'].values:
                # Find lon/lat tile corners
                lon_corners = l1b_ds['corner_longitude'].sel(burst=iburst,tile_sample=itile_sample,tile_line=itile_line).values.flatten().tolist()
                lat_corners = l1b_ds['corner_latitude'].sel(burst=iburst,tile_sample=itile_sample,tile_line=itile_line).values.flatten().tolist()
                # Define the tile polygons
                pts = [geometry.Point(lon_corners[cpt],lat_corners[cpt]) for cpt,_ in enumerate(lon_corners)]
                pts = [pts[0], pts[1], pts[3], pts[2]]
                poly_tile = geometry.Polygon(pts)
                poly_tiles.append(poly_tile)
                # Coordinates
                ibursts.append(iburst)
                itile_samples.append(itile_sample)
                itile_lines.append(itile_line)
                if (variable_names[0] is not None):
                    for variable_name in variable_names:
                        if (variable_name=='macs_Im'):
                            variables[variable_name].append(float(l1b_ds[variable_name].sel(burst=iburst,tile_sample=itile_sample,tile_line=itile_line).values[ik]))
                        else:
                            variables[variable_name].append(float(l1b_ds[variable_name].sel(burst=iburst,tile_sample=itile_sample,tile_line=itile_line).values))
                        
    polygons['tiles'] = poly_tiles
    #
    coordinates['ibursts'] = ibursts 
    coordinates['itile_samples'] = itile_samples 
    coordinates['itile_lines'] = itile_lines
    if (variable_names[0] is not None):
        return polygons, coordinates, variables
    else:
        return polygons,coordinates


def get_swath_tiles_polygons_from_l1bfile(l1b_file, variable_names = [None],ik=0):
    
    swath_only=False
    
    # Initialisation of output structures
    polygons = {}; coordinates = {};  variables = {}
    burst_types = ['intra','inter']
    polygons_varnames = ['swath','tiles']
    coordinates_varnames = ['ibursts', 'itile_samples','itile_lines'] 
    for burst_type in burst_types :
        # polygons
        polygons[burst_type]={}
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = []
        # coordinates
        coordinates[burst_type]={}
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = []
        # variables
        if (variable_names[0] is not None):
            variables[burst_type]={}
            for variable_name in variable_names:
                variables[burst_type][variable_name] = []

    burst_types = ['intra','inter']
    for burst_type in burst_types:
        l1b_ds = xr.open_dataset(l1b_file,group=burst_type+'burst')
        if (variable_names[0] is not None):
            _polygons, _coordinates, _variables = get_swath_tiles_polygons_from_l1bgroup(l1b_ds, variable_names = variable_names, ik=ik)
        else:
            _polygons, _coordinates = get_swath_tiles_polygons_from_l1bgroup(l1b_ds)

        # polygons
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = polygons[burst_type][polygons_varname] + _polygons[polygons_varname]
        # coordinates
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = coordinates[burst_type][coordinates_varname] + _coordinates[coordinates_varname]
        # variables
        if (variable_names[0] is not None):
            for variable_name in variable_names:
                variables[burst_type][variable_name] = variables[burst_type][variable_name] + _variables[variable_name]
        
    if (variable_names[0] is not None):
        return polygons,coordinates,variables
    else:
        return polygons,coordinates


def get_swath_tiles_polygons_from_l1bfiles(l1b_files, variable_names = [None], ik=0):

    # Initialisation of output structures
    polygons = {}; coordinates = {};  variables = {}
    burst_types = ['intra','inter']
    polygons_varnames = ['swath','tiles']
    coordinates_varnames = ['ibursts', 'itile_samples','itile_lines'] 
    for burst_type in burst_types :
        # polygons
        polygons[burst_type]={}
        for polygons_varname in polygons_varnames:
            polygons[burst_type][polygons_varname] = []
        # coordinates
        coordinates[burst_type]={}
        for coordinates_varname in coordinates_varnames:
            coordinates[burst_type][coordinates_varname] = []
        # variables
        if (variable_names[0] is not None):
            variables[burst_type]={}
            for variable_name in variable_names:
                variables[burst_type][variable_name] = []
        
    # Fill the ouput 
    for l1b_file in l1b_files:
        if (variable_names[0] is not None):
            _polygons, _coordinates, _variables = get_swath_tiles_polygons_from_l1bfile(l1b_file, variable_names = variable_names, ik=ik)
        else:
            _polygons, _coordinates = get_swath_tiles_polygons_from_l1bfile(l1b_file)
        for burst_type in burst_types :
            # polygons
            for polygons_varname in polygons_varnames:
                polygons[burst_type][polygons_varname] = polygons[burst_type][polygons_varname] + _polygons[burst_type][polygons_varname]
            # coordinates
            for coordinates_varname in coordinates_varnames:
                coordinates[burst_type][coordinates_varname] = coordinates[burst_type][coordinates_varname] + _coordinates[burst_type][coordinates_varname]
            # variables
            if (variable_names[0] is not None):
                for variable_name in variable_names:
                    variables[burst_type][variable_name] = variables[burst_type][variable_name] + _variables[burst_type][variable_name]

    if (variable_names[0] is not None):
        return polygons,coordinates,variables
    else:
        return polygons,coordinates


# def get_polygons_from_l1b(files, varnames = None):

#     pts_tiles_intra = []; pts_tiles_inter = []; pts_swath = []
#     t0 = time.time()

#     pts_tiles = {}
#     burst_type = ['intraburst','interburst']
#     variables = {}
#     for brst in burst_type:
#         variables[brst] = {}
#         pts_tiles[brst] = []
#         if (varnames is not None):
#             for varname in varnames:
#                 variables[brst][varname] = []

#     for cpt,_file in enumerate(files):

#         dt = datatree.open_datatree(_file)

#         polyswath = wkt.loads(dt['intraburst'].ds.attrs['footprint'])
#         lon,lat = polyswath.exterior.xy
#         _pts_swath = [(x,y) for x,y in zip(lon,lat)]
#         pts_swath.append(_pts_swath)
        
        
#         for brst in burst_type:

#             lon_corners = dt[brst]['corner_longitude'].squeeze()
#             lat_corners = dt[brst]['corner_latitude'].squeeze()
#             Nt = dt[brst].ds.sizes['tile_sample']
#             bursts = dt[brst]['burst'].values        
#             for ib in bursts:
#                 for it in np.arange(Nt):

#                     # Get corner list
#                     _lon1 = lon_corners.sel(burst=ib).isel(tile_sample=it,c_line=0)
#                     _lon2 = lon_corners.sel(burst=ib).isel(tile_sample=it,c_line=1)
#                     _lat1 = lat_corners.sel(burst=ib).isel(tile_sample=it,c_line=0)
#                     _lat2 = lat_corners.sel(burst=ib).isel(tile_sample=it,c_line=1)
#                     lon = list(_lon1.values) + list(_lon2.values[::-1])
#                     lat = list(_lat1.values) + list(_lat2.values[::-1])
#                     _pts_tiles = [ (x,y) for x,y in zip(lon,lat)]
#                     pts_tiles[brst].append(_pts_tiles)

#                     # Get variables
#                     if (varnames is not None):
#                         for varname in varnames:
#                                 variables[brst][varname].append(dt[brst][varname].sel(burst=ib).isel(tile_sample=it).values[0])
                        
                    
                                                            
                    
#     t1 = time.time()
#     print(t1-t0)
#     if (varnames is not None):
#         return pts_swath, pts_tiles, variables
#     else:
#         return pts_swath, pts_tiles
# -

