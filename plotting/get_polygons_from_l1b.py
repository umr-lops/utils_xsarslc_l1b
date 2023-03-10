# +
import datatree
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from shapely import geometry
from shapely import wkt
import time

# from xsarslc.tools import xndindex
# -

def get_polygons_from_l1b(files, varnames = None):

    pts_tiles_intra = []; pts_tiles_inter = []; pts_swath = []
    t0 = time.time()

    pts_tiles = {}
    burst_type = ['intraburst','interburst']
    variables = {}
    for brst in burst_type:
        variables[brst] = {}
        pts_tiles[brst] = []
        if (varnames is not None):
            for varname in varnames:
                variables[brst][varname] = []

    for cpt,_file in enumerate(files):

        dt = datatree.open_datatree(_file)

        polyswath = wkt.loads(dt['intraburst'].ds.attrs['footprint'])
        lon,lat = polyswath.exterior.xy
        _pts_swath = [(x,y) for x,y in zip(lon,lat)]
        pts_swath.append(_pts_swath)
        
        
        for brst in burst_type:

            lon_corners = dt[brst]['corner_longitude'].squeeze()
            lat_corners = dt[brst]['corner_latitude'].squeeze()
            Nt = dt[brst].ds.sizes['tile_sample']
            bursts = dt[brst]['burst'].values        
            for ib in bursts:
                for it in np.arange(Nt):

                    # Get corner list
                    _lon1 = lon_corners.sel(burst=ib).isel(tile_sample=it,c_line=0)
                    _lon2 = lon_corners.sel(burst=ib).isel(tile_sample=it,c_line=1)
                    _lat1 = lat_corners.sel(burst=ib).isel(tile_sample=it,c_line=0)
                    _lat2 = lat_corners.sel(burst=ib).isel(tile_sample=it,c_line=1)
                    lon = list(_lon1.values) + list(_lon2.values[::-1])
                    lat = list(_lat1.values) + list(_lat2.values[::-1])
                    _pts_tiles = [ (x,y) for x,y in zip(lon,lat)]
                    pts_tiles[brst].append(_pts_tiles)

                    # Get variables
                    if (varnames is not None):
                        for varname in varnames:
                                variables[brst][varname].append(dt[brst][varname].sel(burst=ib).isel(tile_sample=it).values[0])
                        
                    
                                                            
                    
    t1 = time.time()
    print(t1-t0)
    if (varnames is not None):
        return pts_swath, pts_tiles, variables
    else:
        return pts_swath, pts_tiles
