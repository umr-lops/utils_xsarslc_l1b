import holoviews as hv
hv.extension('bokeh')
import geoviews as gv

import datatree
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from shapely import geometry
from shapely import wkt


def get_polys_with_varname(files, varname = 'sigma0',burst_type=['intraburst'],cmap='Greys_r',clim=(0,0.15),alpha=0.5):

    gvvbckg = gv.tile_sources.EsriImagery
    polys = gvvbckg
    #return polys
    for cpt,_file in enumerate(files):

        #print(cpt,os.path.basename(_file))

        dt = datatree.open_datatree(_file)

        polyswath = wkt.loads(dt['intraburst'].ds.attrs['footprint'])
        lon,lat = polyswath.exterior.xy
        pts = [(x,y) for x,y in zip(lon,lat)]
        polyswath = gv.Polygons({('Longitude','Latitude'): pts},kdims=['Longitude','Latitude']).opts(color='white',fill_color='white',alpha=0.125)
        polys = polys*polyswath

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
                    var = dt[brst][varname].sel(burst=ib).isel(tile_sample=it).values[0]
                    if (varname == 'imacs_Im'):
                        var = dt[brst][varname].sel(burst=ib,lambda_range_max_macs=250).isel(tile_sample=it).values[0]
                    sig = dt[brst]['sigma0'].sel(burst=ib).isel(tile_sample=it).values[0]
                    #print(var)
                    pts = [ (x,y) for x,y in zip(lon,lat)]
                    if np.isfinite(sig):
                        poly = gv.Polygons({('Longitude','Latitude'): pts,'level':var},
                                        vdims='level',kdims=['Longitude','Latitude']).opts(color='NA',
                                                                        colorbar=True,cmap=cmap,clim=clim,alpha=alpha)
                        #return poly
                        polys = polys*poly                
                    

    return polys


def get_polys_alone(files, burst_type=['intraburst','interburst']):
    
    
    gvvbckg = gv.tile_sources.EsriImagery
    polys = gvvbckg
    for cpt,_file in enumerate(files):
        
        #print(cpt,os.path.basename(_file))

        dt = datatree.open_datatree(_file)

        polyswath = wkt.loads(dt['intraburst'].ds.attrs['footprint'])
        lon,lat = polyswath.exterior.xy
        pts = [(x,y) for x,y in zip(lon,lat)]
        polyswath = gv.Polygons({('Longitude','Latitude'): pts},kdims=['Longitude','Latitude']).opts(color='white',fill_color='white',alpha=0.125)
        polys = polys*polyswath
        
        #burst_type = ['intraburst','interburst']

        for brst in burst_type:

            if (brst=='interburst'):
                colorline='red';fillcolor='red'
                alpha=0.5
            else:
                colorline='blue';fillcolor='blue'
                alpha=0.25

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
                    sig = dt[brst]['sigma0'].sel(burst=ib).isel(tile_sample=it)
                    pts = [ (x,y) for x,y in zip(lon,lat)]
                    if ~np.isfinite(sig):
                        poly = gv.Polygons({('Longitude','Latitude'): pts},kdims=['Longitude','Latitude']).opts(color='gray',fill_color='gray',alpha=alpha)#, colorbar=True,cmap='Greys_r')
                    else:
                        poly = gv.Polygons({('Longitude','Latitude'): pts},kdims=['Longitude','Latitude']).opts(color=colorline,fill_color=fillcolor,alpha=alpha)
                    polys = polys*poly                
           
    
    return polys