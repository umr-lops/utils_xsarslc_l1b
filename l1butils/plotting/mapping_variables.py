def get_varname_for_mapping_from_sarfile(safe_file, varname, burst_type, acq_mode='iw', plot=True,pol='vv',bckgrd=True,level='*_L1B_*',clim=(200,350), cmap='jet', tau='2tau'):

    from glob import glob
    import datatree
    import numpy as np
    
    import holoviews as hv
    hv.extension('bokeh')
    import geoviews as gv
    
    #print(safe_file)
    
    if (burst_type=='intraburst'):
        if (varname == 'imacs'):
            varname = varname + '_' + tau
            
    #print(varname)
    
    gvvbckg = gv.tile_sources.EsriImagery
    if (acq_mode=='iw'):
        #print(safe_file + 's1*-iw*-slc-'+pol+'*'+level+'*.nc')
        files = glob(safe_file + 's1*-iw*-slc-'+pol+'*'+level+'*.nc')
        #print(files)
        if (len(files)==0):
            print('No Files to read')
            return 0
        for cpt,_file in enumerate(files):
            #print(_file)
            dt = datatree.open_datatree(_file)

            dsra = dt[burst_type].ds
            lon = np.squeeze(dsra['longitude'].values)
            lat = np.squeeze(dsra['latitude'].values)
            #print(dsra.keys())
            var = np.squeeze(dsra[varname].values)
            if (cpt==0):
                gvv  = gv.QuadMesh((lon, lat, var), kdims=['longitude','latitude'], vdims=[varname]).opts(color='level', cmap=cmap, colorbar=True, clim=clim, tools=['hover'])
            else:
                gvv  = gvv * gv.QuadMesh((lon, lat, var), kdims=['longitude','latitude'], vdims=[varname]).opts(color='level', cmap=cmap, colorbar=True, clim=clim, tools=['hover'])
        
        if bckgrd:   
            return gvvbckg*gvv
        else:
            return gvv
        
    else:
        return None

def get_varname_for_mapping_from_sarfiles(safe_files, varname, burst_type, acq_mode='iw', plot=True, pol='vv',level='*_L1B_*',clim=(200,350), cmap='jet', tau='2tau'):
    

    from glob import glob
    import datatree
    import numpy as np
    
    import holoviews as hv
    hv.extension('bokeh')
    import geoviews as gv
    
    
    gvvbckg = gv.tile_sources.EsriImagery
    for cpt,safe_file in enumerate(safe_files):
            
            print(safe_file)
            if (cpt==0):
                gvv = get_varname_for_mapping_from_sarfile(safe_file, varname, burst_type, acq_mode=acq_mode, plot=plot,pol=pol,bckgrd=False,level=level,clim=clim,cmap=cmap,tau=tau)
            else:
                gvv = gvv* get_varname_for_mapping_from_sarfile(safe_file, varname, burst_type, acq_mode=acq_mode, plot=plot,pol=pol,bckgrd=False,level=level,clim=clim,cmap=cmap,tau=tau)

    return gvvbckg*gvv
