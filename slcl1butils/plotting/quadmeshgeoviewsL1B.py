import pdb

import xarray as xr
import numpy as np
import holoviews as hv
import geoviews as gv
import logging
import os
import datatree
gv.extension('bokeh')
def doQuadMeshL1BorL1C_onefile(ff,bursttype='intraburst',variable='sigma0',clim=(0,0.2),cmap='Greys_r'):
    if isinstance(ff,str):
        ff = prepare_L1BC(ff,bursttype,variable)
    else: # suppose that it is already parsed L1B through prepare_L1BC()
        pass
    qmeshes = gv.Dataset(ff, kdims=['longitude', 'latitude']).to(gv.QuadMesh)
    # clipping = {"min": "red", "max": "green", "NaN": "gray"}
    clipping = {"NaN": "blue"} # see https://holoviews.org/user_guide/Styling_Plots.html
    fig = qmeshes.opts(width=1200, height=900, colorbar=True, cmap=cmap,clipping_colors=clipping,tools=['hover']) #clim=clim
    return fig

def open_of_use(ff,bursttype):
    """

    Args:
        ff: can be xr.Dataset or full path

    Returns:

    """
    if isinstance(ff,str):
        ds = xr.open_dataset(ff, group=bursttype,engine='h5netcdf')
    else:
        ds = ff
    return ds
def prepare_L1BC(ff,bursttype,variable,**kwargs):
    """

    Args:
        ff: str or xr.Dataset
        bursttype: str
        variable: str
    Kwargs:
        kwargs (dict): optional keyword arguments : k_gp (int) phi_hf (int) lambda_range_max_macs(int) are valid entries
    Returns:

    """
    ds = open_of_use(ff, bursttype)
    # ds.load()
    # ds = datatree.open_datatree(ff)[bursttype].to_dataset() # it doesnt change anything to segmentation fault...
    if variable in ['cwave_params']:
        sub = ds[variable].isel({'2tau':0,'k_gp':kwargs.get('k_gp'),'phi_hf':kwargs.get('phi_hf')}).stack({'y': ['burst', 'tile_line']})
    elif variable in ['macs_Im','macs_Re']:
        sub = ds[variable].isel({'lambda_range_max_macs':kwargs.get('lambda_range_max_macs')}).stack({'y': ['burst', 'tile_line']})
    else:
        sub = ds[variable].stack({'y': ['burst', 'tile_line']})
    if np.isnan(sub).any():
        logging.debug('there are %s NaN in the variable stacked : %s',np.isnan(sub.data).sum(),variable)
    sub = sub.rename({'tile_sample': 'x'})
    sub = sub.assign_coords({'y': np.arange(sub.y.size)})
    sub = sub.assign_coords({'x': np.arange(sub.x.size)})
    if (sub==0).any() and variable=='sigma0':
        pdb.set_trace()
    else:
        pass
        # print('debug not zeros found -> continue')
    if np.isnan(sub).any() and variable=='sigma0' and (ds['land_flag'].data==False).all():
        print('Nan alerte')
        pdb.set_trace()
    return sub

def doQuadMeshL1BorL1C_manyfiles(files,bursttype='intraburst',variable='sigma0',clim=(0,0.2),title='',cmap='Greys_r'):
    all_quads = []
    for ff in files:
        oneQuad = doQuadMeshL1BorL1C_onefile(ff, bursttype=bursttype, variable=variable,clim=clim,cmap=cmap)
        all_quads.append(oneQuad)
    tiles = gv.tile_sources.EsriImagery
    fig = tiles * hv.Overlay(all_quads)
    fig.opts(title=title,tools=['hover'])
    return fig

def doQuadMeshL1BorL1C_manyfiles_opt(files,pol,variables,bursttype='intraburst',clim=(0,0.2),title='',cmap='Greys_r',
                                     outputdir=None,outbasename=None,**kwargs):
    """

    Args:
        files: list of fulll path str
        pol : str vv vh hv hh
        variables: list of string
        bursttype: str intraburts or interburst
        clim: tuple eg (0.2,4.5)
        title: str
        cmap:str eg jet
        outputdir: str
    Kwargs:
        kwargs (dict): optional keyword arguments : k_gp (int) phi_hf (int)  are valid entries

    Returns:

    """
    fig = None
    allds = {}
    for ff in files:
        allds[ff] = open_of_use(ff,bursttype)
    logging.info('all the files are open once!')
    subs = {}
    subdsx = []
    for vvi,var in enumerate(variables):
        if isinstance(cmap,list) and len(cmap)==len(variables):

            thecmap = cmap[vvi]
        else:
            thecmap=cmap
        logging.info('variable = %s',var)
        for ff in files:
            ds = allds[ff]
            sub = prepare_L1BC(ds, bursttype, var,**kwargs)
            subs['%s_%s'%(var,ff)] = sub
            subdsx.append(sub)
        fig = doQuadMeshL1BorL1C_manyfiles(
            subdsx,
            bursttype=bursttype,
            variable=var,
            clim=clim,
            title=title,
            cmap=thecmap,
        )
        if outputdir:
            renderer = hv.renderer("bokeh")
            #
            # Using renderer save
            if outbasename is None:
                outf = os.path.join(outputdir, "map_%s_%s" % (pol,var))
            else:
                outf = os.path.join(outputdir,outbasename)
            renderer.save(fig, outf)
            # save(fig, outf)
            logging.info('output file: %s',outf+'.html')
    return fig

