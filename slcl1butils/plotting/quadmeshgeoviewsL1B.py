import xarray as xr
import numpy as np
import holoviews as hv
import geoviews as gv
gv.extension('bokeh')
def doQuadMeshL1BorL1C_onefile(ff,bursttype='intraburst',variable='sigma0'):
    ds = xr.open_dataset(ff, group=bursttype)
    sub = ds[variable].stack({'y': ['burst', 'tile_line']})

    sub = sub.rename({'tile_sample': 'x'})
    sub = sub.assign_coords({'y': np.arange(sub.y.size)})
    sub = sub.assign_coords({'x': np.arange(sub.x.size)})

    qmeshes = gv.Dataset(sub, kdims=['longitude', 'latitude']).to(gv.QuadMesh)
    fig = qmeshes.opts(width=900, height=600, colorbar=True, cmap='Greys_r')
    return fig

def doQuadMeshL1BorL1C_manyfiles(files,bursttype='intraburst',variable='sigma0'):
    all_quads = []
    for ff in files:
        oneQuad = doQuadMeshL1BorL1C_onefile(ff, bursttype=bursttype, variable=variable)
        all_quads.append(oneQuad)
    tiles = gv.tile_sources.EsriImagery
    fig = tiles * hv.Overlay(all_quads)
    return fig

