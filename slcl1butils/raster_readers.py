# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: xsarslc
#     language: python
#     name: xsarslc
# ---

# +

import xarray as xr
import datetime
import numpy as np
import glob
#from utils import bind, url_get
from slcl1butils.utils import url_get
import pandas as pd



def resource_strftime(resource, **kwargs):
    """
    From a resource string like '%Y/%j/myfile_%Y%m%d%H%M.nc' and a date like 'Timestamp('2018-10-13 06:23:22.317102')',
    returns a tuple composed of the closer available date and string like '/2018/286/myfile_201810130600.nc'
    If ressource string is an url (ie 'ftp://ecmwf/%Y/%j/myfile_%Y%m%d%H%M.nc'), fsspec will be used to retreive the file locally.
   
    Parameters
    ----------
    resource: str
        resource string, with strftime template
    date: datetime
        date to be used
    step: int
        hour step between 2 files
    Returns
    -------
    tuple : (datetime,str)
    """

    date = kwargs['date']
    step = kwargs['step']

    delta = datetime.timedelta(hours=step) / 2
    date = date.replace(
        year=(date+delta).year,
        month=(date+delta).month,
        day=(date+delta).day,
        hour=(date+delta).hour // step * step,
        minute=0,
        second=0,
        microsecond=0
    )
    return date, url_get(date.strftime(resource))



def _to_lon180(ds):
    # roll [0, 360] to [-180, 180] on dim x
    ds = ds.roll(x=-np.searchsorted(ds.x, 180), roll_coords=True)
    ds['x'] = xr.where(ds['x'] >= 180, ds['x'] - 360, ds['x'])
    return ds


def ecmwf_0100_1h(fname,use_dask=True):
    """
    ecmwf 0.1 deg 1h reader (ECMWF_FORECAST_0100_202109091300_10U_10V.nc)
    
    Parameters
    ----------
    fname: str
        
        hwrf filename
    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = xr.open_dataset(fname, chunks={'Longitude': 1000, 'Latitude': 1000}).isel(time=0)
    ecmwf_ds.attrs['time'] = datetime.datetime.fromtimestamp(ecmwf_ds.time.item() // 1000000000)
    ecmwf_ds = ecmwf_ds.drop_vars('time').rename(
        {
            'Longitude': 'x',
            'Latitude': 'y',
            '10U': 'U10',
            '10V': 'V10'
        }
    )
    ecmwf_ds.attrs = {k: ecmwf_ds.attrs[k] for k in ['title', 'institution', 'time']}

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)
    
    if (use_dask==False):
        for var in ecmwf_ds:
            ecmwf_ds[var] = ecmwf_ds[var].compute() 

    return ecmwf_ds

def ecmwf_0125_1h(fname):
    """
    ecmwf 0.125 deg 1h reader (ecmwf_201709071100.nc)
    
    Parameters
    ----------
    fname: str
        
        hwrf filename
    Returns
    -------
    xarray.Dataset
    """
    ecmwf_ds = xr.open_dataset(fname, chunks={'longitude': 1000, 'latitude': 1000})

    ecmwf_ds = ecmwf_ds.rename(
        {'longitude': 'x', 'latitude': 'y'}
    ).rename(
        {'Longitude': 'x', 'Latitude': 'y', 'U': 'U10', 'V': 'V10'}
    ).set_coords(['x', 'y'])

    ecmwf_ds['x'] = ecmwf_ds.x.compute()
    ecmwf_ds['y'] = ecmwf_ds.y.compute()

    # dataset is lon [0, 360], make it [-180,180]
    ecmwf_ds = _to_lon180(ecmwf_ds)

    ecmwf_ds.attrs['time'] = datetime.datetime.fromisoformat(ecmwf_ds.attrs['date'])

    ecmwf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return ecmwf_ds


def ww3_global_yearly_3h(filename, date):
    
    str_time = datetime.datetime.strftime(date, '%Y-%m-%d-T%H:%M:%S.000000000')
    
    ds = xr.open_dataset(filename)
    dst = ds.sel(time = str_time).drop_vars('time').rename(
    {
        'longitude': 'x',
        'latitude': 'y'
    })
    
    dst.rio.write_crs("EPSG:4326", inplace=True)
    
    return dst


def hwrf_0015_3h(fname,**kwargs):
    """
    hwrf 0.015 deg 3h reader ()
    
    
    Parameters
    ----------
    fname: str
        
        hwrf filename
    Returns
    -------
    xarray.Dataset
    """
    hwrf_ds = xr.open_dataset(fname)
    try : 
        hwrf_ds = hwrf_ds.sel(dim_0=kwargs['date'])[['u','v','elon','nlat']]
    except Exception as e: 
        raise ValueError("date '%s' can't be find in %s " % (kwargs['date'], fname))
    
    time_datetime = datetime.datetime.utcfromtimestamp(hwrf_ds.dim_0.values.astype(int) * 1e-9)
    hwrf_ds.attrs['time'] = (time_datetime.strftime("%Y/%m/%d %H:%M:%S"))

    hwrf_ds = hwrf_ds.assign_coords({"x":hwrf_ds.elon.values[0,:],"y":hwrf_ds.nlat.values[:,0]}).drop_vars(['dim_0','elon','nlat']).rename(
            {
                'u': 'U10',
                'v': 'V10'
            }
        )

    hwrf_ds.attrs = {k: hwrf_ds.attrs[k] for k in ['institution', 'time']}
    hwrf_ds = _to_lon180(hwrf_ds)
    hwrf_ds.rio.write_crs("EPSG:4326", inplace=True)

    return hwrf_ds

def gebco(gebco_files):
    """gebco file reader (geotiff from https://www.gebco.net/data_and_products/gridded_bathymetry_data)"""
    return xr.combine_by_coords(
        [
            xr.open_dataset(
                f, chunks={'x': 1000, 'y': 1000}
            ).isel(band=0).drop_vars('band') for f in gebco_files
        ]
    )

# list available rasters as a pandas dataframe
#available_rasters = pd.DataFrame(columns=['resource', 'read_function', 'get_function'])
#available_rasters.loc['gebco'] = [None, gebco, glob.glob]
#available_rasters.loc['ecmwf_0100_1h'] = [None, ecmwf_0100_1h, bind(resource_strftime, ..., step=1)]
#available_rasters.loc['ecmwf_0125_1h'] = [None, ecmwf_0125_1h, bind(resource_strftime, ..., step=1)]
#available_rasters.loc['hwrf_0015_3h'] = [None, hwrf_0015_3h, bind(resource_strftime, ..., step=3)]
