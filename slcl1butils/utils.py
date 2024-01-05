#!/bin/env python
import pdb
import warnings
import os
import slcl1butils
import numpy as np
import logging
import zipfile
import fsspec
import xarray as xr
import aiohttp
from slcl1butils.get_config import get_conf
config = get_conf()
logger = logging.getLogger('xsar.utils')
logger.addHandler(logging.NullHandler())

mem_monitor = True

try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


def netcdf_compliant(dataset):
    """
    Create a dataset that can be written on disk with xr.Dataset.to_netcdf() function. It split complex variable in real and imaginary variable

    Args:
        dataset (xarray.Dataset): dataset to be transformed
    """
    var_to_rm = list()
    var_to_add = list()
    for i in dataset.variables.keys():
        if dataset[i].dtype == complex or dataset[i].dtype=='complex64':
            re = dataset[i].real
            # re.encoding['_FillValue'] = 9.9692099683868690e+36
            im = dataset[i].imag
            # im.encoding['_FillValue'] = 9.9692099683868690e+36
            var_to_add.append({str(i) + '_Re': re, str(i) + '_Im': im})
            var_to_rm.append(str(i))
    ds_to_save = xr.merge([dataset.drop_vars(var_to_rm), *var_to_add], compat='override')
    for vv in ds_to_save.variables.keys():
        if ds_to_save[vv].dtype == 'int64':  # to avoid ncview: netcdf_dim_value: unknown data type (10) for corner_line ...
            ds_to_save[vv] = ds_to_save[vv].astype(np.int16)
        elif ds_to_save[vv].dtype == 'float64':
            ds_to_save[vv] = ds_to_save[vv].astype(np.float32) # to reduce volume of output files
        else:
            logging.debug('%s is dtype %s',vv,ds_to_save[vv].dtype)
    return ds_to_save
def url_get(url, cache_dir=os.path.join(config['data_dir'], 'fsspec_cache')):
    """
    Get fil from url, using caching.
    Parameters
    ----------
    url: str
    cache_dir: str
        Cache dir to use. default to `os.path.join(config['data_dir'], 'fsspec_cache')`
    Raises
    ------
    FileNotFoundError
    Returns
    -------
    filename: str
        The local file name
    Notes
    -----
    Due to fsspec, the returned filename won't match the remote one.
    """

    if '://' in url:
        with fsspec.open(
                'filecache::%s' % url,
                https={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
                #filecache={'cache_storage': os.path.join(os.path.join(config['data_dir'], 'fsspec_cache'))}
        ) as f:
            fname = f.name
    else:
        fname = url

    return fname


def get_test_file(fname):
    """
    get test file from  https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/
    file is unzipped and extracted to `config['data_dir']`

    Parameters
    ----------
    fname: str
        file name to get (without '.zip' extension)

    Returns
    -------
    str
        path to file, relative to `config['data_dir']`

    """
    #res_path = config['data_dir']
    res_path = os.path.join(os.path.dirname(os.path.dirname(slcl1butils.__file__)),'assests')
    #base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    base_url = 'https://cerweb.ifremer.fr/datarmor/sarwave/documentation/processor/sar/l1butils/example_products/iw/slc/l1b/'
    file_url = '%s/%s.zip' % (base_url, fname)
    final = os.path.join(res_path, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        local_file = url_get(file_url)
        warnings.warn("Unzipping %s" % final)

        #shutil.move(local_file,final)
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
             zip_ref.extractall(res_path)
    return final

def get_memory_usage():
    """

    Returns
    -------

    """
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000. / 1000.
    except:  # on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go' % memory_used_go
    return str_mem


def xndindex(sizes):
    """
    xarray equivalent of np.ndindex iterator with defined dimension names

    Args:
        sizes (dict): dict of form {dimension_name (str): size(int)}
    Return:
        iterator over dict
    """
    from itertools import repeat

    for d, k in zip(repeat(tuple(sizes.keys())), zip(np.ndindex(tuple(sizes.values())))):
        yield {k: l for k, l in zip(d, k[0])}