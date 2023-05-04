#!/bin/env python
import warnings
import os
import slcl1butils
import logging
import zipfile
import fsspec
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