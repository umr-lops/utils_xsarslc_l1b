import copy
import datetime
import logging
import os
import warnings
import zipfile

import aiohttp
import fsspec
import numpy as np
import xarray as xr

import slcl1butils
from slcl1butils.get_config import get_conf

config = get_conf()
logger = logging.getLogger("xsar.utils")
logger.addHandler(logging.NullHandler())


def netcdf_compliant(dataset):
    """
    Create a dataset that can be written on disk with xr.Dataset.to_netcdf() function. It split complex variable in real and imaginary variable

    Args:
        dataset (xarray.Dataset): dataset to be transformed
    """
    var_to_rm = list()
    var_to_add = list()
    for i in dataset.variables.keys():
        if dataset[i].dtype == complex or dataset[i].dtype == "complex64":
            re = dataset[i].real
            # re.encoding['_FillValue'] = 9.9692099683868690e+36
            im = dataset[i].imag
            # im.encoding['_FillValue'] = 9.9692099683868690e+36
            var_to_add.append({str(i) + "_Re": re, str(i) + "_Im": im})
            var_to_rm.append(str(i))
    ds_to_save = xr.merge(
        [dataset.drop_vars(var_to_rm), *var_to_add], compat="override"
    )
    for vv in ds_to_save.variables.keys():
        if (
            ds_to_save[vv].dtype == "int64"
        ):  # to avoid ncview: netcdf_dim_value: unknown data type (10) for corner_line ...
            ds_to_save[vv] = ds_to_save[vv].astype(np.int16)
        elif ds_to_save[vv].dtype == "float64":
            ds_to_save[vv] = ds_to_save[vv].astype(
                np.float32
            )  # to reduce volume of output files
        else:
            logging.debug("%s is dtype %s", vv, ds_to_save[vv].dtype)
    return ds_to_save


def url_get(url, cache_dir=os.path.join(config["data_dir"], "fsspec_cache")):
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

    if "://" in url:
        with fsspec.open(
            "filecache::%s" % url,
            https={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
            # filecache={'cache_storage': os.path.join(os.path.join(config['data_dir'], 'fsspec_cache'))}
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
    # res_path = config['data_dir']
    res_path = os.path.join(
        os.path.dirname(os.path.dirname(slcl1butils.__file__)), "assests"
    )
    # base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    base_url = "https://cerweb.ifremer.fr/datarmor/sarwave/documentation/processor/sar/l1butils/example_products/iw/slc/l1b/"
    file_url = "%s/%s.zip" % (base_url, fname)
    final = os.path.join(res_path, fname)
    if not os.path.exists(os.path.join(res_path, fname)):
        warnings.warn("Downloading %s" % file_url)
        local_file = url_get(file_url)
        warnings.warn("Unzipping %s" % final)

        # shutil.move(local_file,final)
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(res_path)
    return final


def get_memory_usage():
    """

    Returns
    -------

    """
    try:
        import resource

        memory_used_go = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 / 1000.0
        )
    except ImportError:  # on windows resource is not usable
        import psutil

        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.0
    str_mem = "RAM usage: %1.1f Go" % memory_used_go
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

    for d, k in zip(
        repeat(tuple(sizes.keys())), zip(np.ndindex(tuple(sizes.values())))
    ):
        yield {k: l for k, l in zip(d, k[0])}


def get_l1c_filepath(l1b_fullpath, version, outputdir=None, makedir=True):
    """

    Args:
        l1b_fullpath: str .nc l1b full path
        version : str (eg. 1.2 or B01)
        outputdir: str [optional] default is l1c subdirectory // l1b inputs
        makedir: bool [optional] default is True
    Returns:
        l1c_full_path: str
    """
    safe_file = os.path.basename(os.path.dirname(l1b_fullpath))
    if outputdir is None:
        run_directory = os.path.dirname(os.path.dirname(l1b_fullpath))
        # Output file directory
        pathout_root = run_directory.replace("l1b", "l1c")
    else:
        pathout_root = outputdir
    # pathout = os.path.join(pathout_root, version, safe_file)
    safe_start_date = datetime.datetime.strptime(
        safe_file.split("_")[5], "%Y%m%dT%H%M%S"
    )
    pathout = os.path.join(
        pathout_root,
        safe_start_date.strftime("%Y"),
        safe_start_date.strftime("%j"),
        safe_file,
    )

    # Output filename
    l1c_full_path = os.path.join(
        pathout, os.path.basename(l1b_fullpath).replace("L1B", "L1C")
    )
    # add the product ID in the SAFE  name
    basesafe = os.path.basename(os.path.dirname(l1c_full_path))
    basesafe0 = copy.copy(basesafe)
    if (
        len(basesafe.split("_")) == 10
    ):  # classical ESA SLC naming #:TODO once xsarslc will be updated this case could be removed
        basesafe = basesafe.replace(".SAFE", "_" + version.upper() + ".SAFE")
    else:  # there is already a product ID in the L1B SAFE name
        lastpart = basesafe.split("_")[-1]
        basesafe = basesafe.replace(lastpart, version.upper() + ".SAFE")
    l1c_full_path = l1c_full_path.replace(basesafe0, basesafe)

    # suffix measurement
    measu_base = os.path.basename(l1c_full_path)
    if "s1" in measu_base[0:2]:
        measu_base = "l1c-" + measu_base
    elif "l1b" in measu_base[0:3]:
        measu_base = measu_base.replace("l1b", "l1c")
    measu_base = measu_base.replace("slc", "xsp")  # security old products
    measu_base = measu_base.replace("L1C_xspec_IFR_", "")  # security
    l1c_full_path = l1c_full_path.replace(os.path.basename(l1c_full_path), measu_base)

    # lastpiece = l1c_full_path.split("_")[-1]
    if "_" in os.path.basename(l1c_full_path):
        lastpiece = "_" + l1c_full_path.split("_")[-1]
    elif "-" in os.path.basename(l1c_full_path):
        # lastpiece = l1c_full_path[-6:]
        lastpiece = "-" + l1c_full_path.split("-")[-1]
    l1c_full_path = l1c_full_path.replace(lastpiece, "-" + version.lower() + ".nc")

    logging.debug("File out: %s ", l1c_full_path)
    if not os.path.exists(os.path.dirname(l1c_full_path)) and makedir:
        os.makedirs(os.path.dirname(l1c_full_path), 0o0775)
    l1c_full_path = os.path.normpath(l1c_full_path).replace(
        "\\", "/"
    )  # platform windows security
    return l1c_full_path
