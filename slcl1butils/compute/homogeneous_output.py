import logging

import numpy as np
import xarray as xr

from slcl1butils.get_config import get_conf

conf = get_conf()


def add_missing_variables(ds_intra, ds_inter) -> (xr.Dataset, xr.Dataset):
    """

    Parameters
    ----------
    ds_intra (xr.Dataset)
    ds_inter (xr.Dataset)

    Returns
    -------
        ds_intra (xr.Dataset)
        ds_inter (xr.Dataset)

    """
    for vv in conf["list_variables_expected_intra"]:
        if vv not in ds_intra:
            attrs = conf["list_variables_expected_intra"][vv]["attrs"]
            # dims_name = conf['list_variables_expected_intra'][vv]['dims']
            coords_name = conf["list_variables_expected_intra"][vv]["coords"]
            # dims = {}
            # for di in dims_name:
            #     if di in ds_intra.dims:
            #         dims[di] = ds_intra.dims[di]
            #     else:
            #         dims[di] = 1
            coords = {}
            for co in coords_name:
                if co in ds_intra.coords:
                    coords[co] = ds_intra.coords[co]
                elif co in ds_intra.dims:
                    coords[co] = np.arange(ds_intra.dims[co])
                else:
                    coords[co] = 1
            dims = [co for co in coords.keys()]
            ds_intra = ds_intra.assign(
                {vv: xr.DataArray(coords=coords, dims=dims, attrs=attrs)}
            )
            logging.info("empty %s added to intra", vv)
    for vv in conf["list_variables_expected_inter"]:
        if vv not in ds_inter:
            attrs = conf["list_variables_expected_intra"][vv]["attrs"]
            # dims_name = conf["list_variables_expected_intra"][vv]["dims"]
            coords_name = conf["list_variables_expected_intra"][vv]["coords"]
            # dims = {}
            # for di in dims_name:
            #     if di in ds_inter.dims:
            #         dims[di] = ds_inter.dims[di]
            coords = {}
            for co in coords_name:
                if co in ds_inter.coords:
                    coords[co] = ds_inter.coords[co]
                elif co in ds_inter.dims:
                    coords[co] = np.arange(ds_inter.dims[co])
                else:
                    coords[co] = 1
            dims = [co for co in coords.keys()]
            ds_inter = ds_inter.assign(
                {vv: xr.DataArray(coords=coords, dims=dims, attrs=attrs)}
            )
            logging.info("empty %s added to intra", vv)
    return ds_intra, ds_inter
