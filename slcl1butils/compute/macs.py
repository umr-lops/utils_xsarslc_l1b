import xarray as xr
import numpy as np


def compute_macs(xs, lambda_range_max=[50.]):
    """
    Parameters:
        xs: xarray.Dataset
        lambda_range_max: list of float
    :return:
        macs : xarray.Dataset containing imaginary and real part of MACS
    """
    # IMACS fix limits from Li et al., JGR 2019
    k_az_max = 2 * np.pi / 600.
    k_az_min = -2 * np.pi / 600.

    k_rg_max = 2 * np.pi / 15.
    # k_rg_min = 2*np.pi/lambda_range[1]#20 # 20

    im_macs = xr.combine_by_coords([xs.sel(k_az=slice(k_az_min, k_az_max),
                                           k_rg=slice(2 * np.pi / _lambda_range_max, k_rg_max)).imag.mean(
        dim=['k_az', 'k_rg']).assign_coords({'lambda_range_max_macs': _lambda_range_max}).expand_dims(
        'lambda_range_max_macs') for ik, _lambda_range_max in enumerate(lambda_range_max)])
    re_macs = xr.combine_by_coords([xs.sel(k_az=slice(k_az_min, k_az_max),
                                           k_rg=slice(2 * np.pi / _lambda_range_max, k_rg_max)).real.mean(
        dim=['k_az', 'k_rg']).assign_coords({'lambda_range_max_macs': _lambda_range_max}).expand_dims(
        'lambda_range_max_macs') for ik, _lambda_range_max in enumerate(lambda_range_max)])
    im_macs = im_macs.rename('macs_Im')
    re_macs = re_macs.rename('macs_Re')
    if 'pol' in im_macs:
        im_macs = im_macs.drop_vars('pol')
        re_macs = re_macs.drop_vars('pol')

    macs = xr.merge([im_macs, re_macs])

    return macs
