import argparse
import pdb

import slcl1butils
from slcl1butils.raster_readers import ecmwf_0100_1h
from slcl1butils.raster_readers import ww3_global_yearly_3h
from slcl1butils.raster_readers import resource_strftime
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import xarray as xr
from datatree import DataTree
import time
import logging
import sys, os
from slcl1butils.get_polygons_from_l1b import get_swath_tiles_polygons_from_l1bgroup
from slcl1butils.coloc.coloc import raster_cropping_in_polygon_bounding_box, coloc_tiles_from_l1bgroup_with_raster
from slcl1butils.compute.compute_from_l1b import compute_xs_from_l1b_wv
from slcl1butils.compute.cwave import compute_cwave_parameters
from slcl1butils.compute.macs import compute_macs
from slcl1butils.get_config import get_conf
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore')
conf = get_conf()


def do_L1C_SAFE_from_L1B_SAFE(full_safe_file, version, outputdir, cwave=True, macs=True, colocat=True,
                              time_separation='2tau', overwrite=False, dev=False):
    """

    Args:
        full_safe_file: str (e.g. /path/to/l1b-ifremer-dataset/..SAFE)
        version: str version of the product to generate
        outputdir: str where to store l1c netcdf files
        cwave: bool
        macs: bool
        colocat: bool
        time_separation: str (e.g. '2tau')
        overwrite: bool True -> overwrite existing l1c if it exists
        dev: bool True -> early stop after one l1b nc file processing to dev/test

    Returns:

    """

    # Ancillary data to be colocated
    ancillary_ecmwf = {}
    ancillary_ecmwf['resource'] = conf['ecmwf0.1_pattern']
    ancillary_ecmwf['step'] = 1
    ancillary_ecmwf['name'] = 'ecmwf_0100_1h'

    ancillary_ww3 = {}
    ancillary_ww3['resource'] = conf['ww3_pattern']
    ancillary_ww3['step'] = 3
    ancillary_ww3['name'] = 'ww3_global_yearly_3h'

    # ancillary_list = [ancillary_ecmwf]#,ancillary_ww3]
    ancillary_list = [ancillary_ecmwf, ancillary_ww3]
    logging.info('ancillary data: %s', ancillary_list)

    #
    safe_file = os.path.basename(full_safe_file)
    run_directory = os.path.dirname(full_safe_file) + '/'

    # Processing Parameters:

    # sth = macs * cwave

    files = glob(os.path.join(run_directory, safe_file, '*_L1B_*nc'))
    logging.info('Number of files: %s', len(files))
    if len(files) == 0:
        return None

    # Loop on L1B netCDF files (per slice)
    if dev:
        logging.info('dev mode -> only one L1B file to treat')
        files = files[0:1]
    pbar = tqdm(range(len(files)))
    for ii in pbar:
        pbar.set_description('')
        l1b_fullpath = files[ii]
        l1c_full_path,l1b_product_version = get_l1c_filepath(l1b_fullpath, version=version, outputdir=outputdir)
        if os.path.exists(l1c_full_path) and overwrite is False:
            logging.info('%s already exists', l1c_full_path)
        else:
            ds_intra = enrich_onesubswath_l1b(l1b_fullpath, ancillary_list=ancillary_list, cwave=cwave, macs=macs,
                                              colocat=colocat,
                                              time_separation=time_separation)
            save_l1c_to_netcdf(l1c_full_path, ds_intra, version=version,version_L1B=l1b_product_version)
            logging.info('successfully wrote  %s',l1c_full_path)
    return 0


def enrich_onesubswath_l1b(l1b_fullpath, ancillary_list=None, cwave=True, macs=True, colocat=True,
                           time_separation='2tau'):
    """

    Parameters
    ----------
    l1b_fullpath str
    ancillary_list [] optional
    cwave bool
    macs bool
    colocat bool
    time_separation str 2tau or 1tau

    Returns
    -------

    """

    logging.info('File in: %s', l1b_fullpath)
    if ancillary_list is None:
        ancillary_list = []
    # ====================
    # X-SPEC
    # ====================
    #
    # Intraburst at 2tau x-spectra
    burst_type = ''  # for WV it is an empty group name

    xs_intra, ds_intra = compute_xs_from_l1b_wv(l1b_fullpath, time_separation=time_separation)

    # ====================
    # CWAVE
    # ====================
    if cwave:
        #
        # CWAVE Processing Parameters
        kmax = 2 * np.pi / 25
        kmin = 2 * np.pi / 600
        Nk = 4
        Nphi = 5
        # Intraburst at 2tau CWAVE parameters
        ds_cwave_parameters_intra = compute_cwave_parameters(xs_intra, save_kernel=False, kmax=kmax, kmin=kmin,
                                                             Nk=Nk, Nphi=Nphi)
        # updating the L1B dataset
        ds_intra = xr.merge([ds_intra, ds_cwave_parameters_intra])

    # ====================
    # MACS
    # ====================
    if macs:
        # MACS parameters
        lambda_range_max = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
        # Intraburst at 2tau MACS
        ds_macs_intra = compute_macs(xs_intra, lambda_range_max=lambda_range_max)

        # updating the L1B dataset
        ds_intra = xr.merge([ds_intra, ds_macs_intra])

    # ====================
    # COLOC
    # ====================
    if colocat:
        for ancillary in ancillary_list:
            ds_intra = append_ancillary_field(ancillary, ds_intra)

    return ds_intra


def append_ancillary_field(ancillary, ds_intra):
    """

    Parameters
    ----------
    ancillary
    ds_intra xarray.Dataset

    Returns
    -------

    """
    # For each L1B
    # l1b_ds = xr.open_dataset(_file,group=burst_type+'burst')

    # ===========================================
    ## Check if the ancillary data can be found
    logging.debug('attrs : %s0',ds_intra.attrs['start_date'])
    sar_date = datetime.strptime(str.split(ds_intra.attrs['start_date'], '.')[0], '%Y-%m-%d %H:%M:%S')
    closest_date, filename = resource_strftime(ancillary['resource'], step=ancillary['step'], date=sar_date)
    if len(glob(filename)) != 1:
        logging.info('no ancillary files matching %s', filename)
    else:
        raster_ds = None
        # Getting the raster from anxillary data
        if ancillary['name'] == 'ecmwf_0100_1h':
            raster_ds = ecmwf_0100_1h(filename)
        if ancillary['name'] == 'ww3_global_yearly_3h':
            raster_ds = ww3_global_yearly_3h(filename, closest_date)

        # Get the polygons of the swath data
        polygons, coordinates, variables = get_swath_tiles_polygons_from_l1bgroup(ds_intra, swath_only=True)
        # Crop the raster to the swath bounding box limit

        raster_bb_ds = raster_cropping_in_polygon_bounding_box(polygons['swath'][0], raster_ds)

        # Loop on the grid in the product
        burst_types = ['']
        for burst_type in burst_types:
            # Define the dataset to work on
            # get the mapped raster onto swath grid for each tile

            # l1b_ds_intra = xr.open_dataset(_file,group=burst_type+'burst')
            # _ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_intra, raster_bb_ds, apply_merging=False)
            # ds_intra_list.append(_ds)
            _ds_intra = coloc_tiles_from_l1bgroup_with_raster(ds_intra, raster_bb_ds, apply_merging=False)
            # ds_intra_list.append(_ds_intra)
            # Merging the datasets
            ds_intra = xr.merge([ds_intra, _ds_intra])

    return ds_intra


def get_l1c_filepath(l1b_fullpath, version, outputdir=None, makedir=True):
    """

    Args:
        l1b_fullpath: str .nc l1b full path
        version : str (eg. 1.2)
        outputdir: str [optional] default is l1c subdirectory // l1b inputs

    Returns:

    """
    safe_file = os.path.basename(os.path.dirname(l1b_fullpath))
    if outputdir is None:
        run_directory = os.path.dirname(os.path.dirname(l1b_fullpath))
        # Output file directory
        pathout_root = run_directory.replace('l1b', 'l1c')
    else:
        pathout_root = outputdir
    pathout = os.path.join(pathout_root, version, safe_file)

    # Output filename
    l1c_full_path = os.path.join(pathout, os.path.basename(l1b_fullpath).replace('L1B', 'L1C'))
    lastpiece = l1c_full_path.split('_')[-1]
    l1b_product_version = lastpiece.replace('.nc','')
    l1c_full_path = l1c_full_path.replace(lastpiece, version + '.nc')
    logging.info('File out: %s ', l1c_full_path)
    if not os.path.exists(os.path.dirname(l1c_full_path)) and makedir:
        os.makedirs(os.path.dirname(l1c_full_path), 0o0775)
    return l1c_full_path,l1b_product_version


def save_l1c_to_netcdf(l1c_full_path, ds_intra, version,version_L1B):
    """

    Args:
        l1c_full_path: str
        ds_intra: xr.Dataset intra burst
        version : str (e.g. 1.4)
        version_L1B : str  (e.g. 1.4)
    Returns:

    """
    #
    # Arranging & saving Results
    #  Building the output datatree
    dt = DataTree()
    burst_type = 'intra'
    dt[burst_type + 'burst'] = DataTree(data=ds_intra)

    dt.attrs['version_l1butils'] = slcl1butils.__version__
    dt.attrs['L1C_product_version'] = version
    dt.attrs['processor'] = __file__
    dt.attrs['generation_date'] = datetime.today().strftime('%Y-%b-%d')
    dt.attrs['L1B_product_version'] = version_L1B

    #
    # Saving the results in netCDF
    dt.to_netcdf(l1c_full_path)


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


def main():
    """

    Returns
    -------

    """
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1B->L1C')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--l1bsafe', required=True, help='L1B IW XSP SAFE (Sentinel-1 IFREMER) path')
    parser.add_argument('--outputdir', required=False, help='directory where to store output netCDF files',
                        default=conf['iw_outputdir'])
    parser.add_argument('--version',
                        help='set the output product version (e.g. 0.3) default version will be read from config.yaml',
                        required=False, default=conf['l1c_iw_version'])
    parser.add_argument('--dev', action='store_true', default=False, help='dev mode stops the computation early')
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    t0 = time.time()
    logging.info('product version to produce: %s', args.version)
    logging.info('outputdir will be: %s', args.outputdir)
    do_L1C_SAFE_from_L1B_SAFE(args.l1bsafe, version=args.version, outputdir=args.outputdir, cwave=True, macs=True,
                              colocat=True,
                              time_separation='2tau', overwrite=args.overwrite, dev=args.dev)
    logging.info('peak memory usage: %s Mbytes', get_memory_usage())
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
