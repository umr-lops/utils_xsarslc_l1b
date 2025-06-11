import argparse
import logging
import os
import pdb
import time
import warnings
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from xarray import DataTree

import slcl1butils
from slcl1butils.coloc.coloc import (
    coloc_tiles_from_l1bgroup_with_raster,
    raster_cropping_in_polygon_bounding_box,
)
from slcl1butils.coloc.coloc_IW_WW3spectra import (
    resampleWW3spectra_on_TOPS_SAR_cartesian_grid,
)
from slcl1butils.coloc.coloc_XSP_with_GRD import add_grd_ifr_wind
from slcl1butils.compute.compute_from_l1b import compute_xs_from_l1b
from slcl1butils.compute.homogeneous_output import add_missing_variables
from slcl1butils.get_config import get_conf
from slcl1butils.get_polygons_from_l1b import get_swath_tiles_polygons_from_l1bgroup
from slcl1butils.raster_readers import (
    ecmwf_0100_1h,
    resource_strftime,
    ww3_global_yearly_3h,
    ww3_IWL1Btrack_hindcasts_30min,
)
from slcl1butils.utils import get_l1c_filepath, get_memory_usage, netcdf_compliant

warnings.simplefilter(action="ignore")
conf = get_conf()


def do_L1C_SAFE_from_L1B_SAFE(
    full_safe_file,
    version,
    outputdir,
    ancillary_list,
    colocat=True,
    time_separation="2tau",
    overwrite=False,
    output_format="nc",
    dev=False,
):
    """

    Args:
        full_safe_file: str (e.g. /path/to/l1b-ifremer-dataset/..SAFE)
        version: str version of the product to generate
        outputdir: str where to store l1c netcdf files
        ancillary_list: dict of dict with names of the dataset (defined in conf.yaml or localconfig.yaml) to be colocated
        colocat: bool
        time_separation: str (e.g. '2tau')
        overwrite: bool True -> overwrite existing l1c if it exists
        output_format (str): .nc only supported for now
        dev: bool True -> early stop after one l1b nc file processing to dev/test

    Returns:

    """
    l1c_full_path = None
    # Ancillary data to be colocated
    # ancillary_ecmwf = {}
    # ancillary_ecmwf["resource"] = conf["ecmwf0.1_pattern"]
    # ancillary_ecmwf["step"] = 1
    # ancillary_ecmwf["name"] = "ecmwf_0100_1h"

    # ancillary_ww3 = {}
    # ancillary_ww3["resource"] = conf["ww3_pattern"]
    # ancillary_ww3["step"] = 3
    # ancillary_ww3["name"] = "ww3_global_yearly_3h"

    # ancillary_ww3hindcast_field = {}
    # ancillary_ww3hindcast_field["resource"] = conf["ww3hindcast_field_pattern"]
    # ancillary_ww3hindcast_field["step"] = 3
    # ancillary_ww3hindcast_field["name"] = "ww3hindcast_field"

    # ancillary_list = [ancillary_ecmwf]#,ancillary_ww3]
    # ancillary_list = [ancillary_ecmwf, ancillary_ww3hindcast_field]
    logging.info("ancillary data: %s", ancillary_list)

    #
    safe_file = os.path.basename(full_safe_file)
    run_directory = os.path.dirname(full_safe_file) + "/"

    # Processing Parameters:

    # files = glob(os.path.join(run_directory, safe_file, "*_L1B_*nc"))
    files = glob(os.path.join(run_directory, safe_file, "l1*.nc"))
    logging.info("Number of files: %s", len(files))
    if len(files) == 0:
        return None

    # Loop on L1B netCDF files (per slice)
    if dev:
        logging.info("dev mode -> only one L1B file to treat")
        files = files[0:1]
    cpt = defaultdict(int)
    pbar = tqdm(range(len(files)))
    for ii in pbar:
        cpt["L1B_treated"] += 1
        pbar.set_description("")
        l1b_fullpath = files[ii]
        l1c_full_path = get_l1c_filepath(
            l1b_fullpath, version=version, outputdir=outputdir
        )
        if os.path.exists(l1c_full_path) and overwrite is False:
            logging.debug("%s already exists", l1c_full_path)
            cpt["nc_out_already_present"] += 1
        else:
            ds_intra, ds_inter, flag_ancillaries_added = enrich_onesubswath_l1b(
                l1b_fullpath,
                ancillary_list=ancillary_list,
                colocat=colocat,
                time_separation=time_separation,
            )
            # add source L1B product
            ds_intra.attrs["L1B_XSP_PATH"] = os.path.dirname(l1b_fullpath)
            ds_inter.attrs["L1B_XSP_PATH"] = os.path.dirname(l1b_fullpath)
            ds_intra, ds_inter = add_missing_variables(ds_intra, ds_inter)
            for anc in flag_ancillaries_added:
                if flag_ancillaries_added[anc]:
                    cpt[anc + " ancillary_field_added"] += 1
                else:
                    cpt[anc + " missing"] += 1
            # if (
            #     "xspectra_Re" in ds_inter
            #     or "xsSAR" in ds_inter
            #     or "xspectra" in ds_inter
            # ):
            #     pass # nothing to do
            # else:
            #     logging.debug(
            #         "there is no xspectra in this subswath -> creation of empty xspectra variables"
            #     )
            #     cpt["L1B_without_spectra"] += 1
            #     freq_sample = 453
            #     freq_line = 50
            #     # list of variables that would currently miss {'bt_thresh', 'azimuth_cutoff_error', 'nesz_filt',
            #     'macs_Im', 'bright_targets_histogram', 'lambda_range_max_macs', 'normalized_variance_filt', 'macs_Re',
            #     'cwave_params', 'tau', 'k_rg', 'azimuth_cutoff', 'sigma0_filt', 'phi_hf', 'doppler_centroid', 'k_az',
            #     'k_gp'}
            #     #k_rg(burst, tile_sample, freq_sample)
            #     #k_az(freq_line)
            #     empty_xsp = np.nan*np.ones((ds_intra.burst.size,
            #                                                         ds_intra.tile_line.size,
            #                                                         ds_intra.tile_sample.size,
            #                                                         freq_line,freq_sample,1))
            #     ds_intra['xspectra_2tau_Re'] = xr.DataArray(empty_xsp,dims=['burst', 'tile_line',
            #                                                         'tile_sample', 'freq_line', 'freq_sample', '2tau'])
            #     ds_intra['xspectra_2tau_Im'] = ds_intra['xspectra_2tau_Re']
            #
            #      # xspectra_2tau_Re(burst, tile_line, tile_sample, freq_line, freq_sample, \2tau)
            ds_intra = netcdf_compliant(ds_intra)
            ds_inter = netcdf_compliant(ds_inter)
            save_l1c_to_netcdf(l1c_full_path, ds_intra, ds_inter, version=version)
            cpt["saved_in_nc"] += 1

    logging.info("cpt: %s", cpt)
    return l1c_full_path


def enrich_onesubswath_l1b(
    l1b_fullpath, ancillary_list=None, colocat=True, time_separation="2tau"
):
    """

    Args:
        l1b_fullpath: str one single sub-swath
        ancillary_list: dict
        colocat: cool
        time_separation: str

    Returns:
        ds_intra: xarray.Dataset
        ds_inter: xarray.Dataset
        ancillary_fields_added: bool
    """

    logging.debug("File in: %s", l1b_fullpath)
    if ancillary_list is None:
        ancillary_list = {}
    # ====================
    # X-SPEC
    # ====================
    #
    # Intraburst at 2tau x-spectra
    burst_type = "intra"

    xs_intra, ds_intra = compute_xs_from_l1b(
        l1b_fullpath, burst_type=burst_type, time_separation=time_separation
    )
    # Interburst x-spectra
    burst_type = "inter"
    time_separation = "None"
    xs_inter, ds_inter = compute_xs_from_l1b(
        l1b_fullpath, burst_type=burst_type, time_separation=time_separation
    )
    # ====================
    # COLOC
    # ====================
    flag_ancillaries = {}
    if colocat:
        for ancillary_name in ancillary_list:
            logging.debug("ancillary: %s", ancillary_name)
            ds_intra, ds_inter, ancillary_fields_added = append_ancillary_field(
                ancillary_list[ancillary_name], ds_intra, ds_inter
            )
            flag_ancillaries[ancillary_name] = ancillary_fields_added
    if (
        "ww3hindcast_spectra" in ancillary_list
        or "ww3CCIseastate_spectra" in ancillary_list
    ):
        idx = None
        for uui, uu in enumerate(ancillary_list):
            if "spectra" in uu:
                idx = uu
        ww3spectra_matching_name = ancillary_list[idx]["name"]
        logging.info(
            "the product used to add wave spectra is: %s", ww3spectra_matching_name
        )
        (
            ds_intra,
            flag_ww3spectra_added,
            flag_ww3spectra_found,
        ) = resampleWW3spectra_on_TOPS_SAR_cartesian_grid(
            dsar=ds_intra, xspeckind="intra", nameWW3sp_product=ww3spectra_matching_name
        )
        flag_ancillaries["ww3spectra_intra"] = flag_ww3spectra_added
        (
            ds_inter,
            flag_ww3spectra_added,
            flag_ww3spectra_found,
        ) = resampleWW3spectra_on_TOPS_SAR_cartesian_grid(
            dsar=ds_inter, xspeckind="inter", nameWW3sp_product=ww3spectra_matching_name
        )
        flag_ancillaries["ww3spectra_inter"] = flag_ww3spectra_added
    if "s1grd" in ancillary_list:
        for uui, uu in enumerate(ancillary_list):
            if "s1grd" in uu:
                idx = uu
        entry_conf = ancillary_list[idx]
        slcgrdlist = pd.read_csv(entry_conf["listing"], names=["slc", "grd"])
        l1cgrids, cpt = add_grd_ifr_wind(
            dsintra=ds_intra,
            dsinter=ds_inter,
            confgrd=entry_conf,
            dfpairs_slc_grd=slcgrdlist,
        )
        ds_intra = l1cgrids["intraburst"]
        ds_inter = l1cgrids["interburst"]
        logging.info("GRD wind added.")

    return ds_intra, ds_inter, flag_ancillaries


def append_ancillary_field(ancillary, ds_intra, ds_inter):
    """

    Args:
        ancillary: xarray.Dataset
        ds_intra: xarray.Dataset
        ds_inter: xarray.Dataset

    Returns:
        ds_intra: xarray.Dataset
        ds_inter: xarray.Dataset
        ancillary_fields_added: bool
    """
    # For each L1B
    # burst_type = 'intra'
    # l1b_ds = xr.open_dataset(_file,group=burst_type+'burst')

    # ===========================================
    # Check if the ancillary data can be found
    ancillary_fields_added = False
    sar_date = datetime.strptime(
        str.split(ds_intra.attrs["start_date"], ".")[0], "%Y-%m-%d %H:%M:%S"
    )
    closest_date, filename = resource_strftime(
        ancillary["pattern"], step=int(ancillary["step"]), date=sar_date
    )
    if len(glob(filename)) != 1:
        logging.debug("no ancillary files matching %s", filename)
        return ds_intra, ds_inter, ancillary_fields_added
    else:
        ancillary_fields_added = True
    # Getting the raster from anxillary data
    if ancillary["name"] == "ecmwf_0100_1h":
        raster_ds = ecmwf_0100_1h(filename)
    elif ancillary["name"] == "ww3_global_yearly_3h":
        raster_ds = ww3_global_yearly_3h(filename, closest_date)
    elif ancillary["name"] in ["ww3hindcast_field", "ww3_global_cciseastate"]:
        raster_ds = ww3_IWL1Btrack_hindcasts_30min(glob(filename)[0], closest_date)
    elif ancillary["name"] in [
        "ww3hindcast_spectra",
        "ww3CCIseastate_spectra",
        "s1-iw-GRD-Ifr-wind",
    ]:
        pass  # nothing to do here, there is a specific method called later in the code.
        return ds_intra, ds_inter, ancillary_fields_added
    else:
        raise ValueError("%s ancillary name not handled" % ancillary["name"])
    # Get the polygons of the swath data
    first_pola_available = ds_intra.coords["pol"].data[0]
    polygons, coordinates, variables = get_swath_tiles_polygons_from_l1bgroup(
        ds_intra, polarization=first_pola_available, swath_only=True
    )
    # Crop the raster to the swath bounding box limit
    raster_bb_ds = raster_cropping_in_polygon_bounding_box(
        polygons["swath"][0], raster_ds
    )
    raster_bb_ds.attrs["name"] = ancillary["name"]
    # Loop on the grid in the product
    burst_types = ["intra", "inter"]
    for burst_type in burst_types:
        # Define the dataset to work on
        # get the mapped raster onto swath grid for each tile
        if burst_type == "intra":
            # l1b_ds_intra = xr.open_dataset(_file,group=burst_type+'burst')
            # _ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_intra, raster_bb_ds, apply_merging=False)
            # ds_intra_list.append(_ds)
            _ds_intra = coloc_tiles_from_l1bgroup_with_raster(
                ds_intra, raster_bb_ds, apply_merging=False
            )
            # ds_intra_list.append(_ds_intra)
            # Merging the datasets
            ds_intra = xr.merge([ds_intra, _ds_intra])
            ds_intra.attrs[ancillary["name"] + "_pattern"] = filename
        else:
            # l1b_ds_inter = xr.open_dataset(_file,group=burst_type+'burst')
            # _ds = coloc_tiles_from_l1bgroup_with_raster(l1b_ds_inter, raster_bb_ds, apply_merging=False)
            # ds_inter_list.append(_ds)
            _ds_inter = coloc_tiles_from_l1bgroup_with_raster(
                ds_inter, raster_bb_ds, apply_merging=False
            )
            # ds_inter_list.append(_ds_inter)
            # Merging the datasets
            ds_inter = xr.merge([ds_inter, _ds_inter])
            ds_inter.attrs[ancillary["name"] + "_pattern"] = filename
    logging.debug("ancillary fields added")
    return ds_intra, ds_inter, ancillary_fields_added


def save_l1c_to_netcdf(l1c_full_path, ds_intra, ds_inter, version):
    """

    Args:
        l1c_full_path: str
        ds_intra: xr.Dataset intra burst
        ds_inter: xr.Dataset inter burst
        version : str (e.g. 1.4)

    Returns:

    """
    #
    # Arranging & saving Results
    # Building the output datatree
    dt = DataTree()
    burst_type = "intra"
    dt[burst_type + "burst"] = DataTree(ds_intra)
    burst_type = "inter"
    dt[burst_type + "burst"] = DataTree(ds_inter)

    dt.attrs["version_slcl1butils"] = slcl1butils.__version__
    dt.attrs["product_version"] = version
    dt.attrs["processor"] = __file__
    dt.attrs["generation_date"] = datetime.today().strftime("%Y-%b-%d")
    # Saving the results in netCDF
    dt.to_netcdf(l1c_full_path)


def save_l1c_to_zarr(l1c_full_path, ds_intra, ds_inter, version):
    """
    zarr if not a good idea -> more than 1400 inodes generated on file system for single measurement
    Args:
        l1c_full_path:
        ds_intra:
        ds_inter:
        version:

    Returns:

    """
    dt = DataTree()
    burst_type = "intra"
    dt[burst_type + "burst"] = DataTree(data=ds_intra)
    burst_type = "inter"
    dt[burst_type + "burst"] = DataTree(data=ds_inter)

    dt.attrs["version_slcl1butils"] = slcl1butils.__version__
    dt.attrs["product_version"] = version
    dt.attrs["processor"] = __file__
    dt.attrs["generation_date"] = datetime.today().strftime("%Y-%b-%d")
    # Saving the results in netCDF
    dt.to_zarr(l1c_full_path)


def main():
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="L1B->L1C")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the existing outputs [default=False]",
        required=False,
    )
    parser.add_argument(
        "--l1bsafe", required=True, help="L1B IW XSP SAFE (Sentinel-1 IFREMER) path"
    )
    parser.add_argument(
        "--outputdir",
        required=False,
        help="directory where to store output netCDF files",
        default=conf["iw_outputdir"],
    )
    parser.add_argument(
        "--version",
        help="set the output product version (e.g. 0.3) default version will be read from config.yaml",
        required=False,
        default=conf["l1c_iw_version"],
    )
    parser.add_argument(
        "--ww3spectra",
        action="store_true",
        default=False,
        help="add WW3 spectra to L1C [default is False]",
    )
    parser.add_argument(
        "--grdwind",
        action="store_true",
        default=False,
        help="add GRD Ifremer (cyclobs) Wind product to L1C [default is False]",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="dev mode stops the computation early",
    )
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()
    logging.info("product version to produce: %s", args.version)
    logging.info("outputdir will be: %s", args.outputdir)
    ancillary_list = {
        "ecmwf_0100_1h": conf["auxilliary_dataset"]["ecmwf_0100_1h"],
        # "ww3hindcast_field": conf["auxilliary_dataset"]["ww3hindcast_field"],
        "ww3hindcast_field": conf["auxilliary_dataset"]["ww3_global_cciseastate"],
    }
    if args.ww3spectra:
        # ancillary_list["ww3hindcast_spectra"] = conf["auxilliary_dataset"][
        #    "ww3hindcast_spectra"
        # ]
        ancillary_list["ww3CCIseastate_spectra"] = conf["auxilliary_dataset"][
            "ww3CCIseastate_spectra"
        ]
    if args.grdwind is True:
        ancillary_list["s1grd"] = conf["auxilliary_dataset"]["s1iwgrdwind"]
    final_L1C_path = do_L1C_SAFE_from_L1B_SAFE(
        args.l1bsafe,
        version=args.version,
        outputdir=args.outputdir,
        ancillary_list=ancillary_list,
        colocat=True,
        time_separation="2tau",
        overwrite=args.overwrite,
        dev=args.dev,
        output_format="nc",
    )
    logging.info("last measurement generated for this SAFE: %s", final_L1C_path)
    logging.info("successful SAFE processing")
    logging.info("peak memory usage: %s ", get_memory_usage())
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
