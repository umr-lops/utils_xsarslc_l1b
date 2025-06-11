import argparse
import logging
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from tqdm import tqdm

import slcl1butils
from slcl1butils.coloc.coloc import (
    coloc_tiles_from_l1bgroup_with_raster,
    raster_cropping_in_polygon_bounding_box,
)
from slcl1butils.coloc.coloc_WV_WW3spectra import (
    resampleWW3spectra_on_SAR_cartesian_grid,
)
from slcl1butils.compute.compute_from_l1b import compute_xs_from_l1b_wv
from slcl1butils.get_config import get_conf
from slcl1butils.raster_readers import (
    ecmwf_0100_1h,
    resource_strftime,
    ww3_global_yearly_3h,
    ww3_IWL1Btrack_hindcasts_30min,
)
from slcl1butils.utils import get_memory_usage, netcdf_compliant, xndindex

warnings.simplefilter(action="ignore")
conf = get_conf()


def do_L1C_SAFE_from_L1B_SAFE(
    full_safe_file,
    productid,
    outputdir,
    colocat=True,
    time_separation="2tau",
    overwrite=False,
    dev=False,
):
    """

    transform a Level-1B XSP WV product into a Level-1C XSP product

    Args:
        full_safe_file: str (e.g. /path/to/l1b-ifremer-dataset/S1A_WV1_XSP__1SSV_20230501T055632_20230501T061158_048336_05D036_6846_A24.nc)
        productid: str productid of the product to generate
        outputdir: str where to store l1c netcdf files
        colocat: bool
        time_separation: str (e.g. '2tau')
        overwrite: bool True -> overwrite existing l1c if it exists
        dev: bool True -> early stop after one l1b nc file processing to dev/test

    Returns:

    """

    # Ancillary data to be colocated
    # ancillary_ecmwf = {}
    # ancillary_ecmwf["resource"] = conf["ecmwf0.1_pattern"]
    # ancillary_ecmwf["step"] = 1
    # ancillary_ecmwf["name"] = "ecmwf_0100_1h"
    #
    # ancillary_ww3 = {}
    # ancillary_ww3["resource"] = conf["ww3_pattern"]
    # ancillary_ww3["step"] = 3
    # ancillary_ww3["name"] = "ww3_global_yearly_3h"
    # "ww3hindcast_field": conf["auxilliary_dataset"]["ww3_global_cciseastate"],
    ancillary_list = {
        "ecmwf_0100_1h": conf["auxilliary_dataset"]["ecmwf_0100_1h"],
        # "ww3hindcast_field": conf["auxilliary_dataset"]["ww3hindcast_field"],
        "ww3hindcast_field": conf["auxilliary_dataset"]["ww3_global_cciseastate"],
    }
    # ancillary_list = [ancillary_ecmwf]#,ancillary_ww3]
    # ancillary_list = [ancillary_ecmwf, ancillary_ww3]
    logging.info("ancillary data: %s", ancillary_list)

    # Processing Parameters:

    # files = glob(os.path.join(full_safe_file, "*_XSP_*nc"))
    # cpt_total = len(files)
    cpt = defaultdict(int)
    # logging.info("Number of files: %s", cpt_total)
    # if len(files) == 0:
    #     return None

    # Loop on L1B netCDF files (per slice)
    # if dev:
    # logging.info("dev mode -> only one L1B file to treat")
    # files = files[0:1]
    # pbar = tqdm(range(len(files)))
    cpt_success = 0
    cpt_already = 0
    cpt_total = 1
    cpt_ancillary_products_found = 0
    # for ii in pbar:
    # if dev:
    #     pbar.set_description(
    #         "sucess: %s/%s ancillary : %s, already: %s"
    #         % (cpt_success, len(files), cpt_ancillary_products_found, cpt_already)
    #     )
    # else:
    #     pbar.set_description()
    # l1b_fullpath = files[ii]
    l1b_fullpath = full_safe_file
    l1c_full_path, l1b_product_version = get_l1c_filepath(
        l1b_fullpath, productid=productid, outputdir=outputdir
    )
    if os.path.exists(l1c_full_path) and overwrite is False:
        logging.debug("%s already exists", l1c_full_path)
        cpt_already += 1
    else:
        dtwv, ancillaries_flag_added = enrich_onesubswath_l1b(
            l1b_fullpath,
            ancillaries=ancillary_list,
            colocat=colocat,
            time_separation=time_separation,
        )
        # if ancillary_product_found:
        #     cpt_ancillary_products_found += 1
        for anc in ancillaries_flag_added:
            if ancillaries_flag_added[anc]:
                cpt[anc + " OK"] += 1
            else:
                cpt[anc + " missing"] += 1
        save_l1c_to_netcdf(
            l1c_full_path,
            dtwv,
            productid_L1C=productid,
            productid_L1B=l1b_product_version,
        )
        # save_l1c_to_zarr(l1c_full_path, ds_intra, productid=productid, version_L1B=l1b_product_version)
        logging.debug("successfully wrote  %s", l1c_full_path)
        cpt_success += 1
    logging.info("cpt %s", cpt)
    logging.info("last file written %s", l1c_full_path)
    return cpt_success, cpt_already, cpt_total, cpt_ancillary_products_found


def enrich_onesubswath_l1b(
    l1b_fullpath, ancillaries=None, colocat=True, time_separation="2tau"
):
    """

    this method allow to associate each WV tiles of a given netcdf file (WV1 or WV2)

    Parameters
    ----------
        l1b_fullpath str: e.g. S1A_WV1_XSP__1SSV_20230501T055632_20230501T061158_048336_05D036_6846_A24.nc
        ancillaries dict: optional [default -> nothing]
        colocat: bool-> add ancillary information
        time_separation: str 2tau or 1tau

    Returns
    -------

    """

    logging.debug("File in: %s", l1b_fullpath)
    if ancillaries is None:
        ancillaries = {}
    # ====================
    # X-SPEC
    # ====================
    #
    # Intraburst at 2tau x-spectra
    xs_intra_groups, ds_intra = compute_xs_from_l1b_wv(
        l1b_fullpath, time_separation=time_separation
    )

    # ====================
    # COLOC
    # ====================
    ancillaries_flag_added = {}
    if colocat:
        for ancillary in ancillaries:
            logging.info("ancillary : %s", ancillary)

            (
                ds_intra,
                ancillary_product_found,
                flag_ancillary_field_added,
            ) = append_ancillary_field(ancillaries[ancillary], ds_intra)
            official_name_ancillary = ancillaries[ancillary]["name"]
            ancillaries_flag_added[official_name_ancillary] = flag_ancillary_field_added

    # this part is commented temporarily to test only the association with raster fields alone
    if "WV" in l1b_fullpath:
        logging.info("ancillary WW3 spectra")
        dims_to_expand = ["tile_sample", "tile_line"]
        imagettestiles_sizes = {d: k for d, k in ds_intra["longitude"].sizes.items()}
        out = []
        all_cases = [ii for ii in xndindex(imagettestiles_sizes)]
        for ix in tqdm(
            range(len(all_cases))
        ):  # loop over tile_sample, tile_line and time
            i = all_cases[ix]
            one_tile = ds_intra[i]
            (
                one_tile,
                flag_ww3spectra_added,
                flag_ww3spectra_found,
            ) = resampleWW3spectra_on_SAR_cartesian_grid(dsar=one_tile)
            if flag_ww3spectra_found:
                ancillaries_flag_added["ww3spectra"] = flag_ww3spectra_added
            if (
                ix == 0
            ):  # approximation all the WV of a given incidence (WV1 or WV2) can use the same k_az and k_rg
                all_coord_to_drop = [
                    uu
                    for uu in one_tile["k_az"].coords
                    if uu not in ["k_az", "k_rg", "ky", "kx"]
                ]
                k_az_ref = one_tile["k_az"].drop(all_coord_to_drop)
                k_rg_ref = one_tile["k_rg"].drop(all_coord_to_drop)
            else:
                one_tile = one_tile.assign_coords({"k_az": k_az_ref})
                one_tile = one_tile.assign_coords({"k_rg": k_rg_ref})
            one_tile["time"] = xr.DataArray([one_tile["time"].values], dims="time")
            one_tile["longitude"] = xr.DataArray([one_tile["longitude"]], dims="time")
            one_tile["latitude"] = xr.DataArray([one_tile["latitude"]], dims="time")
            one_tile["sensing_time"] = xr.DataArray(
                [one_tile["sensing_time"].values], dims="time"
            )
            out.append(one_tile)
        # ds_intra = xr.combine_by_coords([x.expand_dims(dims_to_expand) for x in out], combine_attrs='drop_conflicts') # killed on a 17.5km tile
        ds_intra = xr.concat([x.expand_dims(dims_to_expand) for x in out], dim="time")
    return ds_intra, ancillaries_flag_added


def append_ancillary_field(ancillary, ds_intra):
    """

    method to associate regular grids from numerical models to a SAR dataset composed of sub tiles

    Parameters
    ----------
    ancillary (dict):
    ds_intra (xarray.Dataset): Level-1B XSP WV1 or WV2 intra burst

    Returns
    -------

    """
    ancillary_product_found = False

    # ===========================================
    # Check if the ancillary data can be found
    flag_ancillary_field_added = False
    logging.debug("attrs : %s0", ds_intra.attrs["start_date"])

    sar_date = datetime.strptime(
        str.split(ds_intra.attrs["start_date"], ".")[0], "%Y-%m-%d %H:%M:%S"
    )
    closest_date, filename = resource_strftime(
        ancillary["pattern"], step=ancillary["step"], date=sar_date
    )
    if len(glob(filename)) != 1:
        logging.debug("no ancillary files matching %s", filename)
    else:
        flag_ancillary_field_added = True
        raster_ds = None
        ancillary_product_found = True
        # Getting the raster from anxillary data
        if ancillary["name"] == "ecmwf_0100_1h":
            raster_ds = ecmwf_0100_1h(filename)
        elif ancillary["name"] == "ww3_global_yearly_3h":
            raster_ds = ww3_global_yearly_3h(filename, closest_date)
        elif ancillary["name"] in ["ww3hindcast_field", "ww3_global_cciseastate"]:
            raster_ds = ww3_IWL1Btrack_hindcasts_30min(glob(filename)[0], closest_date)
        elif ancillary["name"] in ["ww3hindcast_spectra", "ww3CCIseastate_spectra"]:
            pass  # nothing to do here, there is a specific method called later in the code.
            return ds_intra, flag_ancillary_field_added
        else:
            raise ValueError("%s ancillary name not handled" % ancillary["name"])

        # Get the polygons of the swath data
        all_imagettes = []
        for ti, tt in enumerate(ds_intra.time):
            subset_imagette = ds_intra.isel(time=ti)
            idx = [0, 1, 3, 2]
            coo = np.stack(
                [
                    subset_imagette["corner_longitude"].values.ravel()[idx],
                    subset_imagette["corner_latitude"].values.ravel()[idx],
                ]
            ).T
            polygon_imagette = Polygon(coo)
            # Crop the raster to the swath bounding box limit

            raster_bb_ds = raster_cropping_in_polygon_bounding_box(
                polygon_imagette, raster_ds
            )
            if "name" not in raster_bb_ds.attrs:
                raster_bb_ds.attrs["name"] = ancillary["name"]
            # Loop on the grid in the product

            _ds_imagette_with_raster = coloc_tiles_from_l1bgroup_with_raster(
                subset_imagette, raster_bb_ds, apply_merging=False
            )
            # Merging the datasets
            all_imagettes.append(_ds_imagette_with_raster)
            # ds_intra = xr.merge([ds_intra, _ds_imagette_with_raster])
        ds_raster = xr.concat(all_imagettes, dim="time")
        ds_intra = xr.merge([ds_intra, ds_raster])

    return ds_intra, ancillary_product_found, flag_ancillary_field_added


def get_l1c_filepath(
    l1b_fullpath, productid, outputdir=None, makedir=True
) -> (str, str):
    """

    transform a Level-1B WV path into a Level-1C XSP WV

    Args:
        l1b_fullpath: str .nc level-1B full path "S1...SAFE.nc"
        productid : str  productid of Level-1C product to write (e.g. B49)
        outputdir: str [optional] default is l1c subdirectory taken from l1b input
        makedir: bool [optional]
    Returns:

    """
    safe_file = os.path.basename(l1b_fullpath)
    if outputdir is None:
        run_directory = os.path.dirname(os.path.dirname(l1b_fullpath))
        # Output file directory
        pathout_root = run_directory.replace("l1b", "l1c")
    else:
        pathout_root = outputdir
    l1b_product_version = (
        safe_file.split("_")[-1].replace(".SAFE.nc", "").replace(".nc", "")
    )
    safe_file_l1c = safe_file.replace(l1b_product_version, productid)
    datedt_start_safe = datetime.strptime(safe_file_l1c.split("_")[5], "%Y%m%dT%H%M%S")
    l1c_full_path = os.path.join(
        pathout_root,
        datedt_start_safe.strftime("%Y"),
        datedt_start_safe.strftime("%j"),
        safe_file_l1c,
    )

    if makedir:
        os.makedirs(os.path.dirname(l1c_full_path), 0o0775, exist_ok=True)
    return l1c_full_path, l1b_product_version


def save_l1c_to_netcdf(l1c_full_path, ds_intra, productid_L1C, productid_L1B):
    """

    save the WV dataset associated to ancillary products on disk in netcdf format

    Args:
        l1c_full_path: str
        ds_intra: xr.Dataset intra burst
        productid_L1C : str (e.g. B78)
        productid_L1B : str  (e.g. A45)
    Returns:

    """
    #
    # Arranging & saving Results
    # Building the output datatree
    ds_intra = netcdf_compliant(ds_intra)

    ds_intra.attrs["version_l1butils"] = slcl1butils.__version__
    ds_intra.attrs["L1C_product_version"] = productid_L1C
    ds_intra.attrs["processor"] = __file__
    ds_intra.attrs["generation_date"] = datetime.today().strftime("%Y-%b-%d")
    ds_intra.attrs["L1B_product_version"] = productid_L1B
    # Saving the results in netCDF

    ds_intra.to_netcdf(l1c_full_path)
    ds_intra.close()
    logging.debug("output file written successfully: %s", l1c_full_path)


def main():
    """

    Returns
    -------

    """
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
        "--l1bsafe",
        required=True,
        help="Level-1B WV XSP SAFE (Sentinel-1 IFREMER) path .nc",
    )
    parser.add_argument(
        "--outputdir",
        required=False,
        help="directory where to store output netCDF files",
        default=conf["wv_outputdir"],
    )
    parser.add_argument(
        "--productid",
        help="set the output product ID (e.g. B48) default product ID will be read from config.yaml",
        required=False,
        default=conf["l1c_wv_version"],
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
    logging.info("product productid to produce: %s", args.productid)
    logging.info("outputdir will be: %s", args.outputdir)
    (
        cpt_success,
        cpt_already,
        cpt_total,
        cpt_ancillary_products_found,
    ) = do_L1C_SAFE_from_L1B_SAFE(
        args.l1bsafe,
        productid=args.productid,
        outputdir=args.outputdir,
        colocat=True,
        time_separation="2tau",
        overwrite=args.overwrite,
        dev=args.dev,
    )
    logging.info(
        "file written: %s/%s ancillary : %s, already: %s",
        cpt_success,
        cpt_total,
        cpt_ancillary_products_found,
        cpt_already,
    )
    if cpt_already + cpt_success == cpt_total:
        logging.info("successful L1C processing")
    else:
        logging.info("there is at least an error in this processing")
    logging.info("outputdir was: %s", args.outputdir)
    logging.info("peak memory usage: %s Mbytes", get_memory_usage())
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
