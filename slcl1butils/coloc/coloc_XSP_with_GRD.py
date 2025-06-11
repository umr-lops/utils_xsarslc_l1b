"""
A. Grouazel
creation: May 2025
purpose: meant to associate the windspeed from GRD processing with the Level-1B data
"""
import collections
import datetime
import glob
import logging
import os

import numpy as np
import xarray as xr
from scipy import spatial
from tqdm import tqdm

from slcl1butils.utils import xndindex

DEFAULT_RADIUS = 0.05  # deg used for the 5km XSP grids


def guess_grd_match(dsintra, confgrd) -> str:
    """
    list all the SAFE within the expected day and find the .nc that is the closest in dates

     Args:
        dsintra: xr.Dataset
        confgrd: dict of the config entry for IFR GRD product

    Returns:
        matching_l2wind_nc_file (str):
    """
    matching_l2wind_nc_file = None
    beg_pat, startdatexsp = build_pattern_search_ifr_wind_prod(dsintra, confgrd)
    pattern = os.path.join(beg_pat, "*.SAFE")
    logging.debug("pattern : %s", pattern)
    lstSAFE = glob.glob(pattern)
    all_start_dates = []
    all_nc = []
    for onesafe in lstSAFE:
        logging.debug("onesafe: %s", onesafe)
        tmplsnc = glob.glob(os.path.join(onesafe, "*.nc"))
        if len(tmplsnc) > 0:
            all_nc.append(tmplsnc[0])
            startdate_nc = datetime.datetime.strptime(
                os.path.basename(tmplsnc[0]).split("-")[4], "%Y%m%dt%H%M%S"
            )
            all_start_dates.append(startdate_nc)
    all_start_dates = np.array(all_start_dates)
    logging.debug("all_start_dates : %s", all_start_dates)
    bestcandidate = np.where(
        abs(all_start_dates - startdatexsp)
        == np.amin(abs(all_start_dates - startdatexsp))
    )[0][0]
    logging.debug("bestcandidate : %s", bestcandidate)
    bestcandidate_path = all_nc[bestcandidate]
    closest_date = datetime.datetime.strptime(
        os.path.basename(bestcandidate_path).split("-")[4], "%Y%m%dt%H%M%S"
    )
    if abs(closest_date - startdatexsp).total_seconds() <= 2:
        matching_l2wind_nc_file = bestcandidate_path
    return matching_l2wind_nc_file


def build_pattern_search_ifr_wind_prod(dsintra, confgrd) -> (str, datetime.datetime):
    """

    create path pattern up to doy sub-directory

    Args:
        dsintra: xr.Dataset
        confgrd: dict of the config entry for IFR GRD product

    Returns:

    """
    slc_safe = dsintra.attrs["safe"]
    mode = dsintra.attrs["swath"]
    satunit_long = "sentinel-1" + slc_safe[2].lower()
    satunit_short = "S1" + slc_safe[2].upper()
    datedt = datetime.datetime.strptime(slc_safe.split("_")[5], "%Y%m%dT%H%M%S")
    beg_pat = os.path.join(
        confgrd["pattern"],
        satunit_long,
        "L1",
        mode,
        satunit_short + "_" + mode + "_GRDH" + "_1S",
        datedt.strftime("%Y"),
        datedt.strftime("%j"),
    )
    return beg_pat, datedt


def get_path_ifr_wind(dsintra, confgrd, dfpairs_slc_grd=None) -> str:
    """

    Args:
        dsintra: xr.Dataset
        confgrd: dict of the config entry for IFR GRD product
        dfpairs_slc_grd: pd.DataFrame should contain columns 'grd' and 'slc' [optional]

    Returns:

    """
    logging.debug("start to add GRD wind information")
    matching_l2wind_nc_file = None
    if dfpairs_slc_grd is None:
        logging.debug("guess path of the closest grd file from XSP date")
        matching_l2wind_nc_file = guess_grd_match(dsintra, confgrd=confgrd)
    else:
        logging.debug("use the DataFrame to find the association XSP->GRD")
        slc_safe = dsintra.attrs["safe"]
        # corresponding_slc = xsp_safe.replace('XSP', 'SLC')
        idx = np.where(dfpairs_slc_grd["slc"].values == slc_safe)
        logging.debug("idx: %s", idx)
        if len(idx) != 0:
            corresponding_grd = dfpairs_slc_grd["grd"].iloc[idx[0][0]]
            logging.debug("corresponding_grd : %s", corresponding_grd)
            # "/home/datawor...1v0v10_1v2v5_1v1v4_1v1v1_1v2v1/cmod5n_ecmwf_norecal_15042025_l1l2only/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2015/269/"
            beg_pat, _ = build_pattern_search_ifr_wind_prod(dsintra, confgrd)
            pattern = os.path.join(beg_pat, corresponding_grd, "*.nc")
            logging.debug("pattern : %s", pattern)
            lstnctmp = glob.glob(pattern)
            if len(lstnctmp) != 0:
                matching_l2wind_nc_file = lstnctmp[0]  # there is only one .nc per .SAFE
    logging.debug("matching_l2wind_nc_file : %s", matching_l2wind_nc_file)
    return matching_l2wind_nc_file


def grd_core_tile_coloc(
    lontile, lattile, treegrd, radius_coloc, dsgrd, indexes_xsp
) -> xr.Dataset:
    """

    Args:
        lontile: float
        lattile: float
        treegrd: scipy.spatial
        radius_coloc: float
        dsgrd: xr.Dataset
        indexes_xsp: dict with coords of the XSP tile selected

    Returns:
        condensated_grd: xr.Dataset
    """
    neighbors = treegrd.query_ball_point([lontile, lattile], r=radius_coloc)
    indices = []
    condensated_grd = xr.Dataset()
    for oneneighbor in neighbors:
        (
            index_original_shape_grd_owiAzSize,
            index_original_shape_grd_owiRaSize,
        ) = np.unravel_index(oneneighbor, dsgrd["owiLon"].shape)
        indices.append(
            (index_original_shape_grd_owiAzSize, index_original_shape_grd_owiRaSize)
        )
    subset = [dsgrd.isel(owiAzSize=i, owiRaSize=j) for i, j in indices]
    if len(subset) > 0:
        grdclosest = xr.concat(subset, dim="points")

        # number of points in the radius
        condensated_grd["nb_grd_points"] = xr.DataArray(len(grdclosest["points"]))
        # variables wind/ std / mean /median
        fcts = {"mean": np.nanmean, "med": np.nanmedian, "std": np.nanstd}
        for vv in [
            "owiLat",
            "owiLon",
            "owiWindSpeed_co",
            "owiWindDirection_co",
            "owiWindSpeed",
            "owiWindDirection",
            "owiWindSpeed_cross",
            "owiWindDirection_cross",
        ]:
            for fct in fcts:
                if np.isfinite(grdclosest[vv].values).any():
                    if np.isfinite(grdclosest[vv].values).sum() == 1:
                        if fct == "std":
                            valval = np.nan
                        else:
                            valval = grdclosest[vv].values
                    else:
                        valval = fcts[fct](grdclosest[vv].values)

                else:
                    valval = (
                        np.nan
                    )  # to avoid RuntimeWarning Degrees of freedom <= 0 for slice
                condensated_grd["%s_%s" % (vv, fct)] = xr.DataArray(
                    valval,
                    attrs={
                        "description": "%s of %s variable from cyclobs Ifremer product"
                        % (fct, vv)
                    },
                )
                condensated_grd["%s_%s" % (vv, fct)].attrs = grdclosest[vv].attrs

        # if not 'lontile' in condensated_grd:
        #     condensated_grd = condensated_grd.assign_coords({'lontile':lontile,'lattile':lattile})
        #     condensated_grd = condensated_grd.expand_dims(
        #         ["lontile", "lattile"]
        #     )
        condensated_grd = condensated_grd.assign_coords(indexes_xsp)
        condensated_grd = condensated_grd.expand_dims(["tile_line", "tile_sample"])
    else:
        logging.info("one tile without GRD neighbors")
    return condensated_grd


def add_grd_ifr_wind(
    dsintra, dsinter, confgrd, dfpairs_slc_grd=None
) -> (dict, collections.defaultdict):
    """

    Args:
        dsintra: xr.Dataset Level-1B XSP intra burst grid
        dsinter: xr.Dataset Level-1B XSP inter burst grid
        dfpairs_slc_grd: pd.DataFrame should contain columns 'grd' and 'slc' [optional]

    Returns:
        l1cgrids: dict containing xr.Dataset Level-1B XSP intra and inter burst grids
        cpt: collections.defaultdict
    """
    cpt = collections.defaultdict(int)
    l1bgrids = {"intraburst": dsintra.load(), "interburst": dsinter.load()}
    l1cgrids = {}

    matching_l2wind_nc_file = get_path_ifr_wind(
        dsintra, confgrd, dfpairs_slc_grd=dfpairs_slc_grd
    )
    if "radius_coloc" in confgrd:
        radius_coloc = confgrd["radius_coloc"]
    else:
        radius_coloc = DEFAULT_RADIUS
    logging.info("coloc with GRD using a %f ° radius", radius_coloc)
    # for 17.5km tiles res 0.2°, for 5km->0.05°
    if matching_l2wind_nc_file is not None:
        dsgrd = xr.open_dataset(matching_l2wind_nc_file, engine="h5netcdf").load()
        owilon = dsgrd["owiLon"].values.ravel()
        owilat = dsgrd["owiLat"].values.ravel()
        maskowi = np.isfinite(owilon) & np.isfinite(owilat)
        logging.info(
            "nb NaN in the 1km² owi grid: %i/%i",
            len(owilat) - maskowi.sum(),
            len(owilat),
        )
        points = np.c_[owilon[maskowi], owilat[maskowi]]
        tree = spatial.KDTree(points)
        for l1bgridname in l1bgrids:
            logging.info("%s ==========================\n", l1bgridname)
            l1bgrid = l1bgrids[l1bgridname]
            gridsarXSP = {
                d: k
                for d, k in l1bgrid.sizes.items()
                # if d in ["burst", "tile_sample", "tile_line"]
                if d in ["tile_sample", "tile_line"]
            }
            list_grd_infos_matchingtiles = []
            all_tile_cases = [i for i in xndindex(gridsarXSP)]
            for x in tqdm(range(len(all_tile_cases))):
                i = all_tile_cases[x]
                lonsar = l1bgrid["longitude"][i].values
                latsar = l1bgrid["latitude"][i].values
                if np.isfinite(lonsar) and np.isfinite(latsar):
                    # sensing_time = l1bgrid["sensing_time"][i].values
                    # heading = l1bgrid["ground_heading"][i].values
                    # logging.debug("heading %s", heading)
                    # rotate = 90 + heading  # deg clockwise wrt North
                    # logging.debug("rotate:%s", rotate)
                    # for each XSP tile find the N closest points in the cyclobs 1km res GRD product
                    condensated_grd_one_tile = grd_core_tile_coloc(
                        lonsar,
                        latsar,
                        treegrd=tree,
                        radius_coloc=radius_coloc,
                        dsgrd=dsgrd,
                        indexes_xsp=i,
                    )
                    if len(condensated_grd_one_tile.data_vars) == 0:
                        cpt["alone_%s_tile" % l1bgridname] += 1
                    else:
                        cpt["colocated_%s_tile" % l1bgridname] += 1
                        list_grd_infos_matchingtiles.append(condensated_grd_one_tile)
            ds_colocation_grd = xr.merge(list_grd_infos_matchingtiles)
            ds_l1c = xr.merge([l1bgrid, ds_colocation_grd])
            ds_l1c.attrs["Ifremer-wind-product-from-GRD"] = matching_l2wind_nc_file
            # ds_l1c.attrs['radius_coloc_in_degrees'] = '%s'%radius_coloc
            l1cgrids[l1bgridname] = ds_l1c
        else:
            logging.info(
                "no corresponding Ifremer Sentinel-1 GRD wind product for %s",
                l1bgrid.attrs["name"],
            )
    logging.info("counters: %s", cpt)
    return l1cgrids, cpt
