import os
import pdb
import glob
import xarray as xr
import datetime
import logging
import numpy as np
from slcl1butils.raster_readers import resource_strftime
from slcl1butils.utils import xndindex

# from ocean.xspectrum import from_ww3
# from ocean.xPolarSpectrum import find_closest_ww3
from slcl1butils.legacy_ocean.ocean.xspectrum import from_ww3
from slcl1butils.legacy_ocean.ocean.xPolarSpectrum import find_closest_ww3
from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum


def resampleWW3spectra_on_TOPS_SAR_cartesian_grid(dsar, xspeckind):
    """

    Args:
        dsar: xarray.core.Dataset IW Level-1B IFREMER dataset
        xspeckind: str inter or intra
    Returns:

    """
    flag_ww3spectra_added = False
    flag_ww3spectra_found = False
    start_date_dt = datetime.datetime.strptime(
        dsar.attrs["start_date"], "%Y-%m-%d %H:%M:%S.%f"
    )
    pathww3sp = get_ww3HINDCAST_TRCKspectra_path_from_date(sar_date=start_date_dt)
    gridsar = {
        d: k
        for d, k in dsar.sizes.items()
        #if d in ["burst", "tile_sample", "tile_line"]
        if d in [ "tile_sample", "tile_line"]
    }
    if (xspeckind == "intra" and "xspectra_2tau_Re" in dsar) or (xspeckind == "inter" and "xspectra_Re" in dsar): # in a future version of L1B xspectra variable could be always present (even on land) but filled by NaN
        # symmetrize and combine Re+1j*Im for all the xspectra SAR
        if xspeckind == "intra":
            xsSAR = dsar["xspectra_2tau_Re"] + 1j * dsar["xspectra_2tau_Im"]
        elif xspeckind == "inter":
            xsSAR = dsar["xspectra_Re"] + 1j * dsar["xspectra_Im"]
        else:
            raise ValueError("%s" % xspeckind)
        xsSAR = xsSAR.assign_coords(
            {
                "k_rg": xsSAR["k_rg"].mean(
                    dim=set(xsSAR["k_rg"].dims) - set(["freq_sample"]),
                    keep_attrs=True,
                )
            }
        ).swap_dims({"freq_sample": "k_rg", "freq_line": "k_az"})
        if xspeckind == "intra":
            xsSAR = symmetrize_xspectrum(xsSAR).squeeze(dim="2tau")
            xsSAR.attrs = dsar["xspectra_2tau_Re"].attrs
        elif xspeckind == "inter":
            xsSAR = symmetrize_xspectrum(xsSAR)
            xsSAR.attrs = dsar["xspectra_Re"].attrs
        else:
            raise ValueError("%s" % xspeckind)
        # replace the half spectrum by a single variable, to save complexe values only possibility is zarr

        list_ww3_cart_sp = []
        list_ww3_efth_sp = []
        dk_az = np.diff(xsSAR["k_az"])
        dk_rg = np.diff(xsSAR["k_rg"])
        dk = (dk_rg.mean(), dk_az.mean())
        # xsSAR["k_rg"].attrs["dkx"] = dk[0]  # temporarily add dkx attrs
        # xsSAR["k_az"].attrs["dky"] = dk[1]  # temporarily add dky attrs
        kmax = (
            np.abs(xsSAR["k_rg"]).max().item(),
            np.abs(xsSAR["k_az"]).max().item(),
        )  # FN reviewed

        if pathww3sp:
            if os.path.exists(pathww3sp):
                flag_ww3spectra_found = True
                dsww3raw = xr.open_dataset(pathww3sp)
                for i in xndindex(gridsar):
                    lonsar = dsar["longitude"][i].values
                    latsar = dsar["latitude"][i].values
                    heading = dsar["ground_heading"][i].values
                    logging.debug("heading %s", heading)
                    logging.debug("timesar :%s", start_date_dt)
                    rotate = 90 + heading  # deg clockwise wrt North
                    logging.debug("rotate:%s", rotate)

                    # add the interpolated cartesian EFTH(kx,ky) spectra from WW3
                    ds_ww3_cartesian = from_ww3(
                        pathww3sp,
                        dk=dk,
                        kmax=kmax,
                        strict="dk",
                        rotate=rotate,
                        clockwise_to_trigo=True,
                        lon=lonsar,
                        lat=latsar,
                        time=start_date_dt,
                    )  # TODO use sensingTime
                    ds_ww3_cartesian.attrs["source"] = "ww3"
                    # TODO: check kx ky names to be same as the one from intra burst ds
                    indiceww3spectra = find_closest_ww3(
                        ww3_path=pathww3sp, lon=lonsar, lat=latsar, time=start_date_dt
                    )
                    # add the raw  EFTH(f,dir) spectra from WW3
                    rawspww3 = (
                        dsww3raw["efth"].isel(time=indiceww3spectra).rename("ww3EFTHraw")
                    )
                    rawspww3.attrs["description"] = "raw EFTH(f,dir) spectra"
                    ds_ww3_cartesian = ds_ww3_cartesian.swap_dims(
                        {"kx": "k_rg", "ky": "k_az"}
                    ).T
                    rawspww3 = rawspww3.assign_coords(i)
                    rawspww3 = rawspww3.expand_dims(["tile_line", "tile_sample"])
                    #rawspww3 = rawspww3.expand_dims(["burst", "tile_line", "tile_sample"])
                    ds_ww3_cartesian = ds_ww3_cartesian.assign_coords(i)
                    ds_ww3_cartesian = ds_ww3_cartesian.expand_dims(["tile_line", "tile_sample"])
                    #ds_ww3_cartesian = ds_ww3_cartesian.expand_dims(["burst", "tile_line", "tile_sample"])
                    list_ww3_cart_sp.append(ds_ww3_cartesian)
                    list_ww3_efth_sp.append(rawspww3)
                ds_ww3_cartesian_merged = xr.merge(list_ww3_cart_sp)
                ds_ww3_efth_merged = xr.merge(list_ww3_efth_sp)
                dsar = xr.merge([dsar, ds_ww3_cartesian_merged, ds_ww3_efth_merged])
                flag_ww3spectra_added = True
        if xspeckind == "intra":
            dsar = dsar.drop_vars(["xspectra_2tau_Re", "xspectra_2tau_Im"])
        elif xspeckind == "inter":
            dsar = dsar.drop_vars(["xspectra_Re", "xspectra_Im"])
        dsar = dsar.drop_dims(["freq_sample", "freq_line"])
        if xspeckind == "intra":
            dsar["xspectra_2tau"] = xsSAR
        elif xspeckind == "inter":
            dsar["xspectra"] = xsSAR
        else:
            raise ValueError("%s" % xspeckind)
    else:
        logging.info('there is no xspectra in subswath %s',dsar.attrs['short_name'].split(':')[2])
    return dsar, flag_ww3spectra_added, flag_ww3spectra_found


def get_ww3HINDCAST_TRCKspectra_path_from_date(sar_date):
    """

    Args:
        sar_date: (datetime)

    Returns:
        filename (str): path of the TRCK WW3 HINDCAST corresponding to the
    """
    from slcl1butils.get_config import get_conf

    filename = None
    conf = get_conf()
    pattern = conf["ww3hindcast_track_pattern"].replace(
        "LOPS_WW3-GLOB-30M_*_trck.nc",
        "LOPS_WW3-GLOB-30M_" + sar_date.strftime("%Y%m") + "_trck.nc",
    )
    datefound, pattern2 = resource_strftime(pattern, step=0.5, date=sar_date)
    filenames = glob.glob(pattern2)

    if len(filenames) > 0:
        filename = filenames[0]
    return filename
