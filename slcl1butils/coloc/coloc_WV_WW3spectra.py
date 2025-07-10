import logging
import os

import numpy as np
import xarray as xr

from slcl1butils.compute.compute_from_l1b import get_start_date_from_attrs
from slcl1butils.legacy_ocean.ocean.xPolarSpectrum import find_closest_ww3

# from ocean.xspectrum import from_ww3
# from ocean.xPolarSpectrum import find_closest_ww3
from slcl1butils.legacy_ocean.ocean.xspectrum import from_ww3
from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum


def resampleWW3spectra_on_SAR_cartesian_grid(dsar) -> (xr.Dataset, bool, bool):
    """

    Args:
        dsar: xarray.core.Dataset WV Level-1B Ifremer dataset for one tile of one imagette

    Returns:
        dsar: xr.Dataset
        flag_ww3spectra_added: bool
        flag_ww3spectra_found: bool
    """
    # basewv = os.path.basename(l1bwv)
    # datedt = datetime.datetime.strptime(basewv.split('-')[4],'%Y%m%dt%H%M%S')
    xsNtau = {}
    for tautau in ["2tau", "1tau"]:
        # temporarily change coord for xpsectra and remove them from the ds
        if "xspectra_%s_Re" % tautau in dsar:
            xsntau = (
                dsar["xspectra_%s_Re" % tautau] + 1j * dsar["xspectra_%s_Im" % tautau]
            )
            xsntau = xsntau.assign_coords(
                {
                    "k_rg": xsntau["k_rg"].mean(
                        dim=set(xsntau["k_rg"].dims) - set(["freq_sample"]),
                        keep_attrs=True,
                    )
                }
            ).swap_dims({"freq_sample": "k_rg", "freq_line": "k_az"})
            if xsntau[tautau].size == 1:
                xsntau = symmetrize_xspectrum(xsntau).squeeze(dim=tautau)
            else:
                xsntau = symmetrize_xspectrum(xsntau)
            xsntau.attrs = dsar["xspectra_%s_Re" % tautau].attrs
            xsNtau[tautau] = xsntau
            # replace the half spectrum by a single variable, to save complexe values only possibility is zarr
            dsar = dsar.drop_vars(
                ["xspectra_%s_Re" % tautau, "xspectra_%s_Im" % tautau]
            )
    dsar = dsar.drop_dims(
        ["freq_sample", "freq_line"]
    )  # this line remove all the other variable depending on freq_sample,freq_line e.g. xspectra_1tau_Im
    for tautau in xsNtau:
        # put back the cross spectra
        dsar["xspectra_%s" % tautau] = xsNtau[tautau]
    xs2tau = xsNtau["2tau"]
    start_date_dt = get_start_date_from_attrs(ds=dsar)
    # unit = basewv.split('-')[0]
    # unit_long = 'sentinel-1'+unit[-1]
    flag_ww3spectra_added = False
    flag_ww3spectra_found = False
    pathww3sp = get_ww3RAWspectra_path_from_date(datedt=start_date_dt)
    if pathww3sp:
        if os.path.exists(pathww3sp):
            flag_ww3spectra_found = True
            dk_az = np.diff(xs2tau["k_az"])
            dk_rg = np.diff(xs2tau["k_rg"])
            dk = (dk_rg.mean(), dk_az.mean())
            xs2tau["k_rg"].attrs["dkx"] = dk[0]  # temporarily add dkx attrs
            xs2tau["k_az"].attrs["dky"] = dk[1]  # temporarily add dky attrs
            kmax = (
                np.abs(xs2tau["k_rg"]).max().item(),
                np.abs(xs2tau["k_az"]).max().item(),
            )  # FN reviewed
            lonwv = dsar["longitude"].values
            latwv = dsar["latitude"].values
            heading = dsar["ground_heading"].values
            logging.debug("heading %s", heading)
            logging.debug("timewv :%s", start_date_dt)
            rotate = 90 + heading  # deg clockwise wrt North
            logging.debug("rotate:%s", rotate)
            # add the raw  EFTH(f,dir) spectra from WW3
            dsww3raw = xr.open_dataset(pathww3sp).load()
            # add the interpolated cartesian EFTH(kx,ky) spectra from WW3
            ds_ww3_cartesian = from_ww3(
                pathww3sp,
                dk=dk,
                kmax=kmax,
                strict="dk",
                rotate=rotate,
                clockwise_to_trigo=True,
                lon=lonwv,
                lat=latwv,
                time=start_date_dt,
            )  # TODO use sensingTime
            ds_ww3_cartesian.attrs["source"] = "ww3"
            indiceww3spectra = find_closest_ww3(
                ww3_path=pathww3sp, lon=lonwv, lat=latwv, time=start_date_dt
            )
            rawspww3 = dsww3raw["efth"].isel(time=indiceww3spectra).rename("ww3EFTHraw")
            rawspww3.attrs["description"] = "raw EFTH(f,dir) spectra"
            ds_ww3_cartesian = ds_ww3_cartesian.swap_dims(
                {"kx": "k_rg", "ky": "k_az"}
            ).T
            ds_ww3_cartesian = ds_ww3_cartesian.rename({"time": "time_ww3"})
            rawspww3 = rawspww3.rename({"time": "time_ww3"})
            dsar = xr.merge([dsar, ds_ww3_cartesian, rawspww3])
            flag_ww3spectra_added = True
    return dsar, flag_ww3spectra_added, flag_ww3spectra_found


def get_ww3RAWspectra_path_from_date(datedt) -> str:
    """


    Args:
        datedt: datetime.datetime

    Returns:

    """
    from slcl1butils.get_config import get_conf

    conf = get_conf()
    DIR_S1_WW3_RAWSPECTRA = conf["DIR_S1_WW3_RAWSPECTRA"]
    baseww3file = "MARC_WW3-GLOB-30M_" + datedt.strftime("%Y%m%d") + "_trck.nc"
    finalpath = None
    # yearmonth = datedt.strftime('%Y%m')
    pat = os.path.join(DIR_S1_WW3_RAWSPECTRA, baseww3file)
    logging.debug("pat: %s", pat)
    if os.path.exists(pat):
        finalpath = pat

    return finalpath
