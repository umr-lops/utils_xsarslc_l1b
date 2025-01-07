#!/usr/bin/env python
"""
A Grouazel
Stack WV L1C data per month for stats and training purpose
"""
import argparse
import datetime
import glob
import logging
import os
import shutil
import time

import numpy as np
import xarray as xr
from dateutil import relativedelta
from tqdm import tqdm

from slcl1butils.get_config import get_conf
from slcl1butils.utils import get_memory_usage

conf = get_conf()
GROUP_NAME_SAR = "sar_sentinel1_wv_l1b_xsp_intraburst"
VARS_WW3_GRID = [
    "MAPSTA",
    "dpt",
    "ucur",
    "vcur",
    "uwnd",
    "vwnd",
    "ice",
    "ibg",
    "ic1",
    "ic5",
    "hs",
    "lm",
    "t02",
    "t0m1",
    "t01",
    "fp",
    "dir",
    "spr",
    "dp",
    "phs0",
    "phs1",
    "phs2",
    "phs3",
    "phs4",
    "phs5",
    "ptp0",
    "ptp1",
    "ptp2",
    "ptp3",
    "ptp4",
    "ptp5",
    "plp0",
    "plp1",
    "plp2",
    "plp3",
    "plp4",
    "plp5",
    "pdir0",
    "pdir1",
    "pdir2",
    "pdir3",
    "pdir4",
    "pdir5",
    "pspr0",
    "pspr1",
    "pspr2",
    "pspr3",
    "pspr4",
    "pspr5",
    "pws0",
    "pws1",
    "pws2",
    "pws3",
    "pws4",
    "pws5",
    "pdp0",
    "pdp1",
    "pdp2",
    "pdp3",
    "pdp4",
    "pdp5",
    "tws",
    "uust",
    "vust",
    "cha",
    "cge",
    "faw",
    "utaw",
    "vtaw",
    "utwa",
    "vtwa",
    "wcc",
    "utwo",
    "vtwo",
    "foc",
    "utus",
    "vtus",
    "uuss",
    "vuss",
    "uabr",
    "vabr",
    "uubr",
    "vubr",
    "mssu",
    "mssc",
    "mssd",
]
VARS_ECMF = ["U10", "V10"]
VARS_SAR_L1B = [
    "incidence",
    "ground_heading",
    "sensing_time",
    "sigma0",
    "nesz",
    "bright_targets_histogram",
    "sigma0_filt",
    "nesz_filt",
    "normalized_variance_filt",
    "doppler_centroid",
    "tau",
    "azimuth_cutoff",
    "azimuth_cutoff_error",
    "corner_longitude",
    "corner_latitude",
    "corner_line",
    "corner_sample",
    "land_flag",
    "burst_corner_longitude",
    "burst_corner_latitude",
    "cwave_params",
    "macs_Re",
    "macs_Im",
    "efth",
    "ww3EFTHraw",
    "xspectra_2tau_Re",
    "xspectra_2tau_Im",
    "wvmode",
]
VARS_TO_KEEP = VARS_SAR_L1B + VARS_WW3_GRID + VARS_ECMF
k_rg_ref_semi = xr.DataArray(
    np.array(
        [
            0.0,
            0.00157081,
            0.00314162,
            0.00471244,
            0.00628325,
            0.00785406,
            0.00942487,
            0.01099568,
            0.01256649,
            0.01413731,
            0.01570812,
            0.01727893,
            0.01884974,
            0.02042055,
            0.02199136,
            0.02356218,
            0.02513299,
            0.0267038,
            0.02827461,
            0.02984542,
            0.03141623,
            0.03298705,
            0.03455786,
            0.03612867,
            0.03769948,
            0.03927029,
            0.04084111,
            0.04241192,
            0.04398273,
            0.04555354,
            0.04712435,
            0.04869517,
            0.05026598,
            0.05183679,
            0.0534076,
            0.05497841,
            0.05654923,
            0.05812003,
            0.05969085,
            0.06126166,
            0.06283247,
            0.06440328,
            0.06597409,
            0.06754491,
            0.06911572,
            0.07068653,
            0.07225734,
            0.07382815,
            0.07539897,
            0.07696978,
            0.07854059,
            0.0801114,
            0.08168221,
            0.08325303,
            0.08482383,
            0.08639465,
            0.08796546,
            0.08953627,
            0.09110709,
            0.09267789,
            0.0942487,
            0.09581952,
            0.09739033,
            0.09896114,
            0.10053195,
            0.10210276,
            0.10367358,
            0.10524439,
            0.1068152,
            0.10838601,
            0.10995682,
            0.11152764,
            0.11309845,
            0.11466926,
            0.11624007,
            0.11781088,
            0.1193817,
            0.12095251,
            0.12252332,
            0.12409413,
            0.12566493,
            0.12723576,
            0.12880656,
            0.13037738,
            0.13194819,
            0.133519,
            0.13508981,
            0.13666062,
            0.13823144,
            0.13980225,
            0.14137305,
            0.14294387,
            0.14451468,
            0.1460855,
            0.1476563,
            0.14922711,
            0.15079793,
            0.15236874,
            0.15393956,
            0.15551037,
            0.15708117,
            0.158652,
        ]
    ),
    coords={"freq_sample": np.arange(102)},
    dims=["freq_sample"],
)
# new version symmetrized
k_rg_ref_symm = xr.DataArray(
    np.array(
        [
            -0.16016316,
            -0.15859294,
            -0.15702271,
            -0.15545249,
            -0.15388227,
            -0.15231204,
            -0.1507418,
            -0.14917158,
            -0.14760135,
            -0.14603113,
            -0.1444609,
            -0.14289068,
            -0.14132045,
            -0.13975021,
            -0.13817999,
            -0.13660976,
            -0.13503954,
            -0.13346931,
            -0.13189909,
            -0.13032885,
            -0.12875862,
            -0.1271884,
            -0.12561817,
            -0.12404795,
            -0.12247772,
            -0.12090749,
            -0.11933727,
            -0.11776704,
            -0.11619681,
            -0.11462659,
            -0.11305635,
            -0.11148613,
            -0.1099159,
            -0.10834567,
            -0.10677545,
            -0.10520522,
            -0.10363499,
            -0.10206477,
            -0.10049454,
            -0.09892431,
            -0.09735408,
            -0.09578386,
            -0.09421363,
            -0.0926434,
            -0.09107318,
            -0.08950295,
            -0.08793272,
            -0.0863625,
            -0.08479226,
            -0.08322204,
            -0.08165181,
            -0.08008158,
            -0.07851136,
            -0.07694113,
            -0.0753709,
            -0.07380068,
            -0.07223045,
            -0.07066023,
            -0.06908999,
            -0.06751977,
            -0.06594954,
            -0.06437931,
            -0.06280909,
            -0.06123886,
            -0.05966863,
            -0.05809841,
            -0.05652818,
            -0.05495795,
            -0.05338772,
            -0.0518175,
            -0.05024727,
            -0.04867704,
            -0.04710681,
            -0.04553659,
            -0.04396636,
            -0.04239613,
            -0.04082591,
            -0.03925568,
            -0.03768545,
            -0.03611523,
            -0.034545,
            -0.03297477,
            -0.03140454,
            -0.02983432,
            -0.02826409,
            -0.02669386,
            -0.02512364,
            -0.02355341,
            -0.02198318,
            -0.02041295,
            -0.01884273,
            -0.0172725,
            -0.01570227,
            -0.01413204,
            -0.01256182,
            -0.01099159,
            -0.00942136,
            -0.00785114,
            -0.00628091,
            -0.00471068,
            -0.00314045,
            -0.00157023,
            0.0,
            0.00157023,
            0.00314045,
            0.00471068,
            0.00628091,
            0.00785114,
            0.00942136,
            0.01099159,
            0.01256182,
            0.01413204,
            0.01570227,
            0.0172725,
            0.01884273,
            0.02041295,
            0.02198318,
            0.02355341,
            0.02512364,
            0.02669386,
            0.02826409,
            0.02983432,
            0.03140454,
            0.03297477,
            0.034545,
            0.03611523,
            0.03768545,
            0.03925568,
            0.04082591,
            0.04239613,
            0.04396636,
            0.04553659,
            0.04710681,
            0.04867704,
            0.05024727,
            0.0518175,
            0.05338772,
            0.05495795,
            0.05652818,
            0.05809841,
            0.05966863,
            0.06123886,
            0.06280909,
            0.06437931,
            0.06594954,
            0.06751977,
            0.06908999,
            0.07066023,
            0.07223045,
            0.07380068,
            0.0753709,
            0.07694113,
            0.07851136,
            0.08008158,
            0.08165181,
            0.08322204,
            0.08479226,
            0.0863625,
            0.08793272,
            0.08950295,
            0.09107318,
            0.0926434,
            0.09421363,
            0.09578386,
            0.09735408,
            0.09892431,
            0.10049454,
            0.10206477,
            0.10363499,
            0.10520522,
            0.10677545,
            0.10834567,
            0.1099159,
            0.11148613,
            0.11305635,
            0.11462659,
            0.11619681,
            0.11776704,
            0.11933727,
            0.12090749,
            0.12247772,
            0.12404795,
            0.12561817,
            0.1271884,
            0.12875862,
            0.13032885,
            0.13189909,
            0.13346931,
            0.13503954,
            0.13660976,
            0.13817999,
            0.13975021,
            0.14132045,
            0.14289068,
            0.1444609,
            0.14603113,
            0.14760135,
            0.14917158,
            0.1507418,
            0.15231204,
            0.15388227,
            0.15545249,
            0.15702271,
            0.15859294,
        ]
    ),
    coords={"k_rg": np.arange(204)},
    dims=["k_rg"],
)


def get_all_l1c_for_a_month(startdate, stopdate, sarunit, l1c_dir, polarisation="V"):
    """

    Args:
        startdate: datetime
        stopdate: datetime
        sarunit: str 'S1A' or 'S1B'
        l1c_dir: str directory
        polarisation: str 'V' or 'H'

    Returns:

    """
    pat = os.path.join(l1c_dir, sarunit + "_WV_XSP__1SS" + polarisation + "*.SAFE")
    logging.info("pattern : %s", pat)
    lst_safe = glob.glob(pat, recursive=False)
    logging.info("nb SAFE L1C total: %s", len(lst_safe))
    safe_of_the_month = []
    for ss in lst_safe:
        tmpdate = datetime.datetime.strptime(
            os.path.basename(ss).split("_")[5], "%Y%m%dT%H%M%S"
        )
        if tmpdate >= startdate and tmpdate < stopdate:
            safe_of_the_month.append(ss)
    logging.info("nb L1C SAFE for the month: %s", len(safe_of_the_month))
    return safe_of_the_month


def preprrocess(
    filee,
    k_rg_ref,
    k_az_ref,
    typee,
    nc_number,
):
    """

    Parameters
    ----------
    filee str full path
    k_rg_ref xr.DataArray
    k_az_ref xr.DataArray
    typee str intra or inter
    nc_number int


    Returns
    -------
    dsu xr.Dataset

    """
    dsu = xr.open_dataset(filee, group=typee + "burst", engine="netcdf4", cache=False)
    # dsu['subswath'] = xr.DataArray(int(os.path.basename(filee).split('-')[1].replace('iw','')))
    tmpsubswath = xr.DataArray(
        int(os.path.basename(filee).split("-")[1].replace("wv", ""))
    )
    dsu = dsu.drop_vars(["sample"])

    # dsu = dsu.assign_coords({'wvmode': tmpsubswath})
    dsu["wvmode"] = xr.DataArray(tmpsubswath)
    try:
        tmpdate = datetime.datetime.strptime(
            dsu.attrs["start_date"], "%Y-%m-%d %H:%M:%S.%f"
        )
    except ValueError:
        tmpdate = datetime.datetime.strptime(
            dsu.attrs["start_date"], "%Y-%m-%d %H:%M:%S"
        )
    # dsu['sardate'] = xr.DataArray(tmpdate)
    # if (
    #     "freq_sample" in dsu.coords
    # ):  # when WV L1C are not symmetrized (no WW3 spectra addition) ie old version of the L1B->L1C processor
    #     dsu = dsu.isel(
    #         freq_sample=slice(0, indkrg_up), freq_line=slice(indkaz_bo, indkaz_up)
    #     )
    # else:
    ind0_rg = get_k0(dsu["k_rg"])
    ind45_rg = get_k45m(dsu["k_rg"])
    distk0_to_k45 = ind45_rg - ind0_rg
    borninf = ind0_rg - distk0_to_k45

    ind0_az = get_k0(dsu["k_az"])
    ind45_az = get_k45m(dsu["k_az"])
    distk0_to_k45_az = ind45_az - ind0_az
    borninf_az = ind0_az - distk0_to_k45_az

    dsu = dsu.isel(k_rg=slice(borninf, ind45_rg), k_az=slice(borninf_az, ind45_az))
    logging.info("cropped cross spectra : size / %i x %i", dsu.k_rg.size, dsu.k_az.size)
    dsu = dsu.assign_coords({"k_rg": k_rg_ref.values, "k_az": k_az_ref.values})
    dsu = dsu.assign_coords({"nc_number": nc_number})
    dsu = dsu.assign_coords(
        {"sardate": np.datetime64(tmpdate, "s").astype("datetime64[ns]")}
    )
    if "xspectra_0tau_Re" in dsu:
        for tautau in range(3):
            dsu["xspectra_%stau" % tautau] = (
                dsu["xspectra_%stau_Re" % tautau]
                + 1j * dsu["xspectra_%stau_Im" % tautau]
            )
            dsu = dsu.drop(["xspectra_%stau_Re" % tautau, "xspectra_%stau_Im" % tautau])
    dsu = dsu.drop("line")
    return dsu


def get_reference_wavenumbers(ds, indkaz_bo=None, indkaz_up=None):
    """

    Parameters
    ----------
    ds xr.Dataset
    indkaz_bo int
    indkaz_up int

    Returns
    -------

    """
    k_az = ds["k_az"]
    if "freq_line" in k_az.coords:  # old version L1C
        k_rg_ref = k_rg_ref_semi
        k_az_ref = k_az.isel(freq_line=slice(indkaz_bo, indkaz_up))
        k_az_ref = k_az_ref.assign_coords({"freq_line": np.arange(len(k_az_ref))})
        k_az_ref = xr.DataArray(
            k_az_ref.values,
            coords={"freq_line": np.arange(len(k_az_ref))},
            dims=["freq_line"],
        )
    else:  # new version L1C WV symmetrized with WW3 sp
        ind0ref_rg = get_k0(k_rg_ref_symm)
        ind45ref_rg = get_k45m(k_rg_ref_symm)
        distk0_to_k45 = ind45ref_rg - ind0ref_rg
        borninf = ind0ref_rg - distk0_to_k45
        k_rg_ref = k_rg_ref_symm.isel({"k_rg": slice(borninf, ind45ref_rg)})

        ind0ref_az = get_k0(k_az)
        ind45ref_az = get_k45m(k_az)
        distk0_to_k45_az = ind45ref_az - ind0ref_az
        borninf = ind0ref_az - distk0_to_k45_az
        k_az_ref = k_az.isel({"k_az": slice(borninf, ind45ref_az)})
        # k_az_ref = k_az.isel(k_az=slice(indkaz_bo, indkaz_up))
        k_az_ref = k_az_ref.assign_coords({"k_az": np.arange(len(k_az_ref))})
        k_az_ref = xr.DataArray(
            k_az_ref.values,
            coords={"k_az": np.arange(len(k_az_ref))},
            dims=["k_az"],
        )

    return k_rg_ref, k_az_ref


def get_index_wavenumbers(ds):
    """

    Args:
        ds: xarray.Dataset

    Returns:

    """
    val_k_rg = 0.16
    indkrg_up = np.argmin(abs(ds["k_rg"].data - val_k_rg))
    indkrg_bo = np.argmin(abs(ds["k_rg"].data + val_k_rg))
    val_k_az = 0.14
    indkaz_up = np.argmin(abs(ds["k_az"].data - val_k_az))
    indkaz_bo = np.argmin(abs(ds["k_az"].data + val_k_az))
    return indkaz_bo, indkaz_up, indkrg_up, indkrg_bo


def get_k0(k):
    return np.where(k.data == 0.0)[0][0]


def get_k45m(k):
    return np.argmin(abs(k.data - 0.1396))


def stack_wv_l1c_per_month(list_SAFEs, dev=False, keep_xspectrum=False):
    """

    Args:
        list_SAFEs: lst of str paths
        dev: bool True -> reduced number of files to process
        keep_xspectrum: bool

    Returns:

    """
    if dev:
        logging.info("dev mode -> reduce number of safe to 10")
        list_SAFEs = list_SAFEs[0:10]
    k_rg_ref = None
    k_az_ref = None
    cpt_nc = 0
    tmp = []
    vL1C = "unknown"
    vL1B = "unknown"
    # ds = None
    dtnew = xr.DataTree(name="root", data=None)

    if len(list_SAFEs) > 0:
        # for safe_xsp in list_SAFEs:
        logging.info("loop over the SAFE : %i", len(list_SAFEs))
        for ssi in tqdm(range(len(list_SAFEs))):
            safe_xsp = list_SAFEs[ssi]
            lst_nc_l1c = glob.glob(os.path.join(safe_xsp, "*nc"))
            logging.debug("nb nc : %s", len(lst_nc_l1c))
            if len(lst_nc_l1c) == 0:
                logging.warning("empty safe: %s", safe_xsp)
            else:
                if ssi == 0:
                    dt = xr.open_datatree(lst_nc_l1c[0])
                    vL1C = dt.attrs["L1C_product_version"]
                    vL1B = dt.attrs["L1B_product_version"]
                if (
                    keep_xspectrum is True and k_rg_ref is None
                ):  # code below executed only once thanks to this line
                    # dsfirst = xr.open_dataset(lst_nc_l1c[0], group='intraburst', engine='netcdf4', cache=False)
                    dt = xr.open_datatree(lst_nc_l1c[0])
                    dsfirst = dt["intraburst"].to_dataset()
                    # indkaz_bo, indkaz_up, indkrg_up, indkrg_bo = get_index_wavenumbers(
                    #     dsfirst
                    # )
                    k_rg_ref, k_az_ref = get_reference_wavenumbers(dsfirst)
                for ff in lst_nc_l1c:
                    if keep_xspectrum:
                        tmpds = preprrocess(
                            ff,
                            k_rg_ref,
                            k_az_ref,
                            typee="intra",
                            nc_number=cpt_nc,
                        )
                    else:
                        tmpds = xr.open_dataset(ff, group="intraburst")
                        vars_to_drop = [
                            "xspectra_2tau_Re",
                            "xspectra_2tau_Im",
                            "efth",
                            "k_rg",
                            "k_az",
                            "nc_number",
                        ]  # for foundation model exercise
                        vars_to_drop_consolidated = []
                        for vv in tmpds:
                            if vv not in VARS_TO_KEEP and vv not in vars_to_drop:
                                vars_to_drop_consolidated.append(vv)

                        tmpds = tmpds.drop_vars(
                            vars_to_drop_consolidated
                        )  # to avoid all the WW3 grid variables that are not mandatory for Hs regression or spectrum regression
                        tmpds = tmpds.drop_dims(["k_rg", "k_az"])
                        tmpsubswath = xr.DataArray(
                            int(os.path.basename(ff).split("-")[1].replace("wv", ""))
                        )  # wv1 -> , wv2 -> 2
                        # tmpds = tmpds.assign_coords({'wvmode': tmpsubswath})
                        tmpds["wvmode"] = xr.DataArray(tmpsubswath)
                        tmpds["wvmode"].attrs = {
                            "description": "sub-swath of the Sentinel-1 SAR WV: wv1 or wv2"
                        }
                        tmpdate = datetime.datetime.strptime(
                            tmpds.attrs["start_date"], "%Y-%m-%d %H:%M:%S.%f"
                        )
                        # tmpds['sardate'] = xr.DataArray(tmpdate)
                        tmpds = tmpds.drop(
                            [vv for vv in tmpds if "xspectra" in vv] + ["sample"]
                        )
                        if "freq_line" in tmpds:
                            tmpds = tmpds.drop_dims(["freq_line", "freq_sample"])
                        tmpds.drop_vars(["corner_longitude", "corner_latitude"])
                        tmpds = tmpds.drop_dims(["c_sample", "c_line"])
                        tmpds = tmpds.drop("line")
                        # tmpds = tmpds.assign_coords({"nc_number": cpt_nc})
                        tmpds = tmpds.assign_coords({"sardate": tmpdate})
                        # tmpds = tmpds.expand_dims('nc_number')
                    cpt_nc += 1
                    tmp.append(tmpds.expand_dims("sardate"))
        logging.info("nb dataset to combine: %s", cpt_nc)

        # ds = xr.concat(tmp, dim="sardate",coords='minimal') # ne fonctionne pas -> conflict in coords
        ds = xr.combine_by_coords(tmp, combine_attrs="drop_conflicts")
        #
        # la ligne du dessous marche mais cest long je tente autre chsoe
        # ds = xr.combine_by_coords(
        #     [tt.expand_dims("sardate") for tt in tmp],  # .expand_dims('wvmode')
        #     combine_attrs="drop_conflicts",
        # )
        ds_ecmwf = xr.Dataset()
        ds_ww3 = xr.Dataset()
        ds_sar_l1b = xr.Dataset()
        for vv in VARS_ECMF:
            if vv in ds:
                ds_ecmwf[vv] = ds[vv]
        if "U10" in ds_ecmwf:
            ds_ecmwf["U10"].attrs[
                "description"
            ] = "zonal component of wind speed from ECMWF hourly 0.1deg resolution forecasts"
            ds_ecmwf["V10"].attrs[
                "description"
            ] = "meridional component of wind speed from ECMWF hourly 0.1deg resolution forecasts"
            ds_ecmwf["longitude"].attrs[
                "description"
            ] = "longitude of the center of level-1B tile from Sentinel-1 WV imagette"
            ds_ecmwf["latitude"].attrs[
                "description"
            ] = "latitude of the center of level-1B tile from Sentinel-1 WV imagette"
        for vv in VARS_WW3_GRID:
            if vv in ds:
                ds_ww3[vv] = ds[vv]
                ds_ww3[vv].attrs[
                    "source"
                ] = "WW3 numerical forecast 3-hourly 0.5 deg resolution"
        if "longitude" in ds_ww3:
            ds_ww3["longitude"].attrs[
                "description"
            ] = "longitude of the center of level-1B tile from Sentinel-1 WV imagette"
            ds_ww3["latitude"].attrs[
                "description"
            ] = "latitude of the center of level-1B tile from Sentinel-1 WV imagette"
        for vv in VARS_SAR_L1B:
            if vv in ds:
                ds_sar_l1b[vv] = ds[vv]
                ds_sar_l1b[vv].attrs[
                    "source"
                ] = "SAR Ifremer level-1B processor for WV SLC"
        ds_sar_l1b["longitude"].attrs[
            "description"
        ] = "longitude of the center of level-1B tile from Sentinel-1 WV imagette"
        ds_sar_l1b["latitude"].attrs[
            "description"
        ] = "latitude of the center of level-1B tile from Sentinel-1 WV imagette"
        _ = xr.DataTree(name="ecmwf_0100_1h", parent=dtnew, data=ds_ecmwf)
        _ = xr.DataTree(name="ww3_global_yearly_3h", parent=dtnew, data=ds_ww3)
        _ = xr.DataTree(name=GROUP_NAME_SAR, parent=dtnew, data=ds_sar_l1b)
        # dt['ecmwf_0100_1h'] = ds_ecmwf
        # dt['ww3_global_yearly_3h'] = ds_ww3
        # dt['sentinel1_wv_l1b'] = ds_sar_l1b
    return dtnew, vL1C, vL1B


def save_to_zarr(stacked_ds, outputfile, L1C_version, L1B_version):
    """

    Args:
        stacked_ds: xr.Dataset
        outputfile: str
        L1C_version: str
        L1B_version: str

    Returns:

    """
    if not os.path.dirname(outputfile):
        os.makedirs(os.path.dirname(outputfile), 0o0775)
    stacked_ds.attrs["L1C_version"] = L1C_version
    stacked_ds.attrs["L1B_version"] = L1B_version
    stacked_ds.attrs["creation_script"] = os.path.basename(__file__)
    logging.info("start saving to disk")
    stacked_ds.to_zarr(outputfile, mode="w")
    logging.info("successfully wrote stacked L1C %s", outputfile)
    return outputfile


def save_to_netcdf(stackdt, outputfile, L1C_version, L1B_version):
    """

    Args:
        stacked_ds: xr.DataTree
        outputfile: str
        L1C_version: str
        L1B_version: str

    Returns:

    """
    if not os.path.dirname(outputfile):
        os.makedirs(os.path.dirname(outputfile), 0o0775)
    stackdt.attrs["L1C_version"] = L1C_version
    stackdt.attrs["L1B_version"] = L1B_version
    stackdt.attrs["creation_script"] = os.path.basename(__file__)
    logging.info("start saving to disk")
    # stackdt.to_netcdf(outputfile, engine='h5netcdf',mode="w")
    stackdt.to_netcdf(outputfile, engine="netcdf4", mode="w")
    logging.info("successfully wrote stacked L1C %s", outputfile)
    return outputfile


def main():
    """

    Returns
    -------

    """
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="L1C-month-STACK")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite the existing outputs [default=False]",
        required=False,
    )
    parser.add_argument("--month", required=True, help="YYYYMM (eg 202103)")
    parser.add_argument(
        "--sar", required=True, choices=["S1A", "S1B"], help="S1A or S1B or ..."
    )
    parser.add_argument(
        "--pol",
        required=False,
        choices=["V", "H"],
        help="polarization V or H [default=V]",
    )
    parser.add_argument(
        "--inputdir", required=True, help="directory where L1C files are stored"
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where L1C stacked files will be stored",
    )
    parser.add_argument(
        "--finalarchive",
        help="the files are produce in args.outputdir and then moved in args.finalarchive (using mv)",
        required=True,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="dev mode stops the computation early",
    )
    parser.add_argument(
        "--keepxspec",
        action="store_true",
        default=False,
        help="keep X-spectrum in the stacked output [default=False]",
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
    logging.info("outputdir will be: %s", args.outputdir)
    logging.info("sar: %s", args.sar)
    logging.info("inputdir: %s", args.inputdir)
    logging.info("outputdir: %s", args.outputdir)
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    if not os.path.exists(args.finalarchive):
        os.makedirs(args.finalarchive)
    startdate = datetime.datetime.strptime(args.month, "%Y%m").replace(day=1)
    stopdate = startdate + relativedelta.relativedelta(months=1)
    logging.info("startdate %s -> %s stopdate", startdate, stopdate)
    # outputfile = os.path.join(
    #     args.outputdir, args.sar + "_WV_L1C_monthly_%s_%s.zarr" % (args.month, args.pol)
    # )
    outputfile = os.path.join(
        args.outputdir, args.sar + "_WV_L1C_monthly_%s_%s.nc" % (args.month, args.pol)
    )
    if os.path.exists(outputfile) and args.overwrite is False:
        logging.info("%s already exists", outputfile)
    else:
        lst_safes_month = get_all_l1c_for_a_month(
            startdate,
            stopdate,
            sarunit=args.sar,
            l1c_dir=args.inputdir,
            polarisation=args.pol,
        )
        stackdt, vL1C, vL1B = stack_wv_l1c_per_month(
            list_SAFEs=lst_safes_month, dev=args.dev, keep_xspectrum=args.keepxspec
        )
        if stackdt:
            # save_to_zarr(
            #     stacked_ds=stackds,
            #     outputfile=outputfile,
            #     L1C_version=vL1C,
            #     L1B_version=vL1B,
            # )
            outputfile = save_to_netcdf(
                stackdt=stackdt,
                outputfile=outputfile,
                L1C_version=vL1C,
                L1B_version=vL1B,
            )
        else:
            logging.info("no data available")
    if os.path.exists(outputfile):
        archive_path = outputfile.replace(args.outputdir, args.finalarchive)
        if outputfile != archive_path:
            logging.info("move file %s to %s", outputfile, archive_path)
            shutil.move(outputfile, archive_path)

    logging.info("peak memory usage: %s Mbytes", get_memory_usage())
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
