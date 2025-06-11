import datetime
import glob
import logging
import os

import numpy as np
import xarray as xr

from slcl1butils.legacy_ocean.ocean.xPolarSpectrum import haversine

# from ocean.xspectrum import from_ww3
# from ocean.xPolarSpectrum import find_closest_ww3
from slcl1butils.legacy_ocean.ocean.xspectrum import from_ww3
from slcl1butils.raster_readers import resource_strftime
from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum
from slcl1butils.utils import xndindex

COLOC_LIMIT_SPACE = 100  # km
COLOC_LIMIT_TIME = 3  # hours
INDEX_WW3_FILL_VALUE = -999


def check_colocation(cartesianspWW3, lon, lat, time):
    """
    from information associated to the closest WW3 spectra found, check whether it is a usable co-location or not.
    and add co-location score information

    Args:
        cartesianspWW3 (xr.DataArray): closest cartesian wave spectra WW3 found (could be too far for colocation)
        lon (float): longitude of interest (eg SAR)
        lat (float): latitude of interest (eg SAR)
        time (datetime.datetime or tuple of int): Tuple of the form (year, month, day, hour, minute, second) (eg SAR)


    Returns:
        (xr.DataArray): time index of spatio-temporal closest point in the WW3 file
    """
    coloc_ds = xr.Dataset()
    mytime = (
        np.datetime64(time)
        if type(time) == datetime.datetime
        else np.datetime64(datetime.datetime(*time))
    )
    # time_dist = np.abs(cartesianspWW3.time - mytime)

    # isel = np.where(time_dist == time_dist.min())
    closest_dist_in_space = haversine(
        lon,
        lat,
        cartesianspWW3.attrs["longitude"],
        cartesianspWW3.attrs["latitude"],
    )
    # spatial_dist = haversine(
    #     lon,
    #     lat,
    #     ww3spds[{"time": isel[0]}].longitude,
    #     ww3spds[{"time": isel[0]}].latitude,
    # )
    #  logging.debug(spatial_dist)
    # relative_index = np.argmin(spatial_dist.data)
    # absolute_index = isel[0][relative_index]
    # closest_dist_in_space = spatial_dist[relative_index].data
    # time_dist_minutes = (ww3spec[{"time": absolute_index}].time - mytime).data / (1e9 * 60)
    # time_dist_minutes = (
    #     ww3spds[{"time": absolute_index}].time - mytime
    # ) / np.timedelta64(1, "m")
    time_dist_minutes = (cartesianspWW3.time - mytime) / np.timedelta64(1, "m")
    logging.debug(
        "Wave Watch III closest point @ {} km and {} minutes".format(
            closest_dist_in_space, time_dist_minutes
        )
    )
    if (
        abs(time_dist_minutes) > COLOC_LIMIT_TIME * 60
        or closest_dist_in_space > COLOC_LIMIT_SPACE
    ):
        logging.debug(
            "closest in time then in space is beyond the limits -> No ww3 spectra will be associated "
        )
        absolute_index = INDEX_WW3_FILL_VALUE
        ww3_lon = np.nan
        ww3_lat = np.nan
        ww3_date = np.nan
        selection = xr.DataArray(
            absolute_index,
            attrs={
                "method": "closest in time then in space",
                "limit_of_selection_in_space": "%f km" % COLOC_LIMIT_SPACE,
                "limit_of_selection_in_time": "%f hours" % COLOC_LIMIT_TIME,
                "description": "index of the WW3 spectra selected (None=no WW3 spectra found)",
            },
        )
    else:
        # ww3_lon = ww3spds[{"time": isel[0]}].longitude.data[relative_index]
        # ww3_lat = ww3spds[{"time": isel[0]}].latitude.data[relative_index]
        # ww3_date = ww3spds[{"time": absolute_index}].time
        ww3_lon = cartesianspWW3.attrs["longitude"]
        ww3_lat = cartesianspWW3.attrs["latitude"]
        ww3_date = cartesianspWW3.time
        selection = xr.DataArray(
            cartesianspWW3.attrs["time_index"],
            attrs={
                "method": "closest in time then in space",
                "limit_of_selection_in_space": "%f km" % COLOC_LIMIT_SPACE,
                "limit_of_selection_in_time": "%f hours" % COLOC_LIMIT_TIME,
                "description": "index of the WW3 spectra selected (None=no WW3 spectra found)",
                "_FillValue": INDEX_WW3_FILL_VALUE,
            },
        )
    coloc_ds["WW3spectra_index"] = selection
    coloc_ds["WW3spectra_delta_time"] = xr.DataArray(
        time_dist_minutes,
        attrs={
            "description": "temporal distance (WW3-SAR) in minutes",
            "units": "minutes",
        },
    )
    coloc_ds["WW3spectra_delta_space"] = xr.DataArray(
        closest_dist_in_space,
        attrs={
            "description": "spatial distance between SAR tile center and WW3 spectra grid point (km)",
            "units": "kilometer",
        },
    )
    coloc_ds["WW3spectra_longitude"] = xr.DataArray(
        ww3_lon, attrs={"description": "longitude of colocated WW3 spectra"}
    )
    coloc_ds["WW3spectra_latitude"] = xr.DataArray(
        ww3_lat, attrs={"description": "latitude of colocated WW3 spectra"}
    )
    coloc_ds["WW3spectra_time"] = xr.DataArray(
        ww3_date, attrs={"description": "time associated to colocated WW3 spectra"}
    )
    coloc_ds["WW3spectra_path"] = xr.DataArray(
        cartesianspWW3.attrs["pathWW3"],
        attrs={"description": "file path used to colocate WW3 spectra"},
    )
    return coloc_ds


def resampleWW3spectra_on_TOPS_SAR_cartesian_grid(dsar, xspeckind, nameWW3sp_product):
    """

    Args:
        dsar: xarray.core.Dataset IW Level-1B IFREMER dataset
        xspeckind: str inter or intra
        nameWW3sp_product: str example "ww3CCIseastate_spectra"
    Returns:

    """
    flag_ww3spectra_added = False
    flag_ww3spectra_found = False
    start_date_dt = datetime.datetime.strptime(
        dsar.attrs["start_date"], "%Y-%m-%d %H:%M:%S.%f"
    )
    pathww3sp = get_ww3_path_from_date(
        sar_date=start_date_dt, nameWW3sp_product=nameWW3sp_product
    )
    gridsar = {
        d: k
        for d, k in dsar.sizes.items()
        # if d in ["burst", "tile_sample", "tile_line"]
        if d in ["tile_sample", "tile_line"]
    }
    if (xspeckind == "intra" and "xspectra_2tau_Re" in dsar) or (
        xspeckind == "inter" and "xspectra_Re" in dsar
    ):  # in a future version of L1B xspectra variable could be always present (even on land) but filled by NaN
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
        list_ww3_coloc_sp_ds = []
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
                    sensing_time = dsar["sensing_time"][i].values
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
                        time=sensing_time,
                    )
                    ds_ww3_cartesian.attrs["source"] = "ww3"
                    ds_ww3_cartesian.attrs[
                        "description"
                    ] = "WW3spectra_EFTHraw resampled on SAR cartesian grid"
                    ds_ww3_cartesian = ds_ww3_cartesian.rename("WW3spectra_EFTHcart")
                    colocww3sp_ds = check_colocation(
                        cartesianspWW3=ds_ww3_cartesian,
                        lon=lonsar,
                        lat=latsar,
                        time=start_date_dt,
                    )
                    del ds_ww3_cartesian.attrs["longitude"]
                    del ds_ww3_cartesian.attrs["latitude"]
                    if colocww3sp_ds["WW3spectra_index"].data != INDEX_WW3_FILL_VALUE:
                        # add the raw  EFTH(f,dir) spectra from WW3
                        rawspww3 = (
                            dsww3raw["efth"]
                            .isel(time=colocww3sp_ds["WW3spectra_index"].data)
                            .rename("WW3spectra_EFTHraw")
                        )
                        rawspww3.attrs[
                            "description"
                        ] = "colocated raw EFTH(f,dir) WAVEWATCH III wave height spectra"
                        colocww3sp_ds = colocww3sp_ds.assign_coords(i)
                        colocww3sp_ds = colocww3sp_ds.expand_dims(
                            ["tile_line", "tile_sample"]
                        )
                        list_ww3_coloc_sp_ds.append(colocww3sp_ds)

                        ds_ww3_cartesian = ds_ww3_cartesian.swap_dims(
                            {"kx": "k_rg", "ky": "k_az"}
                        ).T
                        rawspww3 = rawspww3.assign_coords(i)
                        rawspww3 = rawspww3.expand_dims(["tile_line", "tile_sample"])
                        ds_ww3_cartesian = ds_ww3_cartesian.assign_coords(i)
                        ds_ww3_cartesian = ds_ww3_cartesian.expand_dims(
                            ["tile_line", "tile_sample"]
                        )
                        list_ww3_cart_sp.append(ds_ww3_cartesian)
                        list_ww3_efth_sp.append(rawspww3)
                        flag_ww3spectra_added = True
                ds_ww3_cartesian_merged = xr.merge(list_ww3_cart_sp)
                ds_ww3_efth_merged = xr.merge(list_ww3_efth_sp)
                ds_ww3_coloc_merged = xr.merge(list_ww3_coloc_sp_ds)

                dsar = xr.merge(
                    [
                        dsar,
                        ds_ww3_cartesian_merged,
                        ds_ww3_efth_merged,
                        ds_ww3_coloc_merged,
                    ]
                )
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
        logging.info(
            "there is no xspectra in subswath %s",
            dsar.attrs["short_name"].split(":")[2],
        )
    return dsar, flag_ww3spectra_added, flag_ww3spectra_found


def get_ww3_path_from_date(sar_date, nameWW3sp_product):
    """

    Args:
        sar_date: (datetime)
        nameWW3sp_product : (str) eg ww3CCIseastate_spectra


    Returns:
        filename (str): path of the TRCK WW3 HINDCAST corresponding to the
    """
    from slcl1butils.get_config import get_conf

    filename = None
    conf = get_conf()
    if "ww3handcast_track_pattern" in nameWW3sp_product:
        pattern = conf["ww3hindcast_track_pattern"].replace(
            "LOPS_WW3-GLOB-30M_*_trck.nc",
            "LOPS_WW3-GLOB-30M_" + sar_date.strftime("%Y%m") + "_trck.nc",
        )
    elif "ww3CCIseastate_spectra" in nameWW3sp_product:
        pattern = (
            conf["auxilliary_dataset"][nameWW3sp_product]["pattern"]
            .replace("%Y/", sar_date.strftime("%Y/"))
            .replace("%Y%m", sar_date.strftime("%Y%m"))
        )

    else:
        raise ValueError("nameWW3sp_product: not handled : %s" % nameWW3sp_product)
    logging.info("pattern for spectra : %s", pattern)
    datefound, pattern2 = resource_strftime(pattern, step=0.5, date=sar_date)
    filenames = glob.glob(pattern2)

    if len(filenames) > 0:
        filename = filenames[0]
    return filename
