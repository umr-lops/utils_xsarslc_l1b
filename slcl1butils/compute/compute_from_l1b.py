import logging
from datetime import datetime

import numpy as np
import xarray as xr


def get_start_date_from_attrs(ds) -> datetime:
    """

    read and decode attribute to get start time of SAR acquisition

    """
    if "start_date" in ds.attrs:
        logging.debug("attrs : %s0", ds.attrs["start_date"])

        sar_date = datetime.strptime(
            str.split(ds.attrs["start_date"], ".")[0], "%Y-%m-%d %H:%M:%S"
        )
    else:
        # ASAR attribute format
        sar_date = datetime.strptime(
            str.split(ds.attrs["time_coverage_start"], ".")[0], "%Y-%m-%dT%H:%M:%S"
        )
    return sar_date


def compute_xs_from_l1b(
    _file, burst_type="intra", time_separation="2tau"
) -> (xr.DataArray, xr.Dataset):
    """

    :params _file (str): full path L1B nc file
    :params burst_type (str): intra or inter
    :params time_separation (str): 2tau or 1tau...

    :Returns:

        xs (xarray.DataArray):
        ds (xarray.Dataset):
    """
    # Reading the l1b file
    # Loading the specified burst group
    if "wv" in _file:
        ds = xr.open_dataset(_file, group=burst_type)
    else:
        ds = xr.open_dataset(_file, group=burst_type + "burst")

    # ds = dt[burst_type+'burst_xspectra'].to_dataset()
    # drop variables

    logging.debug("time_separation : %s", time_separation)
    if burst_type == "intra":
        consolidated_list = []
        list_to_drop = ["var_xspectra_0tau", "var_xspectra_1tau", "var_xspectra_2tau"]
        for toto in range(0, 2):
            if int(time_separation[0]) != toto:
                list_to_drop.append("xspectra_" + str(toto) + "tau" + "_Re")
                list_to_drop.append("xspectra_" + str(toto) + "tau" + "_Im")
        for vv in list_to_drop:
            if vv not in ds:
                logging.warning("%s not present in the dataset %s", vv, burst_type)
            else:
                consolidated_list.append(vv)
        ds = ds.drop_vars(consolidated_list)
    else:  # inter burst case
        pass  # no variable to remove in interburst

    if burst_type == "intra" or burst_type == "":
        if (
            "xspectra_" + time_separation + "_Re" not in ds
            or "xspectra_" + time_separation + "_Im" not in ds
        ):
            xsRe = None
            xsIm = None
        else:
            xsRe = ds[
                "xspectra_" + time_separation + "_Re"
            ]  # +1j*ds_intra['xspectra_1tau_Im']).mean(dim=['1tau'])
            xsIm = ds["xspectra_" + time_separation + "_Im"]
            if time_separation == "2tau":
                xsRe = xsRe.squeeze("2tau")
                xsIm = xsIm.squeeze("2tau")
            if time_separation == "1tau":
                xsRe = xsRe.mean(dim=["1tau"])
                xsIm = xsIm.mean(dim=["1tau"])

    elif burst_type == "inter":
        if "xspectra_Re" in ds:
            xsRe = ds["xspectra_Re"]  # +1j*ds_inter['xspectra_Im']
            xsIm = ds["xspectra_Im"]
        else:
            logging.warning("xspectra_Re absent from interburst group")
            xsRe = None
            xsIm = None
    else:  # WV case
        raise Exception("not handle case")
    if xsRe is None:
        xs = None
    else:
        xs = xsRe + 1j * xsIm
        # Remove unique dimensions
        # xs=xs.squeeze()
        # convert the wavenumbers variables in range and azimuth into coordinates after selection of one unique vector without any other dimsension dependency
        dims_to_average = []
        if "tile_sample" in xs.k_rg.dims:
            dims_to_average.append("tile_sample")

        if "burst" in xs.k_rg.dims:
            dims_to_average.append("burst")
        if "tile_line" in xs.k_rg.dims:
            dims_to_average.append("tile_line")
        xs = xs.assign_coords({"k_rg": xs.k_rg.mean(dim=dims_to_average)})

        # Replace the dimension name for frequencies
        xs = xs.swap_dims({"freq_sample": "k_rg", "freq_line": "k_az"})
        # Bug Fix to define the wavenumber in range direction.
        xs.k_rg.attrs.update(
            {"long_name": "wavenumber in range direction", "units": "rad/m"}
        )

    return xs, ds


def compute_xs_from_l1b_wv(
    file, time_separation="2tau", crop_limits=None
) -> (xr.DataArray, xr.Dataset):
    """
    # Reading the level1-b file to reconstruct complex full cross spectra
    # Loading the specified burst group

    Args:
        file: str full path of Level-1B XSP WV SAFE.nc
        time_separation: str '2tau' or '1tau' [default="2tau"]
        crop_limits: dict to crop the variable depending on freq_sample and freq_line e.g. {'rg':0.15,'az':0.20} [optional,default= None -> no cropping]
    Returns:
        xs: xr.DataArray (complex cross spectra)
        dt: xr.Dataset of the WV L1B (intra burst, since there is no inter burst group for WV)
    """
    # dt = xr.open_datatree(file)
    ds = xr.open_dataset(file, engine="h5netcdf").load()
    if crop_limits is not None:
        logging.info("crop spectra with wave numbers below : %s", crop_limits)
        indrg_to_keep = np.where(ds["k_rg"].isel(time=0) <= crop_limits["rg"])[0]
        indaz_to_keep = np.where(ds["k_az"].isel(time=0) <= crop_limits["az"])[0]
        logging.info(
            "new half cross spectra should be cropped from :[rg x az] %ix%i -> %ix%i",
            len(ds["freq_sample"]),
            len(ds["freq_line"]),
            len(indrg_to_keep),
            len(indaz_to_keep),
        )
        ds = ds.isel(freq_sample=indrg_to_keep, freq_line=indaz_to_keep)

        # for vv in ds:
        # if 'freq_sample' in ds[vv].dims:
        # ds[vv] = ds[vv].where((ds['k_rg']<=crop_limits['rg']) & (ds['k_az']<=crop_limits['az']),drop=True)
    # ds = xr.open_dataset(_file, group="")
    # xs_wv = {}
    # xs = None
    # for group in dt.children:

    # ds = dt[burst_type+'burst_xspectra'].to_dataset()
    # ds = dt[group].to_dataset()

    # ds = dt.to_dataset()
    xsRe = ds[
        "xspectra_" + time_separation + "_Re"
    ]  # +1j*ds_intra['xspectra_1tau_Im']).mean(dim=['1tau'])
    xsIm = ds["xspectra_" + time_separation + "_Im"]
    if time_separation == "2tau":
        xsRe = xsRe.squeeze("2tau")
        xsIm = xsIm.squeeze("2tau")
    if time_separation == "1tau":
        xsRe = xsRe.mean(dim=["1tau"])
        xsIm = xsIm.mean(dim=["1tau"])

    xs = xsRe + 1j * xsIm

    # Remove unique dimensions
    # xs=xs.squeeze()
    # convert the wavenumbers variables in range and azimuth into coordinates after
    # selection of one unique vector without any other dimension dependency
    # xs = xs.assign_coords({'k_rg': xs.k_rg})

    # even WV can have many tiles per imagette
    # so we need to use the same k_rg for all tiles
    dims_to_average = []
    if "time" in xs.k_rg.dims:
        dims_to_average.append("time")
    if "tile_sample" in xs.k_rg.dims:
        dims_to_average.append("tile_sample")
    if "tile_line" in xs.k_rg.dims:
        dims_to_average.append("tile_line")
    xs = xs.assign_coords({"k_rg": xs.k_rg.mean(dim=dims_to_average)})

    # Replace the dimension name for frequencies

    # same for k_az : we take the mean of all the tiles
    xs = xs.assign_coords(
        {
            "k_az": xs["k_az"].mean(
                dim=set(xs["k_az"].dims) - set(["freq_line"]), keep_attrs=True
            )
        }
    )
    xs = xs.swap_dims({"freq_sample": "k_rg", "freq_line": "k_az"})
    # Bug Fix to define the wavenumber in range direction.
    xs.k_rg.attrs.update(
        {"long_name": "wavenumber in range direction", "units": "rad/m"}
    )
    # xs_wv[group] = xs
    return xs, ds
