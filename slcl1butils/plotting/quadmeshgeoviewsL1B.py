import logging
import os
import pdb

import geoviews as gv
import holoviews as hv
import numpy as np
import xarray as xr

gv.extension("bokeh")


def doQuadMeshL1BorL1C_onefile(
    ff,
    bursttype="intraburst",
    variable="sigma0",
    clim=(0, 0.2),
    cmap="Greys_r",
    lst_vars=None,
    minis=None,
    maxis=None,
):
    if isinstance(ff, str):
        ff = prepare_L1BC(ff, bursttype, variable)
    else:  # suppose that it is already parsed L1B through prepare_L1BC()
        # print(variable, "suppose that it is already parsed L1B through prepare_L1BC()")
        pass
    logging.info("ff : %s", ff)
    qmeshes = gv.Dataset(ff, kdims=["longitude", "latitude"]).to(gv.QuadMesh)
    # clipping = {"min": "red", "max": "green", "NaN": "gray"}
    clipping = {
        "NaN": "blue"
    }  # see https://holoviews.org/user_guide/Styling_Plots.html
    # clim = (ff.values.min(),ff.values.max())
    if minis is not None:
        clim = (minis[variable], maxis[variable])
        print("extrema", variable, clim)
        fig = qmeshes.opts(
            width=1200,
            height=900,
            colorbar=True,
            cmap=cmap,
            clipping_colors=clipping,
            tools=["hover"],
            clim=clim,
        )
    else:
        if lst_vars:
            clim = lst_vars[variable][1]
            fig = qmeshes.opts(
                width=1200,
                height=900,
                colorbar=True,
                cmap=cmap,
                clipping_colors=clipping,
                tools=["hover"],
                clim=clim,
            )  #
        else:
            fig = qmeshes.opts(
                width=1200,
                height=900,
                colorbar=True,
                cmap=cmap,
                clipping_colors=clipping,
                tools=["hover"],
            )  #

    return fig


def open_of_use(ff, bursttype):
    """

    Args:
        ff: can be xr.Dataset or full path

    Returns:

    """
    if isinstance(ff, str):
        ds = xr.open_dataset(ff, group=bursttype, engine="h5netcdf")
    else:
        ds = ff
    return ds


def prepare_L1BC(ff, bursttype, variable, **kwargs):
    """

    Args:
        ff: str or xr.Dataset
        bursttype: str
        variable: str
    Kwargs:
        kwargs (dict): optional keyword arguments : k_gp (int) phi_hf (int) lambda_range_max_macs(int) are valid entries
    Returns:

    """
    ds = open_of_use(ff, bursttype)
    if "burst" in ds[variable].coords:
        stackable_coords = ["burst", "tile_line"]
    else:
        stackable_coords = ["tile_line"]
    if variable in ["cwave_params"]:
        sel_coords = {"pol": kwargs.get("pol")}
        if "2tau" in ds.dims:
            selection_coords = {
                "2tau": 0,
                "k_gp": kwargs.get("k_gp"),
                "phi_hf": kwargs.get("phi_hf"),
            }
        else:
            selection_coords = {
                "k_gp": kwargs.get("k_gp"),
                "phi_hf": kwargs.get("phi_hf"),
            }
        sub = (
            ds[variable]
            .sel(sel_coords)
            .isel(selection_coords)
            .stack({"y": stackable_coords})
        )
    elif variable in ["macs_Im", "macs_Re"]:
        sub = (
            ds[variable]
            .isel({"lambda_range_max_macs": kwargs.get("lambda_range_max_macs")})
            .stack({"y": stackable_coords})
        )
    else:
        if "pol" in ds[variable].coords:
            sel_coords = {"pol": kwargs.get("pol")}
            sub = ds[variable].sel(sel_coords)
        else:
            sub = ds[variable]
        if "burst" in ds[variable].coords:
            sub = sub.stack({"y": ["burst", "tile_line"]})
        elif (
            "time" in ds[variable].coords
        ):  # for WV (many imagette along time dimension)
            sub = sub.stack({"y": ["time", "tile_line"]})
        else:
            sub = sub.stack({"y": ["tile_line"]})

        logging.debug("sub : %s", sub)
    if np.isnan(sub).any():
        logging.debug(
            "there are %s NaN in the variable stacked : %s",
            np.isnan(sub.data).sum(),
            variable,
        )
    sub = sub.rename({"tile_sample": "x"})
    sub = sub.drop_vars(["y", "tile_line"])
    sub = sub.assign_coords({"y": np.arange(sub.y.size)})
    sub = sub.assign_coords({"x": np.arange(sub.x.size)})
    if (sub == 0).any() and variable == "sigma0":
        pdb.set_trace()
    else:
        pass
        # print('debug not zeros found -> continue')
    if (
        np.isnan(sub).any()
        and variable == "sigma0"
        and (ds["land_flag"].data == False).all()
    ):
        print("Nan alerte")
        pdb.set_trace()
    return sub


def doQuadMeshL1BorL1C_manyfiles(
    files,
    bursttype="intraburst",
    variable="sigma0",
    clim=(0, 0.2),
    title="",
    cmap="Greys_r",
    lst_vars=None,
    minis=None,
    maxis=None,
):
    all_quads = []
    for ff in files:
        oneQuad = doQuadMeshL1BorL1C_onefile(
            ff,
            bursttype=bursttype,
            variable=variable,
            clim=clim,
            cmap=cmap,
            lst_vars=lst_vars,
            minis=minis,
            maxis=maxis,
        )
        all_quads.append(oneQuad)
    tiles = gv.tile_sources.EsriImagery
    fig = tiles * hv.Overlay(all_quads)
    fig.opts(title=title, tools=["hover"])
    return fig


def doQuadMeshL1BorL1C_manyfiles_opt(
    files,
    pol,
    variables,
    bursttype="intraburst",
    clim=(0, 0.2),
    title="",
    cmap="Greys_r",
    outputdir=None,
    outbasename=None,
    lst_vars=None,
    **kwargs,
):
    """

    Args:
        files: list of fulll path str
        pol : str vv vh hv hh
        variables: list of string
        bursttype: str intraburts or interburst
        clim: tuple eg (0.2,4.5)
        title: str
        cmap:str eg jet
        outputdir: str
    Kwargs:
        kwargs (dict): optional keyword arguments : k_gp (int) phi_hf (int)  are valid entries

    Returns:

    """
    fig = None
    allds = {}
    for ff in files:
        allds[ff] = open_of_use(ff, bursttype)
    logging.info("all the files are open once!")
    subs = {}
    subdsx = []
    minis = {}
    maxis = {}
    for vvi, var in enumerate(variables):
        if isinstance(cmap, list) and len(cmap) == len(variables):
            thecmap = cmap[vvi]
        else:
            thecmap = cmap
        logging.info("variable = %s", var)
        for ff in files:
            ds = allds[ff]
            kwargs["pol"] = pol
            sub = prepare_L1BC(ds, bursttype, var, **kwargs)
            subs["%s_%s" % (var, ff)] = sub
            subdsx.append(sub)
            if var not in minis:
                minis[var] = sub.values.min()
                maxis[var] = sub.values.min()
            else:
                if sub.values.min() < minis[var]:
                    minis[var] = sub.values.min()
                if sub.values.max() > maxis[var]:
                    maxis[var] = sub.values.max()
        if lst_vars is None:
            maxis_go = maxis
            minis_go = minis
        else:
            maxis_go = None
            minis_go = None
        fig = doQuadMeshL1BorL1C_manyfiles(
            subdsx,
            bursttype=bursttype,
            variable=var,
            clim=clim,
            title=title,
            cmap=thecmap,
            lst_vars=lst_vars,
            minis=minis_go,
            maxis=maxis_go,
        )
        if outputdir:
            renderer = hv.renderer("bokeh")
            #
            # Using renderer save
            if outbasename is None:
                outf = os.path.join(outputdir, "map_%s_%s" % (pol, var))
            else:
                outf = os.path.join(outputdir, outbasename)
            renderer.save(fig, outf)
            # save(fig, outf)
            logging.info("output file: %s", outf + ".html")
    return fig
