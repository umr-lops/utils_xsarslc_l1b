# import slcl1butils.legacy_ocean.ocean as ocean
import numpy as np
import xarray as xr
def from_ww3(ds,kx=None,ky=None, **kwargs):
    """
    Return a xCartesianSpectrum from a WW3 spectrum. The new grid depends on what is passed in the keyword argument. All keyword argument are passed to from_xPolarSpectrum()

    Args:
        ds (str) : the path of the WW3 spectrum

    Keyword Args:
        station_index (int, optional) : index of the required station
        time_index (int, optional) : index of the required time
        shortWavesFill (bool, optional) : Fill the short wave part of the spectrum. If True, kwargs arguments are passed to ocean.xPolarSpectrum.shortWavesFill()
        kwargs (dict, optional): other keywords arguments passed to from_xPolarSpectrum()
    """
    import slcl1butils.legacy_ocean.ocean.xPolarSpectrum as xPS
    specww3 = xPS.from_ww3(ds, **kwargs)
    return from_xPolarSpectrum(specww3,kx=kx,ky=ky, **kwargs)


def surf_params_to_wavenumber_params(**kwargs):
    """
    Transform surface parameters into wavenumber parameters based on FFT transform assumptions

    Keyword Args:
        dr (tuple of float, optional) : spatial sampling step [m]
        size (tuple of float, optional) : surface size [m]

    Returns:
        (dict) : dictionary with keys 'dk' (tuple of float) wavenumber sampling, and 'kmax' (tuple of float) maximum wavenumber
    """
    size = kwargs.pop("size")
    dr = kwargs.pop("dr")
    (kex, key) = (2 * np.pi / dr[0], 2 * np.pi / dr[1])  # sampling k
    return {
        "dk": (2 * np.pi / size[0], 2 * np.pi / size[1]),
        "kmax": (kex, key),
    }  # kmax used in the spectrum has to be the sampling rate ke (twice the real kmax)   

def from_xPolarSpectrum(ds, **kwargs):
    """
    Convert an xPolarSpectrum into a xCartesianSpectrum. Grid of the return spectrum depends on the keywords arguments
    If 'size' and 'dr' are provided as keyword arguments, they are used prioritary. Otherwise, kmax and dk are used instead

    Args:
        ds (xarray.dataset or xarray.dataarray): an xPolar Spectrum instance

    Keyword Args:
        dr (tuple of float, optional) : spatial sampling step [m]
        size (tuple of float, optional) : surface size [m]
        dk (tuple of float, optional) : wavenumbers spacing  (dkx, dky) : - Default : (2*np.pi/400, 2*np.pi/400) --
        kmax (tuple of float, optional) : (kxmax, kymax) maximum norm wavenumbers - Default : (2*np.pi/0.2, 2*np.pi/0.2)--
        strict (str, optional) : Default : 'kmax'
            dk' : means that 'dk' is preserved as passed and kmax is adjusted in order to have kmax to be a multiple of dk
            'kmax' : means that 'kmax' is preserved as passed and 'dk' is adjusted in order to have kmax to be a multiple of dk

    Returns:
        (xarray) : xCartesianSpectrum
    """

    if "size" in kwargs and "dr" in kwargs:
        params = surf_params_to_wavenumber_params(**kwargs)
        kwargs.update(params)

    (kx, ky), (dkx, dky) = SpectrumCartesianGrid(**kwargs)

    kx = xr.DataArray(kx, dims="kx", attrs={"spacing": dkx, "units": "rad/m"})
    ky = xr.DataArray(ky, dims="ky", attrs={"spacing": dky, "units": "rad/m"})
    k = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)

    #   Extend dataset to pi angle to avoid nan at the border near t=pi during interpolation
    nds = ds[{"phi": 0}].copy()
    nds["phi"] = ds[{"phi": 0}].phi.data + 2.0 * np.pi
    nds = xr.concat((ds, nds), dim="phi")

    myspec = nds.interp(k=k, phi=phi).assign_coords(kx=kx, ky=ky)

    myspec.attrs.update({"dkx": dkx, "dky": dky})
    myspec = myspec.where(
        np.sqrt(myspec.kx**2 + myspec.ky**2) > min(ds.k).data, other=0.0
    )  # spectrum at wavenumbers smaller than the minimum one of the polar spectrum are set to zero
    myspec.attrs.pop("dphi", None)
    return myspec.drop(("k", "phi", "dk"))

def SpectrumCartesianGrid(*, dk= (2*np.pi/400., 2*np.pi/400.), kmax=(2*np.pi/0.2, 2*np.pi/0.2), strict = 'kmax', next_fast_len=False, **kwargs):
    """
    Generate the cartesian grid of the spectrum

    Args :
        dk (tuple of float, optional) : wavenumbers spacing  (dkx, dky)
        kmax (tuple of float, optional) : (kxmax, kymax) maximum norm wavenumbers - Default : (2*np.pi/0.2, 2*np.pi/0.2)--
        strict (str, optional) :
            dk' : means that 'dk' is preserved as passed and kmax is adjusted in order to have kmax to be a multiple of dk
            'kmax' : means that 'kmax' is preserved as passed and 'dk' is adjusted in order to have kmax to be a multiple of dk
        next_fast_len (bool, optional): choose a good number of point for fast calculation. pyfftw module required
    Returns :
        k (tuple of two 1darray) : Arrays of wavenumber in x and y directions. Shapes are (Nkx,) and  (Nky,)
        dk (tuple of float) : wavenumber spacing in x and y directions
    """

    # strict = kwargs.pop('strict', 'kmax')
    # dk = kwargs.pop('dk', (2*np.pi/400., 2*np.pi/400.))
    # kmax =  kwargs.pop('kmax', (2*np.pi/0.2, 2*np.pi/0.2))

    #--------------Wavenumber initialization---------------------------
    Nk = (int(np.rint(kmax[0]/dk[0]))*2, int(np.rint(kmax[1]/dk[1]))*2) # the //2*2 ensure an even number of points

    if next_fast_len:
        from pyfftw import next_fast_len
        Nk = next_fast_len(Nk[0]), next_fast_len(Nk[1])

    if strict=='dk':
        k = (np.fft.fftshift(np.fft.fftfreq(Nk[0], 1./(Nk[0]*dk[0]))), np.fft.fftshift(np.fft.fftfreq(Nk[1], 1./(Nk[1]*dk[1])))) # dk is preserved
        kmax = (max(k[0]), max(k[1]))
    else:
        k = (np.fft.fftshift(np.fft.fftfreq(Nk[0], 1./kmax[0])), np.fft.fftshift(np.fft.fftfreq(Nk[1], 1./kmax[1]))) # kmax is preserved
        dk = (kmax[0]/Nk[0], kmax[1]/Nk[1])

    return k, dk