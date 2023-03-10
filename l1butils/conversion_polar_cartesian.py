import numpy as np
import xarray as xr
import logging
def from_xCartesianSpectrum(ds, *, Nphi=120, ksampling='log',keep_nan=True, **kwargs):
    """
    Convert a xCartesianSpectrum into a xPolarSpectrum. Grid of the return spectrum depends on the keywords arguments
    If 'size' and 'dr' are provided as keyword arguments, they are used prioritary. Otherwise, kmax and dk are used instead

    Args:
        ds (xarray.dataset or xarray.dataarray): a xCartesianSpectrum instance

    Keyword Args:
        Nphi (float, optional) : Number of azimuthal sampling
        ksampling (string, optional):  'log' or 'lin' for logarithmic or linear sampling

    Returns:
        (xarray) : xPolarSpectrum
    """

    kmin = max(np.sort(np.abs(ds.kx))[1], np.sort(np.abs(ds.ky))[1]) # first element is zero
    kmax = min(np.sort(np.abs(ds.kx))[-3], np.sort(np.abs(ds.ky))[-3]) # if last one is used, interpolation do not work well

    kwargs.update({'kmin':kmin, 'kmax':kmax, 'ksampling':ksampling, 'Nphi':Nphi})
    logging.debug('kwargs %s',kwargs)
    if 'Nk' in kwargs:
        k, dk, phi, dphi = SpectrumPolarGrid(**kwargs)
    elif 'k' in kwargs: #add agrouaze
        #logging.debug('second way to define k')
        k = kwargs['k']
        Nk = 60
        dk = np.log(10) * (np.log10(kmax) - np.log10(kmin)) / (Nk - 1) * k
        _,_,phi,dphi = SpectrumPolarGrid(**kwargs)
    else:
        raise Exception('impossible to define the spectrum grid')
    k = xr.DataArray(k, dims='k')
    phi = xr.DataArray(phi, dims='phi')

    kx = k*np.cos(phi)
    ky = k*np.sin(phi)
    if keep_nan:
        myspec= ds.interp(kx=kx, ky=ky).assign_coords(k=k, phi=phi)
    else:
        myspec = ds.interp(kx=kx, ky=ky, kwargs={"fill_value": None}).assign_coords(k=k, phi=phi) # correction Grouazel May 2022 to avoid NaN in polar spectrum
    myspec = myspec.assign_coords({'dk':xr.DataArray(dk, dims='k')}) if ksampling=='log' else myspec.assign_coords({'dk':xr.DataArray(dk)})
    myspec.attrs.update({'dphi':dphi})
    myspec.attrs.pop('dkx', None)
    myspec.attrs.pop('dky', None)

    return myspec.drop(('kx','ky'))


def SpectrumPolarGrid(**kwargs):
    """
    Generate the polar grid of the spectrum
    kwargs :
    ------------
        Nk : int :  Number of wavenumbers norms - optional : default 1000 --
        Nphi : int :  Number of wavenumbers directions - optional : default 120 --
        kmin : float : minimum wavenumber - optional : Default : 2*np.pi/800 --
        kmax : float : maimum wavenumber - optional : Default : 4000 --
        ksampling : string : - optional : Default : 'log' --
            'log' : logarithmic sampling is used
            'lin' : linear sampling is applied

    output :
    ----------
        dict with keys:
            k : 1D array : array of wavenumber norms [rad/m]
            dk : 1D array : array of wavenumber spacing [rad/m]
            phi : 1D array : array of azimuth [radians]
            dphi : 1D array : array of azimuth spacing [radians]
            km : 2D array : matrix of wavenumber distance [rad/m]
            tm :  : 2D array : matrix of wavenumber [radians]
            Nk : int : Number of wavenumber points
            Nphi : int : Number of azimuth points
            kmin    : float : minimum wavenumber - optional : Default : .... --
            kmax : float : maximum wavenumber - optional : Default : ... --
            ksampling : string : - optional : Default : 'log' --
                'log' : a logarithmic sampling is used
                'lin' : a linear sampling is applied
    """

    Nk = kwargs.pop('Nk', 400)
    Nphi =  kwargs.pop('Nphi', 120)
    kmin =  kwargs.pop('kmin', 2*np.pi/800.)
    kmax =  kwargs.pop('kmax', 4000.)
    ksampling = kwargs.pop('ksampling', 'log')

    #--------------Wavenumber initialization---------------------------

    if ksampling == 'log':
        k = np.logspace(np.log10(kmin), np.log10(kmax),Nk)
        dk = np.log(10)*(np.log10(kmax)-np.log10(kmin))/(Nk-1)*k
    elif ksampling == 'lin':
        k = np.linspace(kmin, kmax,Nk)
        #  dk = (kmax-kmin)/(Nk-1)*np.ones(k.shape)
        dk = (kmax-kmin)/(Nk-1)
    else:
        raise ValueError('Unknown k sampling method')

    dphi = 2*np.pi/Nphi
    phi = np.arange(-np.pi, np.pi, dphi)
    return k,dk, phi,dphi #, 'km':km, 'tm':tm, 'Nk':Nk, 'Nphi':Nphi, 'kmin':kmin, 'kmax':kmax, 'ksampling':ksampling