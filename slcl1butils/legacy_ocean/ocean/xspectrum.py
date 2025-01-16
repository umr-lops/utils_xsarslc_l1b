import numpy as np
import xarray as xr
import logging
# from ocean.spectrum_private.spectra_functions import *
from slcl1butils.legacy_ocean.ocean.spectrum_private.spectra_functions import tony_omni,tony_spread,findUstar
# from shared.my_functions import *
import dask

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

    #--------------Wavenumber initialization---------------------------
    Nk = (int(np.rint(kmax[0]/dk[0]))*2, int(np.rint(kmax[1]/dk[1]))*2) # Nk points over the [-kmax, kmax[ span

    if next_fast_len:
        pass

    if strict=='dk':
        k = (np.fft.fftshift(np.fft.fftfreq(Nk[0], 1./(Nk[0]*dk[0]))), np.fft.fftshift(np.fft.fftfreq(Nk[1], 1./(Nk[1]*dk[1])))) # dk is preserved
        kmax = (max(k[0]), max(k[1]))
    else:
        k = (np.fft.fftshift(np.fft.fftfreq(Nk[0], 0.5/kmax[0])), np.fft.fftshift(np.fft.fftfreq(Nk[1], 0.5/kmax[1]))) # kmax is preserved
        dk = (2*kmax[0]/Nk[0], 2*kmax[1]/Nk[1])

    return k, dk


def surf_params_to_wavenumber_params(**kwargs):
    """
    Transform surface parameters into wavenumber parameters based on FFT transform assumptions

    Keyword Args:
        dr (tuple of float, optional) : spatial sampling step [m]
        size (tuple of float, optional) : surface size [m]

    Returns:
        (dict) : dictionary with keys 'dk' (tuple of float) wavenumber sampling, and 'kmax' (tuple of float) maximum wavenumber
    """
    size = kwargs.pop('size')
    dr = kwargs.pop('dr')
    (kex, key)=(np.pi/dr[0], np.pi/dr[1]) # kmax (range of wavenumber is [-kmax, kmax[)
    return {'dk':(2*np.pi/size[0], 2*np.pi/size[1]), 'kmax' : (kex, key)} # kmax used in the spectrum has to be the sampling rate ke (twice the real kmax)


def from_surfparams(**kwargs):
    """
    return a CartesianOceanSpectrum from surface parameters.

    Keyword Args:
        dr (tuple of float, optional) : spatial sampling step [m]
        size (tuple of float, optional) : surface size [m]

    Returns:
        () : see from_wavenumbers() Keyword Args
    """
    params = surf_params_to_wavenumber_params(**kwargs)
    return from_wavenumber(**params, **kwargs) # kmax used in the spectrum has to be the sampling rate ke (twice the real kmax)


def set_directivity(ds, directivity, **kwargs):
    """
    set wave directivity.
    kwargs
    ---------
        wd: wind direction in degrees relative to x direction - optional - default wind direction is x
        if directivity is 'alongwind' additional arguments can be defined:
            dir_weight : float in [0.,1.]: weight ponderation of the angular function. weight = 0.8 means 80 % energy in the wind direction half quadrant -- optional, default 1. --
            dir_alpha : float : In alongwind mode, alpha is the angular cutting velocity in the tanh angular profile -- optional, default 10 --
    """
    ds.attrs.update({'wd':np.radians(kwargs.pop('wd', np.degrees(ds.wd)))})
    dir_alpha = kwargs.pop('dir_alpha', 10.)
    dir_weight = kwargs.pop('dir_weight', 1.)
    #         reset_derivations(ds)
    tm = np.arctan2(ds.ky, ds.kx).transpose(*ds.dims)  # direction of wavenumber (relative to x axis)
    if directivity == 'alongwind':
        #  self.magnitude+=np.roll(np.flipud(np.roll(np.fliplr(self.magnitude), 1,axis=1)),1,axis=0) # it works when spectrum is on a fftfreq kind of grid
        ds+=np.roll(np.flipud(np.roll(np.fliplr(ds), 1,axis=1)),1,axis=0) # it works when spectrum is on a fftfreq kind of grid
        angular_weight_func = lambda t: 0.5*(np.tanh(dir_alpha*(t+np.pi/2.))-np.tanh(dir_alpha*(t-np.pi/2.)))
        angle_mpipi = lambda x:np.arctan2(np.sin(x), np.cos(x))
        weight = angular_weight_func (angle_mpipi(tm-ds.wd))
        ds*=(dir_weight*(weight-0.5)+0.5)
    elif directivity == 'dissymmetry':
        ds = 2.*ds*(np.sin(((tm-ds.wd)+np.pi)/2)**2)
    #  elif directivity == 'None' and ds.directivity!='None':
    elif directivity == 'None':
        ds.attrs.update({'directivity':'None'})
    #             self.eval_spectrum()
    elif directivity == 'None' and ds.directivity=='None':
        pass
    else:
        raise Exception('Wrong directivity for {} spectrum : {}'.format(ds.name, directivity))




def from_wavenumber(**kwargs):
    """
    Generate Cartesian Ocean spectrum from wavenumbers inputs specificities and spectrum characteristics

    Keyword Args:
        ws (float) : wind speed [m/s]-- optional : default 0. --
        wd (float) : wind direction [deg]-- optional : default 0. --
        omega (float) : wave age-- optional : default 0.84 --
        name (str) : name of the wind sea spectrum ('elfouhaily', 'kudryavtsev')-- optional : default elfouhaily --
        cur (float) : current speed [m/s] -- optional : default : 0. --
        curdir (float) : current direction [deg] -- optional : default : 0. --
        sa (float or list of float) : swell amplitude -- optional --
        sk (float or list of float) : swell wavenumber -- optional --
        sd (float or list of float) : swell direction [deg] in [-pi,pi] range -- optional --
        sas (float or list of float) : swell azimuthal spreading -- optional default 0. --
        sks (float or list of float) : swell wavenumber spreading -- optional default 0.  --
        client (dask.distributed.client) : client to be used for parallel calculation. If not provided, dask configuration is used
    Returns:
        (xarray) : Cartesian Ocean spectrum
    """

    (kx,ky), (dkx, dky) = SpectrumCartesianGrid(**kwargs)
    name = kwargs.pop('name', 'elfouhaily')
    ws = kwargs.pop('ws', 0.)
    wd = np.radians(kwargs.pop('wd', 0.))
    cur = kwargs.pop("cur", 0.)
    curdir = np.radians(kwargs.pop("curdir", 0.))
    myscheduler = kwargs.pop('client', None)

    if ('omega' not in kwargs) and ('fetch' not in kwargs):
        omega = 0.84
        fetch = np.inf
    elif ('omega' in kwargs) and ('fetch' in kwargs):
        raise ValueError('Both omega and fetch can not be defined')
    elif ('omega' in kwargs) and ('fetch' not in kwargs):
        omega = kwargs.pop('omega')
        fetch = 2.2e4*(np.arctanh((omega/0.84)**(-1./0.75)))**(2.5)
    else:
        fetch = kwargs.pop('fetch')
        omega = 0.84*np.tanh((fetch/(2.2e4))**0.4)**(-0.75)

    #  ds = xr.DataArray(np.zeros((len(kx), len(ky))), dims = ('kx','ky'), coords={'kx':kx, 'ky':ky} , attrs = {'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dkx':dkx, 'dky':dky, 'units':'m**4/rad**2', 'cur':cur, 'curdir':curdir}, name='spectrum magnitude')
    kx = xr.DataArray(kx , dims = ('kx',) , attrs = {'spacing':dkx, 'units':'rad/m'}).chunk({'kx':500})
    ky = xr.DataArray(ky , dims = ('ky',) , attrs = {'spacing':dky, 'units':'rad/m'}).chunk({'ky':500})
    km = np.sqrt(kx**2+ky**2)
    tm = np.arctan2(ky, kx)

    # if name == 'elfouhaily' and ws>0.:
    # ustar = findUstar(ws, 10.)
    # curv = dask.array.map_blocks(tony_omni, km.data, ws, ustar, omega)
    # spread = dask.array.map_blocks(tony_spread, km.data, ws, omega)
    # tm = tm.transpose(*km.dims)
    ## spread = 0.; logging.debug('*******************WARNING : SPECTRUM IS FORCED OMNI !!**********************')
    # ds = xr.DataArray((curv/(2*np.pi*km**4)*(1.+spread*np.cos(2*(tm-wd)))).compute(scheduler = myscheduler), dims = ('kx','ky'), coords={'kx':kx, 'ky':ky} , attrs = {'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dkx':dkx, 'dky':dky, 'units':'m**4/rad**2', 'cur':cur, 'curdir':curdir}, name='spectrum magnitude')
    # logging.debug('Future warning : map_blocks function for xarray input is not available for now. It can leads to confusion on dimension when moving back to xarray. Please update ASAP', end='')
    # set_directivity(ds, kwargs.pop('directivity', 'alongwind'))
    # elif name == 'kudryavtsev':
    # raise ValueError('kudryavtsev spectrum not implemented in xspectrum')
    ## self.magnitude = np.transpose(kudryavtsev(self.k, angle_mpipi(self.t-self.windDir), self.dt, self.windSpeed, self.fetch))/(self.km**4)
    # else:
    # logging.debug('No wind sea spectrum defined')
    if name == 'elfouhaily' and ws>0.:
        ustar = findUstar(ws, 10.)
        curv = xr.map_blocks(tony_omni, km, [ws, ustar, omega]).load()
        spread = xr.map_blocks(tony_spread, km, [ws, omega]).load()
        ds = (curv/(2*np.pi*km**4)*(1.+spread*np.cos(2*(tm-wd))))
        ds = ds.load()

        # ds = (curv/(2*np.pi*km**4)*(1.+spread*np.cos(2*(tm-wd)))).compute(scheduler = myscheduler)
        ds = ds.assign_coords(kx=kx, ky=ky).rename('spectrum magnitude')
        ds.attrs.update({'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dkx':dkx, 'dky':dky, 'units':'m**4/rad**2', 'cur':cur, 'curdir':curdir})
        #  spread = 0.; logging.debug('*******************WARNING : SPECTRUM IS FORCED OMNI !!**********************')
        set_directivity(ds, kwargs.pop('directivity', 'alongwind'))
    elif name == 'kudryavtsev':
        raise ValueError('kudryavtsev spectrum not implemented in xspectrum')
        #  self.magnitude = np.transpose(kudryavtsev(self.k, angle_mpipi(self.t-self.windDir), self.dt, self.windSpeed, self.fetch))/(self.km**4)
    else:
        logging.debug('No wind sea spectrum defined')

    if 'ds' not in locals(): ds = xr.DataArray(np.zeros((len(kx), len(ky))), dims = ('kx','ky'), coords={'kx':kx, 'ky':ky} , attrs = {'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dkx':dkx, 'dky':dky, 'units':'m**4/rad**2', 'cur':cur, 'curdir':curdir}, name='spectrum magnitude')

    if ('sa' in kwargs) or ('sk' in kwargs)  or ('sd' in kwargs):

        sa =  np.array(np.double(kwargs.pop('sa', 1.)), ndmin=1)
        sk =  np.array(np.double(kwargs.pop('sk', 2*np.pi/10.)), ndmin=1)
        sd =  np.array(np.double(kwargs.pop('sd', 0.)), ndmin=1)
        if len(sk) != len(sa):
            sa = np.tile(sa[0], len(sk))
        if len(sk) != len(sd):
            raise ValueError('sk and sd must have the same length')

        swell_magnitude = np.zeros(km.shape)
        jkx = list()
        jky = list()
        logging.debug('Sine wavenumbers and directions are redefined to :')
        for e in sk*np.cos(np.radians(sd)):
            jkx.append(np.argmin(np.abs(ds.kx.data-e)))
        for e in sk*np.sin(np.radians(sd)):
            jky.append(np.argmin(np.abs(ds.ky.data-e)))

        for i,(ikx,iky) in enumerate(zip(jkx, jky)):
            swell_magnitude[ikx, iky] = 0.5*(sa[i]/np.sqrt((ds.dkx*ds.dky)))**2
            logging.debug(('\tk{} = {} rad/m - A{} = {} m - azi{} = {} deg - lambda{} = {} m - f{} = {} Hz').format(i, np.round(float(km[ikx, iky]),3), i, sa[i], i, np.round(np.rad2deg(float(np.arctan2(ds.ky[iky].data, ds.kx[ikx].data))),3), i, np.round(2*np.pi/float(km[ikx, iky]),3), i, np.round(np.sqrt(9.81*float(km[ikx, iky]))/(2.*np.pi),3)))
        ds+= swell_magnitude
    ds[np.where(np.isinf(ds))] = 0.
    ds[np.where(np.isnan(ds))] = 0.
    return ds




def compute_moments(ds, moments):
    """
        Evaluate numerical integration of the spectrum moments:  \int (k*cos(a))**ni (k*sin(a))**nj (omega**nw) spec(k) k dk dt where ni, nj, nw are the three digits in the moment list and 'a' the angular direction [degrees] relative to original direction
        moments : tuple composed of the 3 indexes ni,nj and nw and angular direction. ex : (1,0,0,90)
        If angle is not provided (ex: moment=(1, 0, 1)) angle "a" is assumed to be zero
    """
    if 'moments' not in ds.attrs : ds.attrs.update({'moments':{}})
    if type(moments)==tuple: moments = [moments] # This is different from list(moment)
    moments = [m if len(m)==4 else (*m,0.) for m in moments]
    moments = [(int(m[0]), int(m[1]), int(m[2]), float(m[3]))  for m in moments if m not in ds.moments.keys()]
    moments = dict.fromkeys(moments)

    km = np.sqrt(ds.kx**2+ds.ky**2)
    tm = np.arctan2(ds.ky, ds.kx).transpose(*ds.dims)  # direction of wavenumber (relative to x axis)
    w = np.sqrt(9.81*km*(1.+(km/363.2)**2))

    for key in moments.keys():
        ni, nj, nw, a = key
        if (nw>0) and (ds.cur>0.) : logging.debug('Current is not taken into account for computing moment {}'.format(key))
        moments[key]=(((km*np.cos(tm-np.radians(a)))**int(ni)*(km*np.sin(tm-np.radians(a)))**int(nj)*(w**int(nw))*ds).sum()*ds.dkx*ds.dky).data
    ds.attrs['moments'].update(moments)


def compute_swh(ds):
    compute_moments(ds, (0,0,0,0))
    ds.attrs.update({'swh':4.*np.sqrt(ds.moments[(0,0,0,0.)])})
    return ds.swh

def compute_mss(ds):
    compute_moments(ds, [(2,0,0,0.), (0,2,0,0.)])
    ds.attrs.update({'mss':ds.moments[(2,0,0,0.)]+ds.moments[(0,2,0,0.)]})
    return ds.mss


def from_ww3(ds, **kwargs):
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
    return from_xPolarSpectrum(specww3, **kwargs)


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

    if 'size' in kwargs and 'dr' in kwargs:
        params = surf_params_to_wavenumber_params(**kwargs)
        kwargs.update(params)

    (kx,ky), (dkx, dky) = SpectrumCartesianGrid(**kwargs)

    kx = xr.DataArray(kx, dims='kx' , attrs = {'spacing':dkx, 'units':'rad/m'})
    ky = xr.DataArray(ky, dims='ky' , attrs = {'spacing':dky, 'units':'rad/m'})
    k = np.sqrt(kx**2+ky**2)
    phi = np.arctan2(ky,kx)

    #   Extend dataset to pi angle to avoid nan at the border near t=pi during interpolation
    nds = ds[{'phi':0}].copy()
    nds['phi'] = ds[{'phi':0}].phi.data+2.*np.pi
    nds = xr.concat((ds, nds), dim='phi')

    myspec= nds.interp(k=k, phi=phi).assign_coords(kx=kx, ky=ky)

    myspec.attrs.update({'dkx':dkx, 'dky':dky})
    myspec = myspec.where(np.sqrt(myspec.kx**2+myspec.ky**2)>min(ds.k).data, other=0.)# spectrum at wavenumbers smaller than the minimum one of the polar spectrum are set to zero
    myspec.attrs.pop('dphi', None)
    return myspec.drop_vars(('k','phi','dk'))


#  ******************************************************************************************************************************************************************
#  -------------------------------------------------------------------------BELOW ARE OLD VERSIONS OF CODE--------------------------------------------------------------------------------------------------------------
#  ******************************************************************************************************************************************************************


def from_wavenumber_old(**kwargs):
    """
    Generate an ocean spectrum
    inputs :
        ws : float : wind speed [m/s]-- optional : default 0. --
        wd : float : wind direction [deg]-- optional : default 0. --
        omega : float : wave age-- optional : default 0.84 --
        name : str : name of the wind sea spectrum ('elfouhaily', 'kudryavtsev')-- optional : default elfouhaily --
        cur : float : current speed [m/s] -- optional : default : 0. --
        curdir : float : current direction [deg] -- optional : default : 0. --
        sa : float or list of float : swell amplitude -- optional --
        sk : float or list of float : swell wavenumber -- optional --
        sd : float or list of float : swell direction [deg] in [-pi,pi] range -- optional --
        sas : float or list of float : swell azimuthal spreading -- optional default 0. --
        sks : float or list of float : swell wavenumber spreading -- optional default 0.  --
        sp : float or list of float : swell phase [deg] -- optional --
    ouputs:
        xarray : Cartesian Ocean Spectrum
    """
    (kx,ky), (dkx, dky) = SpectrumCartesianGrid(**kwargs)
    name = kwargs.pop('name', 'elfouhaily')
    ws = kwargs.pop('ws', 0.)
    wd = np.radians(kwargs.pop('wd', 0.))
    cur = kwargs.pop("cur", 0.)
    curdir = np.radians(kwargs.pop("curdir", 0.))

    if ('omega' not in kwargs) and ('fetch' not in kwargs):
        omega = 0.84
        fetch = np.inf
    elif ('omega' in kwargs) and ('fetch' in kwargs):
        raise ValueError('Both omega and fetch can not be defined')
    elif ('omega' in kwargs) and ('fetch' not in kwargs):
        omega = kwargs.pop('omega')
        fetch = 2.2e4*(np.arctanh((omega/0.84)**(-1./0.75)))**(2.5)
    else:
        fetch = kwargs.pop('fetch')
        omega = 0.84*np.tanh((fetch/(2.2e4))**0.4)**(-0.75)

    ds = xr.DataArray(np.zeros((len(kx), len(ky))), dims = ('kx','ky'), coords={'kx':kx, 'ky':ky} , attrs = {'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dkx':dkx, 'dky':dky, 'units':'m**4/rad**2', 'cur':cur, 'curdir':curdir}, name='spectrum magnitude')
    ds.kx.attrs.update({'units':'rad/m'})
    ds.ky.attrs.update({'units':'rad/m'})

    km = np.sqrt(ds.kx**2+ds.ky**2)
    tm = np.arctan2(ds.ky, ds.kx) # direction of wavenumber (relative to x axis)

    if name == 'elfouhaily' and ws>0.:
        ustar = findUstar(ws, 10.)
        curv = tony_omni(km, ws, ustar, omega)
        spread = tony_spread(km, ws, omega)
        #  spread = 0.; logging.debug('*******************WARNING : SPECTRUM IS FORCED OMNI !!**********************')
        magnitude = curv/(2*np.pi*km**4)*(1.+spread*np.cos(2*(tm-ds.wd)))
        #  ds.merge((xr.DataArray(magnitude, name = 'magnitude', dims=(['kx', 'ky']), attrs={'units':'m**4/rad**2'})).to_dataset(), join='left', inplace=True)
        ds+=magnitude
        set_directivity(ds, kwargs.pop('directivity', 'alongwind'))
    elif name == 'kudryavtsev':
        raise ValueError('kudryavtsev spectrum not implemented in xspectrum')
        #  self.magnitude = np.transpose(kudryavtsev(self.k, angle_mpipi(self.t-self.windDir), self.dt, self.windSpeed, self.fetch))/(self.km**4)
    else:
        logging.debug('No wind sea spectrum defined')

    if ('sa' in kwargs) or ('sk' in kwargs)  or ('sd' in kwargs):
        sa =  np.array(np.double(kwargs.pop('sa', 1.)), ndmin=1)
        sk =  np.array(np.double(kwargs.pop('sk', 2*np.pi/10.)), ndmin=1)
        sd =  np.array(np.double(kwargs.pop('sd', 0.)), ndmin=1)
        if len(sk) != len(sa):
            sa = np.tile(sa[0], len(sk))
        if len(sk) != len(sd):
            raise ValueError('sk and sd must have the same length')

        swell_magnitude = np.zeros(km.shape)
        jkx = list()
        jky = list()
        logging.debug('Swell spectrum wavenumbers and directions are redefined to :')
        for e in sk*np.cos(np.radians(sd)):
            jkx.append(np.argmin(np.abs(ds.kx-e)))
        for e in sk*np.sin(np.radians(sd)):
            jky.append(np.argmin(np.abs(ds.ky-e)))

        for i,(ikx,iky) in enumerate(zip(jkx, jky)):
            swell_magnitude[ikx, iky] = 0.5*(sa[i]/np.sqrt((ds.dkx*ds.dky)))**2
            logging.debug(('\tk{} = {} rad/m - A{} = {} m - azi{} = {} deg - lambda{} = {} m - f{} = {} Hz').format(i, np.round(km[ikx, iky].data,3), i, sa[i], i, np.round(np.rad2deg(np.arctan2(ds.ky[iky].data, ds.kx[ikx].data)),3), i, np.round(2*np.pi/km[ikx, iky].data,3), i, np.round(np.sqrt(9.81*km[ikx, iky].data)/(2.*np.pi),3)))
        ds+= swell_magnitude
    ds[np.where(np.isinf(ds))] = 0.
    ds[np.where(np.isnan(ds))] = 0.
    return ds