import numpy as np
import xarray as xr
import logging
from slcl1butils.legacy_ocean.ocean.spectrum_private.spectra_functions import tony_omni,tony_spread,findUstar
#NG from shared.my_functions import *

gravity = 9.81

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
    return k,dk, phi,dphi #, 'km':km, 'tm':tm, 'Nk':Nk, 'Nphi':Nphi, 'kmin':kmin, 'kmax':kmax, 'ksampling':ksampling}


def from_surfparams(**kwargs):
    """
    return a xPolarSpectrum from surface parameters.
    input:
        dr : tuple of float : spatial sampling step [m]
        size : tuple of float : surface size [m]
    """
    size = kwargs.pop('size')
    dr = kwargs.pop('dr')
    ke = max(2*np.pi/dr[0], 2*np.pi/dr[1]) # sampling k
    kmin = min(2*np.pi/size[0], 2*np.pi/size[1])
    return from_wavenumber(kmin = kmin, kmax = ke, ksampling = 'lin', **kwargs) # kmax used in the spectrum has to be the sampling rate ke (twice the real kmax)


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
    #  self.reset_derivations()
    if directivity == 'alongwind':
        ds+=ds.roll(phi=ds.sizes['phi']//2, roll_coords=False).data
        angular_weight_func = 0.5*(np.tanh(dir_alpha*(np.mod(ds.phi-ds.wd+np.pi,2*np.pi)-np.pi+np.pi/2.))-np.tanh(dir_alpha*(np.mod(ds.phi-ds.wd+np.pi,2*np.pi)-np.pi-np.pi/2.)))
        # angular_weight_func = 0.5*(np.tanh(dir_alpha*(ds.phi+np.pi/2.))-np.tanh(dir_alpha*(ds.phi-np.pi/2.)))
        # angular_weight_func=angular_weight_func.roll(phi=int(ds.sizes['phi']/2+np.argmin(np.abs(ds.phi-ds.wd))), roll_coords=False).data
        ds*=(dir_weight*(angular_weight_func-0.5)+0.5)
    elif directivity == 'dissymmetry':
        ds = 2.*ds*(np.sin(((ds.phi-ds.wd)+np.pi)/2)**2)
    elif directivity in (None,'None') and ds.directivity=='None':
        pass
    elif directivity in (None,'None'):
        ds.attrs.update({'directivity':'None'})
    else:
        raise Exception('Wrong directivity for {} spectrum : {}'.format(ds.name, directivity))


def from_wavenumber(**kwargs):
    """
    Keyword Args:
        ws : wind speed
        wd : wind direction
        cur : float : current speed [m/s] -- optional : default : 0. --
        curdir : float : current direction [deg] -- optional : default : 0. --
        sa : sine amplitude
        sk : sine wavenumber
        sd : sine direction [deg]
        sp : sine phase
        omega : inverse wave age
    """
    k, dk, phi, dphi = SpectrumPolarGrid(**kwargs)
    name = kwargs.pop('name', 'elfouhaily')
    ws = kwargs.pop('ws', 0.)
    wd = np.radians(kwargs.pop('wd', 0.))
    wd = np.arctan2(np.sin(wd), np.cos(wd))
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

    ds = xr.DataArray(np.zeros((len(k), len(phi))), dims = ('k','phi'), coords={'k':k, 'phi':phi} , attrs = {'omega':omega,'ws':ws,'wd':wd,'fetch':fetch, 'dphi':dphi, 'units':'m^4', 'cur':cur, 'curdir':curdir}, name='spectrum magnitude')
    # ds.attrs.update({'dk':dk})
    dk = xr.DataArray(dk, dims='k') if len(np.array(dk, ndmin=1))>1 else xr.DataArray(dk)
    ds = ds.assign_coords(dk=dk)
    ds.k.attrs.update({'units':'rad/m'})
    ds.phi.attrs.update({'units':'radians'})

    if name == 'elfouhaily' and ws>0.:
        ustar = findUstar(ws, 10.)
        curv = tony_omni(ds.k, ws, ustar, omega)
        spread = tony_spread(ds.k, ws, omega)
        # spread = 0.; logging.debug('*******************WARNING : SPECTRUM IS FORCED OMNI !!**********************')
        magnitude = curv/(2*np.pi*ds.k**4)*(1.+spread*np.cos(2*(ds.phi-ds.wd)))
        ds+=magnitude
        set_directivity(ds, kwargs.pop('directivity', 'alongwind'))
    elif name == 'kudryavtsev':
        raise ValueError('kudryavtsev spectrum not implemented in xPolarSpectrum')
        #  self.magnitude = np.transpose(kudryavtsev(self.k, angle_mpipi(self.t-self.windDir), self.dphi, self.windSpeed, self.fetch))/(self.km**4)
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

        swell_magnitude = np.zeros(ds.shape)
        jk = list()
        jt = list()
        logging.debug('Sine wavenumbers and directions are (re)defined to :')
        for e in sk:
            jk.append(np.argmin(np.abs(ds.k.data-e)))
        for e in np.radians(sa):
            jt.append(np.argmin(np.abs(ds.phi.data-np.arctan2(np.sin(e), np.cos(e)))))
        for i, (ik,it) in enumerate(zip(jk, jt)):
            dk = ds.dk[ik] if len(np.array(ds.dk, ndmin=1))>1 else ds.dk
            swell_magnitude[ik, it] = 0.5*(sa[i]/np.sqrt((ds.k[ik]*dk*ds.dphi)))**2
            logging.debug(('\tk{} = {} rad/m - A{} = {} m - azi{} = {} deg - lambda{} = {} m - f{} = {} Hz').format(i, round(float(ds.k[ik]),3), i, sa[i], i, round(np.rad2deg(float(ds.phi[it])),3), i, round(2*np.pi/float(ds.k[ik]),3), i, round(np.sqrt(gravity*float(ds.k[ik]))/(2.*np.pi),3)))

        ds+= swell_magnitude
    ds[np.where(np.isinf(ds))] = 0.
    ds[np.where(np.isnan(ds))] = 0.
    return ds


def compute_moments(ds, moments, *, inplace=True):
    """
    Evaluate numerical integration of the spectrum moments:  \int (k*cos(a))**ni (k*sin(a))**nj (omega**nw) spec(k) k dk dphi where ni, nj, nw are the three digits in the moment list and 'a' the angular direction [degrees] relative to original direction

    Args:
        moments : tuple composed of 4 parameters: 3 exponents ni,nj and nw and one angular direction. ex : (1,0,0,90.). If angle is not provided (ex: moment=(1, 0, 1)) angle "a" is assumed to be zero

    Keyword Args:
        inplace (bool, optional): if True, results are added to the original dataset. If False, result is returned
    """
    #  if 'moments' not in ds.attrs : ds.attrs.update({'moments':{}})
    if type(moments)==tuple: moments = [moments] # This is different from list(moment)
    moments = [m if len(m)==4 else (*m,0.) for m in moments]
    moments = [(int(m[0]), int(m[1]), int(m[2]), float(m[3]))  for m in moments] # reevaluate moment in any case
    moments = dict.fromkeys(moments)

    w = np.sqrt(gravity*ds.k*(1.+(ds.k/363.2)**2))
    dk = ds.dk #if len(np.array(ds.dk, ndmin=1))==1 else xr.DataArray(ds.dk, dims = ('k',))

    for key in moments.keys():
        ni, nj, nw, a = key
        moments[key]=(((ds.k*np.cos(ds.phi-np.radians(a)))**int(ni)*(ds.k*np.sin(ds.phi-np.radians(a)))**int(nj)*(w**int(nw))*ds*ds.k*dk).sum()*ds.dphi).data
    if inplace:
        if 'moments' not in ds.attrs: ds.attrs['moments']={}
        ds.attrs['moments'].update(moments)
    else:
        return moments


def compute_swh(ds):
    compute_moments(ds, (0,0,0,0))
    ds.attrs.update({'swh':4.*np.sqrt(ds.moments[(0,0,0,0.)])})
    return ds.swh

def compute_mss(ds):
    compute_moments(ds, [(2,0,0,0.), (0,2,0,0.)])
    ds.attrs.update({'mss':ds.moments[(2,0,0,0.)]+ds.moments[(0,2,0,0.)]})
    return ds.mss

def from_ww3(ds, *, station_index=None, time_index=None, rotate=0., shortWavesFill = False, clockwise_to_trigo = False, **kwargs):
    """
    Import a WW3 spectrum as an xPolar Spectrum

    Args:
        ds (str or xrarray.Dataset) : the path of the WW3 spectrum or  its imported version by xarray

    Keyword Args:
        lon (float, optional) : If provided together with lat and time, it will look to the closest WW3 spectrum in the dataset
        lat (float, optional) : If provided together with lon and time, it will look to the closest WW3 spectrum in the dataset
        time (float, optional) : If provided together with lon and lat, it will look to the closest WW3 spectrum in the dataset
        station_index (int, optional) : station index of WW3 spectrum
        time_index (int, optional) : time index of WW3 spectrum
        rotate (float, optional): apply of rotation on the spectrum of angle rotate [deg]. Same angle convention of the spectrum
        shortWavesFill (bool, optional): Fill the high frequency part of the spectrum
        kwargs : other arguments to be passed to apply_shortWavesFill()
    """
    if isinstance(ds, str):
        specWW3 = xr.open_dataset(ds)
    elif isinstance(ds, xr.Dataset):
        specWW3 = ds
    else:
        raise ValueError('Invalid WW3 spectrum type')
    kmin = (2.*np.pi*min(specWW3.frequency).data)**2/gravity
    kmax = (2.*np.pi*max(specWW3.frequency).data)**2/gravity

    k,dk, phi,dphi = SpectrumPolarGrid(Nk=specWW3.sizes['frequency'], Nphi= specWW3.sizes['direction'], kmin=kmin, kmax=kmax, ksampling = 'log')

    angle_mpipi = lambda t: np.arctan2(np.sin(t), np.cos(t))

    if specWW3.direction.units == 'degree':
        dirww3 = angle_mpipi(np.radians(specWW3.direction))
    elif specWW3.direction.units == 'rad':
        dirww3 = angle_mpipi(specWW3.direction)
    else:
        raise ValueError('Unknown angle unit: {}'.format(specWW3.direction.units))

    id = np.argsort(dirww3) # direction vector is not monotonous. This line gives reordered direction indices over [-pi,pi[ interval
    ww3_kwargs={'direction':id.data}

    if not np.allclose(dirww3[id.data], phi):
        raise ValueError('WW3 angles do not match -pi,pi interval')

    if (not time_index):
        if ('lon' in kwargs) and ('lat' in kwargs) and ('time' in kwargs):
            time_index = find_closest_ww3(ds, kwargs['lon'], kwargs['lat'], kwargs['time'])
        else:
            time_index = 0
    if not station_index:
        station_index = 0

    if 'time' in specWW3.efth.dims:
        ww3_kwargs.update({'time':time_index})
        logging.debug('spectrum imported at time index {} :{} @ (lon:{}, lat:{})'.format(time_index, specWW3.time[{'time':time_index}].data, specWW3[ww3_kwargs]['longitude'].data, specWW3[ww3_kwargs]['latitude'].data))
    if 'station' in specWW3.efth.dims:
        ww3_kwargs.update({'station':station_index})
        logging.debug('spectrum imported at station index {} :{}'.format(station_index, specWW3.station[{'station':station_index}].data))

    myspec = (2.*np.pi)/np.sqrt(gravity*k[:,np.newaxis])*gravity*specWW3.efth[ww3_kwargs]/(8.*np.pi**2*k[:,np.newaxis])
    myspec = myspec.assign_coords(frequency=k)
    # myspec = myspec.assign_coords(direction=dirww3[id.data])
    myspec = myspec.assign_coords(direction=phi)
    myspec = myspec.rename({'frequency':'k'})
    myspec = myspec.rename({'direction':'phi'})
    myspec.attrs.update(specWW3.attrs)
    dk = xr.DataArray(dk, dims='k') if len(np.array(dk, ndmin=1))>1 else xr.DataArray(dk)
    myspec = myspec.assign_coords(dk=dk)
    # myspec.attrs.update({'dk':dk, 'dphi':dphi})
    myspec.attrs.update({'dphi':dphi})
    file_path = specWW3.encoding.get('source', None)
    myspec.attrs.update({'pathWW3':file_path})
    myspec.attrs.update({'longitude':specWW3[ww3_kwargs]['longitude'].data, 'latitude':specWW3[ww3_kwargs]['latitude'].data,'time_index':time_index})

    try:
        myspec.attrs.update({'wd':np.radians(specWW3[ww3_kwargs].wnddir.data+180), 'ws':specWW3[ww3_kwargs].wnd.data})# wnddir (in WW3 files) is the direction the wind is coming from
    except:
        logging.debug('No wind information found in file !')
        # if 'ws' in kwargs and 'wd' in kwargs:
        # myspec.attrs.update({'wd':np.radians(kwargs.pop('wd')), 'ws':kwargs.pop('ws')})
        # logging.debug('wind speed and direction have been set to provided ones: {} m/s, {} deg from North-clockwise.'.format(myspec.ws, np.degrees(myspec.wd)))
        # else:


    if clockwise_to_trigo and (rotate!=0.):logging.debug('Rotation angle has to be in clockwise convention because clockwise-to-trigo transformation is applied BEFORE the rotation. (Apply a minus sign on rotation angle to do it in clockwise convention).')
    if clockwise_to_trigo:
        myspec = swap_clockwise_trigo(myspec)
        if 'wd' in kwargs: kwargs.update({'wd':-kwargs['wd']})
        if 'curdir' in kwargs: kwargs.update({'curdir':-kwargs['curdir']})
        if 'sd' in kwargs: kwargs.update({'sd':-kwargs['sd']})
    if rotate!=0.:
        myspec=apply_rotation(myspec, rotate)
        if 'wd' in kwargs: kwargs.update({'wd':kwargs['wd']+rotate})
        if 'curdir' in kwargs: kwargs.update({'curdir':kwargs['curdir']+rotate})
        if 'sd' in kwargs: kwargs.update({'sd':kwargs['sd']+rotate})
    if shortWavesFill: myspec = apply_shortWavesFill(myspec, **kwargs)

    return myspec


def apply_shortWavesFill(ds, *, kcut = 4000., **kwargs):
    """
    Fill the provided spectrum magnitude with short waves spectrum.
    wind speed and direction used to define the high wavenumber part of the spectrum has to be in the provided spectrum: ds.ws and ds.wd [radians]

        Args:
            ds (xarray) : xPolarSpectrum with a log scale of wavenumber

         Keyword Args:
            ki (float, optional) : lower wavenumber bound of the connection between the two spectra. ki has to be smaller than max(ds.k). Default is the last wavenumber of the provided spectrum (ds)
            ke (float, optional) : higher wavenumber bound of the connection between the two spectra. Default is 5 ki. Raise error if ke > kcut
            kcut (float, optional) : maximum wavenumber to which you want to fill. Raise error if ke > kcut
            ws (float, optional) : wind speed to use for short wave filling (ws will be updated in spec)
            wd (float, optional) : wind direction to use for short wave filling (wd will be updated in spec)
            kwargs (dict, optional): other parameters passed to from_wavenumber() used for short wave spectrum
    """
    import ocean, xarray

    if ds.k.dims!=ds.dk.dims:
        raise ValueError('Provided spectrum has to be on a wavenumber log scale')
    elif np.allclose(ds.dk[0], ds.dk):
        raise ValueError('Provided spectrum has to be on a wavenumber log scale')
    else:
        pass

    if 'ki' in kwargs:
        ki = kwargs.pop('ki')
        if (ki>max(ds.k).data): raise ValueError('You must respect ki < ke < kcut and ki < max k of your dataset')
    else:
        ki = ds.k[-1].data

    ke = kwargs.pop('ke', 5*ki)

    if (ke>=kcut) or (ki>=ke): raise ValueError('You must respect ki < ke < kcut and ki < max k of your dataset')

    dlogk = np.log10(ds.k[1])-np.log10(ds.k[0])
    Nk = int(np.rint((np.log10(kcut)-np.log10(ds.k.min()))/dlogk))+1 # cannot start from ki, because ki would not be on the continuing log scale of k

    if 'ws' in  kwargs:
        ds.attrs.update({'ws':kwargs.pop('ws')})
        logging.debug('Wind speed replaced by provided one ({} m/s).'.format(ds.ws))

    if 'wd' in  kwargs:
        ds.attrs.update({'wd':np.radians(kwargs.pop('wd'))})
        logging.debug('Wind direction replaced by provided one ({} deg in local convention).'.format(np.degrees(ds.wd)))

    spec = ocean.xPolarSpectrum.from_wavenumber(Nk = Nk, Nphi = ds.sizes['phi'], kmin = min(ds.k).data, kmax=kcut, ws = ds.ws, wd = np.degrees(ds.wd),**kwargs)

    specshort = spec.where(spec.k>ke, drop = True)
    kmiddle = spec.where(np.logical_and(spec.k>ki, spec.k<=ke), drop=True).k

    dims_to_drop = list(set(ds.coords.keys()).intersection(set(['time','station']))) # dimensions to be dropped. Using ds.dims no dor work because singleton dimensions are not considered as dimensions.
    ds = ds.drop(dims_to_drop)
    ds = ds.where(ds.k<=ki, drop = True)

    xe = np.log(specshort.k[0])
    xi = np.log(ds.k[-1])
    ye = np.log(specshort[{'k':0}])
    yi = np.log(ds[{'k':-1}])
    specmiddle = np.exp((ye-yi)/(xe-xi)*(np.log(kmiddle)-xi)+yi)

    spec_filled = xarray.concat((ds, specmiddle, specshort), dim='k')
    Nk = spec_filled.sizes['k']
    dk = np.log(10)*(np.log10(spec_filled.k.max())-np.log10(spec_filled.k.min()))/(Nk-1)*spec_filled.k
    spec_filled.attrs.update(spec.attrs)
    spec_filled.attrs.update(ds.attrs)
    return spec_filled



def from_xCartesianSpectrum(ds, *, Nphi=120, ksampling='log', **kwargs):
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

    k, dk, phi, dphi = SpectrumPolarGrid(**kwargs)

    k = xr.DataArray(k, dims='k')
    phi = xr.DataArray(phi, dims='phi')

    kx = k*np.cos(phi)
    ky = k*np.sin(phi)

    #  myspec= np.log(ds).interp(kx=kx, ky=ky).assign_coords(k=k, phi=phi)
    #  myspec=np.exp(myspec)

    # fill_value = None below allows extrapolation when kmax is slightly greater to maximum available kmax (numerical precision)
    myspec = ds.interp(kx=kx, ky=ky, assume_sorted=True, kwargs={'fill_value':None}).assign_coords(k=k, phi=phi)

    myspec = myspec.assign_coords({'dk':xr.DataArray(dk, dims='k')}) if ksampling=='log' else myspec.assign_coords({'dk':xr.DataArray(dk)})
    myspec.attrs.update({'dphi':dphi})
    myspec.attrs.pop('dkx', None)
    myspec.attrs.pop('dky', None)

    return myspec.drop(('kx','ky'))


def apply_rotation(ds, angle):
    """
    Rotate spectrum. Angle is assumed to be in the same angle convention than ds

    Args:
        ds (xarray): xPolarSpectrum
        angle (float): rotation angle [deg]. Same convention than the spectrum)
    """
    angle_mpipi = lambda t: np.arctan2(np.sin(t), np.cos(t))
    ds = ds.roll(phi=int(np.rint(np.radians(angle)/ds.dphi)), roll_coords=False)
    if 'wd' in ds.attrs: ds.attrs.update({'wd':angle_mpipi(ds.wd+np.radians(angle))}) # wind direction has to be rotated too
    if 'curdir' in ds.attrs: ds.attrs.update({'curdir':angle_mpipi(ds.curdir+np.radians(angle))}) # current direction has to be rotated too
    return ds


def apply_clockwise_to_trigo(ds):
    import warnings
    warnings.warn('Deprecated function. Please use swap_clockwise_trigo')
    return swap_clockwise_trigo(ds)


def swap_clockwise_trigo(ds):
    """
    swap spectrum clockwise/trigo angular conventions in both ways !
    CAUTION : It does not change the original direction !!!
    It assumes that provided spectrum have phi angles on [-pi, pi[ domain

    Args:
        ds (xarray): xPolarSpectrum
    """
    p = ds.phi.data
    ds = (ds.assign_coords(phi=np.append(p[0], np.flip(p[1:], axis=-1)))).sortby('phi')
    if 'wd' in ds.attrs: ds.attrs.update({'wd':-ds.wd}) # wind convention has to be changed too
    if 'curdir' in ds.attrs: ds.attrs.update({'curdir':-ds.curdir}) # curdir convention has to be changed too
    return ds


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    import numpy as np
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def find_closest_ww3(ww3_path, lon, lat, time):
    """
    Find spatio-temporal closest point in ww3 file

    Args:
        ww3_path (str): path to the WW3 file
        lon (float): longitude of interest
        lat (float): latitude of interest
        time (datetime or tuple of int): Tuple of the form (year, month, day, hour, minute, second)


    Returns:
        (int): time indice of spatio-temporal closest point
    """
    from datetime import datetime
    import numpy as np
    import xarray as xr
    ww3spec = xr.open_dataset(ww3_path)
    if type(time)==datetime:
        mytime = np.datetime64(time)
    elif isinstance(time,np.datetime64):
        mytime = time
    else:
        mytime = np.datetime64(datetime(*time))
    time_dist = np.abs(ww3spec.time-mytime)
    isel = np.where(time_dist==time_dist.min())
    spatial_dist = haversine(lon, lat, ww3spec[{'time':isel[0]}].longitude, ww3spec[{'time':isel[0]}].latitude)
    #  logging.debug(spatial_dist)
    myind = isel[0][np.argmin(spatial_dist.data)]

    logging.debug('Wave Watch III closest point @ {} km and {} minutes'.format(np.round(haversine(lon, lat, ww3spec[{'time':myind}].longitude, ww3spec[{'time':myind}].latitude).data,2), (ww3spec[{'time':myind}].time-mytime).data/(1e9*60)))

    return myind

def to_WW3(ds):
    """
    Transform xPolarSpectrum into WW3 format. Angle convention "phi" has to be trigonometric, from EAST. Direction
    is the direction the waves, wind, current are going to.

    Args:
        ds (xarray.DataArray): an xPolarSpectrum.
    Returns:
        (xarray.Dataset) : A WW3 spectrum-like
    """
    f = np.sqrt(gravity*ds.k)/(2.*np.pi)
    E = f/gravity*8.*np.pi**2*ds.k*ds
    E = xr.DataArray(E.transpose('k','phi').data, dims=('frequency', 'direction'), coords={'frequency':f.data, 'direction':ds.phi.data})
    E = xr.concat([E, E[{'direction':0}].assign_coords(direction=E.direction[-1]+np.diff(E.direction)[0])], dim='direction')
    E = xr.concat([E, E[{'frequency':-1}].assign_coords(frequency=E.frequency[-1]+np.diff(E.frequency)[-1])], dim='frequency')
    f,df, phi,dphi = SpectrumPolarGrid(Nk=32, Nphi= 24, kmin=0.0373, kmax=0.7159492, ksampling = 'log')
    E = E.interp(frequency=f, direction=phi, assume_sorted=True)
    # Below is angle conversion: from EAST clockwise TO From North clockwise
    E = E.assign_coords(direction = 90-np.degrees(E.direction))
    E = E.roll({'direction':12}, roll_coords=True)
    E = E.assign_coords(direction=E.direction.where(E.direction>-np.degrees(dphi)/2., E.direction+360))
    E = E.to_dataset(name='efth')

    if 'ws' in ds.attrs:
        E = E.merge(xr.DataArray(np.array(ds.ws, ndmin=1), dims='time', name='wnd'))
    if 'wd' in ds.attrs: # angle conversion: from EAST clockwise TO From North clockwise and wind coming from TO wind going to
        E = E.merge(xr.DataArray(np.array(90.-np.degrees(ds.wd)-180, ndmin=1), dims='time', name='wnddir'))
    if 'cur' in ds.attrs:
        E = E.merge(xr.DataArray(np.array(ds.cur, ndmin=1), dims='time', name='cur'))
    if 'curdir' in ds.attrs: # angle conversion: from EAST clockwise TO From North clockwise and current coming from TO current going to
        E = E.merge(xr.DataArray(np.array(90.-np.degrees(ds.curdir)-180, ndmin=1), dims='time', name='curdir'))

    E = E.merge(xr.DataArray(np.array(ds.attrs.get('longitude', np.nan), ndmin=1), dims='time', name='longitude'))
    E = E.merge(xr.DataArray(np.array(ds.attrs.get('latitude', np.nan), ndmin=1), dims='time', name='latitude'))

    E.efth.data = np.nan_to_num(np.clip(E.efth, a_min=0., a_max=None)) # Ensure positive values and convert nan to zeros
    return E





