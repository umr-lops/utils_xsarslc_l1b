import numpy as np
import logging
def apply_clockwise_to_trigo(ds):
    """
    Convert spectrum clockwise angular convention to trigo convention OR from trigo to clockwise convention.
    CAUTION : It does not change the original direction !!!
    It assume that provided spectrum has clockwise convention and that phi angles is on a [-pi, pi[ domain

    Args:
        ds (xarray): xPolarSpectrum
    """
    p = ds.phi.data
    ds = (ds.assign_coords(phi=np.append(p[0], np.flip(p[1:], axis=-1)))).sortby('phi')
    if 'convention' in ds.attrs:
        if ds.attrs['convention']=='trigo':
            ds.attrs['convention'] = 'clockwise'
        elif ds.attrs['convention']=='clockwise':
            ds.attrs['convention'] = 'trigo'
        else:
            logging.debug('unknown convention')
            pass
    if 'wd' in ds.attrs: ds.attrs.update({'wd':-ds.wd}) # wind convention has to be changed too
    if 'curdir' in ds.attrs: ds.attrs.update({'curdir':-ds.curdir}) # curdir convention has to be changed too
    return ds