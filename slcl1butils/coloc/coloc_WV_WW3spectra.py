import os
import pdb

import xarray as xr
import datetime
import logging
import numpy as np
from ocean.xspectrum import from_ww3
from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum
def resampleWW3spectra_on_SAR_cartesian_grid(dsar):
    """

    Args:
        dsar: xarray.core.Dataset WV Level-1B IFREMER dataset

    Returns:

    """
    # basewv = os.path.basename(l1bwv)
    # datedt = datetime.datetime.strptime(basewv.split('-')[4],'%Y%m%dt%H%M%S')
    xs2tau = dsar['xspectra_2tau_Re']+1j*dsar['xspectra_2tau_Im']
    xs2tau = xs2tau.assign_coords({'k_rg':xs2tau['k_rg'].mean(dim=set(xs2tau['k_rg'].dims)-set(['freq_sample']), keep_attrs=True)}).swap_dims({'freq_sample':'k_rg', 'freq_line':'k_az'})
    xs2tau = symmetrize_xspectrum(xs2tau).squeeze(dim='2tau')
    xs2tau.attrs = dsar['xspectra_2tau_Re'].attrs
    # replace the half spectrum by a single variable, to save complexe values only possibility is zarr
    dsar = dsar.drop_vars(['xspectra_2tau_Re','xspectra_2tau_Im'])
    dsar = dsar.drop_dims(['freq_sample','freq_line'])
    dsar['xs2tau'] =  xs2tau
    start_date_dt = datetime.datetime.strptime(dsar.attrs['start_date'],'%Y-%m-%d %H:%M:%S.%f')
    # unit = basewv.split('-')[0]
    #unit_long = 'sentinel-1'+unit[-1]
    flag_ww3spectra_added = False
    flag_ww3spectra_found = False
    pathww3sp = get_ww3RAWspectra_path_from_date(datedt=start_date_dt)
    if pathww3sp:
        if os.path.exists(pathww3sp):
            flag_ww3spectra_found = True
            # print(pathww3sp,os.path.exists(pathww3sp))
            #ds_raw_ww3 = xr.open_dataset(pathww3sp)
            # dsar = xr.open_dataset(l1bwv)
            dk_az = np.diff(xs2tau['k_az'])
            dk_rg = np.diff(xs2tau['k_rg'])
            dk = (dk_rg.mean(),dk_az.mean())
            xs2tau["k_rg"].attrs['dkx'] = dk[0] #temporarily add dkx attrs
            xs2tau["k_az"].attrs["dky"] = dk[1] #temporarily add dky attrs
            kmax = (xs2tau['k_rg'].values.max(),xs2tau['k_az'].values.max())
            lonwv1 = dsar['longitude'].values
            latwv1 = dsar['latitude'].values
            # timewv1 = datetime.datetime.strptime(os.path.basename(l1bwv).split('-')[4],'%Y%m%dt%H%M%S')
            heading = dsar['ground_heading'].values
            logging.info('heading %s',heading)
            logging.info('timewv1 :%s',start_date_dt)
            rotate = 90+heading # deg clockwise wrt North
            logging.info('rotate:%s',rotate)
            extended_kx = np.hstack([np.array([-np.pi]),xs2tau['k_rg'].values,np.array([-np.pi])])
            extended_ky = np.hstack([np.array([-np.pi]),xs2tau['k_az'].values,np.array([-np.pi])])
            extended_kx = xr.DataArray(extended_kx,dims=['k_rg'],coords={'k_rg':extended_kx})
            extended_ky = xr.DataArray(extended_ky, dims=["k_az"], coords={"k_az": extended_ky})
            ds_ww3_cartesian = from_ww3(pathww3sp,kx=extended_kx,ky=extended_ky, dk=dk,kmax=kmax,strict='kmax',
                                        rotate=rotate,clockwise_to_trigo=True,
                                        lon=lonwv1,lat=latwv1,time=start_date_dt)
            ds_ww3_cartesian.attrs['source'] = 'ww3'
            #TODO: check kx ky names to be same as the one from intra burst ds
            dsar = xr.merge([dsar, ds_ww3_cartesian])
            flag_ww3spectra_added = True
    return dsar,flag_ww3spectra_added,flag_ww3spectra_found


def get_ww3RAWspectra_path_from_date(datedt):
    from slcl1butils.get_config import get_conf
    conf = get_conf()
    DIR_S1_WW3_RAWSPECTRA = conf['DIR_S1_WW3_RAWSPECTRA']
    baseww3file = 'MARC_WW3-GLOB-30M_'+datedt.strftime('%Y%m%d')+'_trck.nc'
    finalpath = None
    # yearmonth = datedt.strftime('%Y%m')
    pat = os.path.join(DIR_S1_WW3_RAWSPECTRA,baseww3file)
    logging.info('pat: %s',pat)
    if os.path.exists(pat):
        finalpath = pat

    return finalpath
