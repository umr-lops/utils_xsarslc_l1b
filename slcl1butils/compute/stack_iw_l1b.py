import xarray as xr
import logging
import numpy as np
import os
from tqdm import tqdm

def get_a_valid_krg_vector(ds):
    notfound=True
    bb=0
    ts=0
    while notfound:
        k_rg = ds['k_rg'].isel(burst=bb, tile_sample=0)
        if np.isfinite(k_rg).all():
            notfound=False
        bb += 1
        ts += 1
    return k_rg

def get_index_wavenumbers(ds):
    #print(2 * np.pi / 0.16, 'm')
    k_rg = get_a_valid_krg_vector(ds)
    indkrg = np.where(abs(k_rg- 0.16) == np.amin(abs(k_rg - 0.16)))[0][0]
    # if True:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(k_rg,'.-')
    #     plt.axvline(x=indkrg)
    #     plt.show()
    #print('indkrg', indkrg)
    #print(ds['k_rg'].isel(freq_sample=indkrg).values)

    val_k_az = 0.14
    indkaz_up = np.where(abs(ds['k_az'] - val_k_az) == np.amin(abs(ds['k_az'] - val_k_az)))[0][0]
    #print('indkaz_up', indkaz_up)
    indkaz_bo = np.where(abs(ds['k_az'] + val_k_az) == np.amin(abs(ds['k_az'] + val_k_az)))[0][0]
    #print('indkaz_bo', indkaz_bo)
    #print(ds['k_az'].isel(freq_line=indkaz_bo).values)
    return indkaz_bo,indkaz_up,indkrg


def get_reference_wavenumbers(ds,indkaz_bo,indkaz_up,indkrg):
    """

    Parameters
    ----------
    ds xr.Dataset
    indkaz_bo int
    indkaz_up int

    Returns
    -------

    """

    #k_rg = ds['k_rg'].isel(burst=0,tile_sample=0)
    k_rg = get_a_valid_krg_vector(ds)
    #k_rg_ref = k_rg[k_rg < 0.16]
    k_rg_ref = k_rg.isel(freq_sample=slice(0,indkrg))

    #k_rg_ref = k_rg_ref.swap_dims({'freq_sample':'k_rg'})
    k_rg_ref = k_rg_ref.assign_coords({'freq_sample':np.arange(len(k_rg_ref))})
    k_rg_ref = xr.DataArray(k_rg_ref.values,coords={'freq_sample':np.arange(len(k_rg_ref))},dims=['freq_sample'])
    logging.info('k_rg_ref %s',k_rg_ref)
    k_az = ds['k_az']
    k_az_ref = k_az.isel(freq_line=slice(indkaz_bo, indkaz_up))

    #k_az_ref = k_az_ref.swap_dims({'freq_line': 'k_az'})
    k_az_ref = k_az_ref.assign_coords({'freq_line': np.arange(len(k_az_ref))})
    k_az_ref = xr.DataArray(k_az_ref.values, coords={'freq_line': np.arange(len(k_az_ref))}, dims=['freq_line'])
    return k_rg_ref,k_az_ref


def preprrocess(filee,indkrg,indkaz_bo,indkaz_up,k_rg_ref,k_az_ref,typee,tiffnumber):
    """

    Parameters
    ----------
    filee str full path
    indkrg int
    indkaz_bo int
    indkaz_up int
    k_rg_ref xr.DataArray
    k_az_ref xr.DataArray
    typee str intra or inter

    Returns
    -------
    dsu xr.Dataset

    """
    dsu = xr.open_dataset(filee,group=typee+'burst',engine='netcdf4',cache=False)
    #dsu['subswath'] = xr.DataArray(int(os.path.basename(filee).split('-')[1].replace('iw','')))
    tmpsubswath = xr.DataArray(int(os.path.basename(filee).split('-')[1].replace('iw','')))
    dsu = dsu.assign_coords({'subswath':tmpsubswath})
    dsu = dsu.isel(freq_sample=slice(0, indkrg), freq_line=slice(indkaz_bo, indkaz_up))
    dsu = dsu.assign_coords({'k_rg': k_rg_ref.values, 'k_az': k_az_ref.values})
    dsu = dsu.assign_coords({'tiff':tiffnumber})
    if 'xspectra_0tau_Re' in dsu:
        for tautau in range(3):
            dsu['xspectra_%stau' % tautau] = dsu['xspectra_%stau_Re' % tautau] + 1j * dsu['xspectra_%stau_Im' % tautau]
            dsu = dsu.drop(['xspectra_%stau_Re' % tautau, 'xspectra_%stau_Im' % tautau])
    dsu = dsu.drop(['xspectra_0tau','xspectra_1tau','var_xspectra_0tau','var_xspectra_1tau','var_xspectra_2tau'])
    dsu['xspectra_2tau'] = dsu['xspectra_2tau'].mean(dim=['2tau'])
    return dsu

def read_data_L1B(all_l1B, typee='intra',sens='Ascending'):
    """

    Parameters
    ----------
    all_l1B list of str
    typee str intra or inter [optional]
    sens str : Ascending or Descending

    Returns
    -------

    """
    tmp = []
    dsfirst = xr.open_dataset(all_l1B[0],group=typee+'burst',engine='netcdf4',cache=False)
    indkaz_bo,indkaz_up,indkrg = get_index_wavenumbers(dsfirst)
    k_rg_ref,k_az_ref = get_reference_wavenumbers(dsfirst, indkaz_bo, indkaz_up,indkrg=indkrg)
    dsfirst.close()
    print('start loop')
    # xx = partial(preprrocess,indkrg=indkrg,indkaz_bo=indkaz_bo,indkaz_up=indkaz_up,k_rg_ref=k_rg_ref,k_az_ref=k_az_ref,typee=typee)
    consolidated_list = []
    pbar = tqdm(range(len(all_l1B)))
    for ffi in pbar:
        pbar.set_description('')
        #for ffi,ff in enumerate(all_l1B):
        ff = all_l1B[ffi]
        tmpds = xr.open_dataset(ff,group=typee+'burst',engine='netcdf4')
        if 'freq_line' in tmpds.dims and tmpds.orbit_pass==sens and 'xspectra_2tau_Re' in tmpds:
            tmpds = preprrocess(ff, indkrg, indkaz_bo, indkaz_up, k_rg_ref, k_az_ref, typee, tiffnumber=ffi)
            consolidated_list.append(ff)
            tmp.append(tmpds)
        else:
            logging.debug('%s seems empty or not in right orbit direction',ff)

    # print('nb nc file to read',len(consolidated_list))
    #ds = xr.concat(tmp,dim='tiff')
    feinte_combine_by_coords = True
    if feinte_combine_by_coords:
        # feinte FN (on remplace des index de coords par des vrais coords) pcq xr.align ne gere pas bien cela
        tmp2 = [
            x.assign_coords({'tile_sample': range(x.sizes['tile_sample']), 'tile_line': range(x.sizes['tile_line'])}) for x
            in tmp]  # coords assignement is for alignment below
        dims_not_align = set()
        for x in tmp2:
            dims_not_align = dims_not_align.union(set(x.dims))
        dims_not_align = dims_not_align - set(['tile_sample', 'tile_line'])
        tmp3 = xr.align(*tmp2, exclude=dims_not_align,
                            join='outer')  # tile sample/line are aligned (thanks to their coordinate value) to avoid bug in combine_by_coords below
        # end feinte Fred
        ds = xr.combine_by_coords([tt.expand_dims('tiff').expand_dims('subswath') for tt in tmp3],combine_attrs='drop_conflicts')
    else:
        ds = xr.concat(tmp,dim='dummydim')
    return ds