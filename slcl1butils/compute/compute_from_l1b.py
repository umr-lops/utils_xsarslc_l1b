import xarray as xr


def compute_xs_from_l1b(_file, burst_type='intra', time_separation='2tau'):
    # Reading the l1b file
    # Loading the specified burst group
    # dt = datatree.open_datatree(_file)
    # Version 1.4
    # ds = xr.open_dataset(_file,group=burst_type+'burst_xspectra')
    # Version 1.4a
    if 'wv' in _file:
        ds = xr.open_dataset(_file, group=burst_type)
    else:
        ds = xr.open_dataset(_file, group=burst_type + 'burst')

    # ds = dt[burst_type+'burst_xspectra'].to_dataset()

    if burst_type == 'intra' or burst_type == '':
        xsRe = ds['xspectra_' + time_separation + '_Re']  # +1j*ds_intra['xspectra_1tau_Im']).mean(dim=['1tau'])
        xsIm = ds['xspectra_' + time_separation + '_Im']
        if time_separation == '2tau':
            xsRe = xsRe.squeeze('2tau')
            xsIm = xsIm.squeeze('2tau')
        if time_separation == '1tau':
            xsRe = xsRe.mean(dim=['1tau'])
            xsIm = xsIm.mean(dim=['1tau'])

    elif burst_type == 'inter':
        xsRe = ds['xspectra_Re']  # +1j*ds_inter['xspectra_Im']
        xsIm = ds['xspectra_Im']
    else:  # WV case
        raise Exception('not handle case')

    xs = xsRe + 1j * xsIm

    # Remove unique dimensions
    # xs=xs.squeeze()
    # convert the wavenumbers variables in range and azimuth into coordinates after selection of one unique vector without any other dimsension dependency
    xs = xs.assign_coords({'k_rg': xs.k_rg.mean(dim=['tile_sample', 'burst'])})
    #  Replace the dimesion name for frequencies
    xs = xs.swap_dims({'freq_sample': 'k_rg', 'freq_line': 'k_az'})
    # Bug Fix to define the wavenumber in range direction. 
    xs.k_rg.attrs.update({'long_name': 'wavenumber in range direction', 'units': 'rad/m'})

    return xs, ds


def compute_xs_from_l1b_wv(_file, time_separation='2tau'):
    # Reading the l1b file
    # Loading the specified burst group
    # dt = datatree.open_datatree(_file)
    # Version 1.4
    # ds = xr.open_dataset(_file,group=burst_type+'burst_xspectra')
    # Version 1.4a
    ds = xr.open_dataset(_file, group='')


    # ds = dt[burst_type+'burst_xspectra'].to_dataset()


    xsRe = ds['xspectra_' + time_separation + '_Re']  # +1j*ds_intra['xspectra_1tau_Im']).mean(dim=['1tau'])
    xsIm = ds['xspectra_' + time_separation + '_Im']
    if time_separation == '2tau':
        xsRe = xsRe.squeeze('2tau')
        xsIm = xsIm.squeeze('2tau')
    if time_separation == '1tau':
        xsRe = xsRe.mean(dim=['1tau'])
        xsIm = xsIm.mean(dim=['1tau'])

    xs = xsRe + 1j * xsIm

    # Remove unique dimensions
    # xs=xs.squeeze()
    # convert the wavenumbers variables in range and azimuth into coordinates after selection of one unique vector without any other dimsension dependency
    #xs = xs.assign_coords({'k_rg': xs.k_rg})
    #  Replace the dimesion name for frequencies
    xs = xs.swap_dims({'freq_sample': 'k_rg', 'freq_line': 'k_az'})
    # Bug Fix to define the wavenumber in range direction.
    xs.k_rg.attrs.update({'long_name': 'wavenumber in range direction', 'units': 'rad/m'})

    return xs, ds