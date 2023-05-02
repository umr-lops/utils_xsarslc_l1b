import numpy as np
import xarray as xr


def compute_cwave_parameters(xs, save_kernel=False, kmax=2 * np.pi / 25, kmin=2 * np.pi / 600, Nk=4, Nphi=5):
    krg = xs.k_rg
    kaz = xs.k_az

    # XS decomposition Kernel define on krg and kaz of the product XS
    kernel = compute_kernel(krg, kaz, save_kernel=False, kmax=kmax, kmin=kmin, Nk=Nk, Nphi=Nphi)

    # Cross-Spectra Low Frequency Filtering  
    kk = np.sqrt((krg) ** 2. + (kaz) ** 2.)
    xxs = xs.where((kk > kmin) & (kk < kmax))

    # Cross-Spectra normalization    
    dkx = np.mean(np.diff(xxs.k_rg))
    dky = np.mean(np.diff(xxs.k_az))
    xxsm = np.sqrt(xxs.real ** 2. + xxs.imag ** 2.)
    xxsmn = xxsm / xxsm.sum(dim=['k_rg', 'k_az']) * dkx * dky

    # CWAVE paremeters compution
    cwave_parameters = ((kernel.cwave_kernel * xxsmn) * dky * dky).sum(dim=['k_rg', 'k_az']).rename(
        'cwave_params').to_dataset()

    return cwave_parameters


def compute_kernel(krg, kaz, save_kernel=False, kmax=2 * np.pi / 25, kmin=2 * np.pi / 600, Nk=4, Nphi=5):
    # Kernel Computation
    #
    gamma = 2
    a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
    a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (kmax ** 2 - gamma ** 2 * kmin ** 2)
    tmp = a1 * np.power(krg, 4) + a2 * krg ** 2 + kaz ** 2
    # alpha k
    alpha_k = 2 * ((np.log10(np.sqrt(tmp)) - np.log10(kmin)) / (np.log10(kmax) - np.log10(kmin))) - 1
    # alpha phi
    alpha_phi = np.arctan2(krg, kaz).rename(None)
    # eta
    eta = np.sqrt((2. * tmp) / ((krg ** 2 + kaz ** 2) * tmp * np.log10(kmax / kmin)))

    Gnk = xr.combine_by_coords([gegenbauer_polynoms(alpha_k, ik - 1, lbda=3 / 2.) * coef(ik - 1) * nu(
        alpha_k).assign_coords({'k_gp': ik}).expand_dims('k_gp') for ik in np.arange(Nk) + 1])
    Fnphi = xr.combine_by_coords(
        [harmonic_functions(alpha_phi, iphi).assign_coords({'phi_hf': iphi}).expand_dims('phi_hf') for iphi in
         np.arange(Nphi) + 1])

    Kernel = Gnk * Fnphi * eta
    Kernel.k_gp.attrs.update({'long_name': 'Gegenbauer polynoms dimension'})
    Kernel.phi_hf.attrs.update({'long_name': 'Harmonic functions dimension (odd number)'})

    _Kernel = Kernel.rename('cwave_kernel').to_dataset()
    if 'pol' in _Kernel:
        _Kernel = _Kernel.drop_vars('pol')
    _Kernel['cwave_kernel'].attrs.update({'long_name': 'CWAVE Kernel'})

    ds_G = Gnk.rename('Gegenbauer_polynoms').to_dataset()
    ds_F = Fnphi.rename('Harmonic_functions').to_dataset()
    ds_eta = eta.rename('eta').to_dataset()
    if 'pol' in ds_G:
        ds_G = ds_G.drop_vars('pol')
        ds_F = ds_F.drop_vars('pol')
        ds_eta = ds_eta.drop_vars('pol')

    Kernel = xr.merge([_Kernel, ds_G, ds_F, ds_eta])

    if (save_kernel):
        Kernel.to_netcdf('cwaves_kernel.nc')

    return Kernel


def gegenbauer_polynoms(x, nk, lbda=3 / 2.):
    C0 = 1
    if (nk == 0):
        return C0 + x * 0
    C1 = 3 * x
    if (nk == 1):
        return C1 + x * 0

    Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
                nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))
    Cnk = (1 / nk) * (2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda) - (
                nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda))

    return Cnk


def coef(nk):
    coef_frac = (nk + 3 / 2.) / ((nk + 2.) * (nk + 1.))
    return coef_frac


def nu(x):
    return np.sqrt(1 - x ** 2.)


def harmonic_functions(x, nphi):
    if (nphi == 1):
        Fn = np.sqrt(1 / np.pi) + x * 0
        return Fn

    # Even & Odd case
    if (nphi % 2 == 0):
        Fn = np.sqrt(2 / np.pi) * np.sin((nphi) * x)
    else:
        Fn = np.sqrt(2 / np.pi) * np.cos((nphi - 1) * x)

    return Fn
