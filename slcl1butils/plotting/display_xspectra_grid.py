from matplotlib import pyplot as plt
import os
import numpy as np
import logging
import sys
from matplotlib import colors as mcolors
from slcl1butils.conversion_polar_cartesian import from_xCartesianSpectrum
from slcl1butils import spectrum_clockwise_to_trigo
from slcl1butils import spectrum_rotation
from spectra_plot_circles_wavenumbers import circle_plot
def draw_signal_and_grid(slc,splitting_image,one_tiff,variable_displayed='digital_number'):
    """

    :param slc: xarray Dataset with digital number from SLC S1 prod
    :param splitting_image: dict with the slices of each sub domain
    :param one_tiff: (str) path of the tiff analyzed
    :return:
    """

    plt.figure(figsize=(15,10),dpi=100)
    plt.title(os.path.basename(one_tiff),fontsize=20)



    if variable_displayed=='digital_number':
        sli = slice(None,None,5)
        vals = slc['digital_number'].values.squeeze()[sli,sli]
        azi_coords = slc['azimuth'].values[sli]
        range_coords = slc['range'].values[sli]
        print(vals.shape,type(vals),vals.dtype)
        vmin = np.percentile(vals.real.ravel(),15)
        vmax = np.percentile(vals.real,85)
        print(vmin,vmax)
        print(vals.real.max(),vals.real.min())
        levels = np.linspace(vmin,vmax,40)
        plt.contourf(range_coords,azi_coords,vals.real,levels,cmap='Greys_r')  # ,vmin=-1500,vmax=1500
    else:
        step = 10
        sys.path.append('/home1/datahome/agrouaze/git/cerbere')
        sys.path.append('/home1/datahome/agrouaze/git/sar/')
        from cerbere.mapper.safegeotifffile import SAFEGeoTiffFile
        from sar.data.sarimage import SARImage
        imagefile = SAFEGeoTiffFile(url=one_tiff)
        surcouche = SARImage(imagefile)
        #pixelsspacing = surcouche.meters2pixels(1000) #1km grid spacing
        #lons = surcouche.get_data('lon',spacing=pixelsspacing)
        sigma0cerb = surcouche.get_data('sigma0')
        print('coords atrack',slc['atrack'].values)
        logging.info('expected shape (xsar): %s',slc['sigma0'].shape)
        logging.info('current shape(cerbre) : %s ',sigma0cerb.shape)
        #sigma0 = slc['sigma0'].values.squeeze()[: :step,: :step]
        diffatrack = slc['sigma0'].values.shape[1]-sigma0cerb.shape[0]
        diffxtrack = slc['sigma0'].values.shape[2]-sigma0cerb.shape[1]
        logging.info('padding vals: %s %s',diffatrack,diffxtrack)
        if diffatrack<0 and diffxtrack>0:
            sigma0cerb_pad = np.pad(sigma0cerb[0:diffatrack,:],((0,0),(0,diffxtrack)))
        elif diffatrack>0 and diffxtrack<0  :
            sigma0cerb_pad = np.pad(sigma0cerb[:,0:diffxtrack],((0,diffatrack),(0,0)))
        else:
            sigma0cerb_pad = np.pad(sigma0cerb,((0,diffatrack),(0,diffxtrack)))
        sigma0cerb_pad = sigma0cerb_pad.reshape((1,sigma0cerb_pad.shape[0],sigma0cerb_pad.shape[1]))
        logging.info('sigma0cerb_pad %s mean = %s',sigma0cerb_pad.shape,np.nanmean(sigma0cerb_pad))
        slc['sigma0'].values = sigma0cerb_pad #on remplace par cerbere (temporairement)
        #
        sigma0  = slc['sigma0'].values.squeeze()[: :step,: :step]
        sigma0[sigma0==0] = np.nan
        #sigma0 = 10. * np.log10(sigma0)
        maskfinite = np.isfinite(sigma0)
        vmin = np.percentile(sigma0[maskfinite].ravel(),1)
        vmax = np.percentile(sigma0[maskfinite].ravel(),99)
        #vmin= .1
        #vmax = .15
        #vmin=-20
        #vmax = 0
        print('vmin',vmin,'vmax,',vmax)
        levels = np.linspace(vmin,vmax,40)
        #test with sigma0 NICE display

        incidence = slc['incidence'].values[: :step,: :step]
        polarisation = slc.attrs['pols'] #expect VV
        logging.info('incidence = %s pol : %s',incidence.shape,polarisation)
        #roughness = compute_roughness(sigma0,incidence,polarisation)
        roughness = sigma0
        logging.info('roughness : %s nb NaN : %s',roughness.shape,np.isnan(roughness).sum())
        logging.info('mea roughness : %s std %s',np.nanmean(roughness),np.nanstd(roughness))
        plt.contourf(slc['xtrack'].values[::step],slc['atrack'].values[::step],roughness,levels,cmap='Greys_r')
    cb = plt.colorbar()
    cb.set_label(variable_displayed)
    for lab in splitting_image :
        rect = splitting_image[lab]
        subset = slc['digital_number'].isel(azimuth=rect['azimuth'],range=rect['range'])
        X,Y = np.meshgrid(subset.range,subset.azimuth)

        # plt.plot(Y.ravel()[::10],X.ravel()[::10],'.',label=lab)
        # get corners
        range_positions_corners_ind = np.array(
            [rect['range'].start,rect['range'].start,rect['range'].stop,rect['range'].stop,rect['range'].start])
        range_positions_corners = slc['range'].values[range_positions_corners_ind]
        #print('range_positions_corners',range_positions_corners)
        azimuth_positions_corners_ind = np.array(
            [rect['azimuth'].start,rect['azimuth'].stop,rect['azimuth'].stop,rect['azimuth'].start,
             rect['azimuth'].start])
        azimuth_positions_corners = slc['azimuth'].values[azimuth_positions_corners_ind]
        plt.plot(range_positions_corners,azimuth_positions_corners,'.-',label=lab,lw=5)
        mean_x = X.mean()
        mean_y = Y.mean()
        plt.annotate('#%i' % lab,(mean_x,mean_y),weight='bold',fontsize=20)
    plt.legend(ncol=2)
    plt.xlabel('range',fontsize=20)
    plt.ylabel('azimuth',fontsize=20)
    plt.grid(True)
    plt.show()


def display_xspectra_per_domain(allspecs_per_sub_domain,part='Re'):
    """

    :param allspecs_per_sub_domain:
    :param (str): Re or Im, component of complex x spectra to display
    :return:
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("",["white","violet","mediumpurple","cyan","springgreen","yellow",
                                                         "red"])
    ncols = int(np.sqrt(len(allspecs_per_sub_domain)))
    nrows = 0
    while nrows*ncols<len(allspecs_per_sub_domain):
        nrows += 1
        #print('plus')
    # PuOr = mcolors.LinearSegmentedColormap.from_list("", ["darkgoldenrod","white","purple"])
    PuOr = plt.get_cmap('PuOr')
    indices = np.arange(ncols*nrows)[: :-1].reshape(ncols,nrows)

    final_inds = indices[: :-1].T


    fig = plt.figure(figsize=(8 *ncols ,6 *nrows ),dpi=70)
    NCOLS = ncols
    NROWS = nrows
    specgrid = fig.add_gridspec(ncols=NCOLS,nrows=NROWS)
    # xindplot = NCOLS-1
    # yindplot = 0
    vmin = None
    for subdom in list(allspecs_per_sub_domain.keys()) :  # [::-1]
        # set the vmin and vmax using the first cross spectra

        # yindplot = int(np.floor(subdom/NCOLS))
        # print(np.where(final_inds==subdom))
        xindplot,yindplot = np.where(final_inds == subdom)
        # print()
        fig.add_subplot(specgrid[xindplot[0],yindplot[0]])
        # print('subdom',subdom,'->',xindplot,yindplot)
        # plt.subplot(4,4,subdom+1)
        plt.title('sub domain #%s' % subdom)
        if part=='Re':
            crossSpectraRe = np.abs(allspecs_per_sub_domain[subdom]['cross-spectrum_2tau'].mean(dim='2tau').real)
            crossSpectraRe_red = crossSpectraRe.where(
                np.logical_and(np.abs(crossSpectraRe.kx) <= 0.14,np.abs(crossSpectraRe.ky) <= 0.14),drop=True)
            crossSpectraRe_red = crossSpectraRe_red.rolling(kx=3,center=True).mean().rolling(ky=3,center=True).mean()
            if vmin is None :
                vmin = 0
                vmax = crossSpectraRe_red.max()
            levels = np.linspace(vmin,vmax,25)
            crossSpectraRe_red.plot(x='kx',cmap=cmap,levels=levels,vmin=0)
        else:
            crossSpectraIm = allspecs_per_sub_domain[subdom]['cross-spectrum_2tau'].mean(dim='2tau').imag
            crossSpectraIm_red = crossSpectraIm.where(
                np.logical_and(np.abs(crossSpectraIm.kx) <= 0.14,np.abs(crossSpectraIm.ky) <= 0.14),drop=True)
            crossSpectraIm_red = crossSpectraIm_red.rolling(kx=3,center=True).mean().rolling(ky=3,center=True).mean()
            if vmin is None :
                vmin = -crossSpectraIm_red.max()
                vmin = crossSpectraIm_red.min()
                vmax = crossSpectraIm_red.max()
            levels = np.linspace(vmin,vmax,25)
            crossSpectraIm_red.plot(x='kx',cmap=PuOr,levels=levels,vmin=vmin)
        plt.grid()
        plt.axis('scaled')





def display_xspectra_per_domain_polar(allspecs_per_sub_domain,heading,part='Re',min_wavelength=100):
    """

    :param allspecs_per_sub_domain:
    :return:
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("",["white","violet","mediumpurple","cyan","springgreen","yellow",
                                                         "red"])
    ncols = int(np.sqrt(len(allspecs_per_sub_domain)))

    nrows = 0
    while nrows*ncols<len(allspecs_per_sub_domain):
        nrows += 1
    PuOr = plt.get_cmap('PuOr')
    indices = np.arange(ncols*nrows)[: :-1].reshape(ncols,nrows)

    final_inds = indices[: :-1].T


    fig = plt.figure(figsize=(8 *ncols ,6 *nrows ),dpi=50)
    NCOLS = ncols
    NROWS = nrows
    specgrid = fig.add_gridspec(ncols=NCOLS,nrows=NROWS)
    # xindplot = NCOLS-1
    # yindplot = 0
    vmin = None
    for subdom in list(allspecs_per_sub_domain.keys()) :  # [::-1]
        # set the vmin and vmax using the first cross spectra

        # yindplot = int(np.floor(subdom/NCOLS))

        xindplot,yindplot = np.where(final_inds == subdom)

        ax = fig.add_subplot(specgrid[xindplot[0],yindplot[0]],polar=True)
        ax.set_theta_direction(-1)  # theta increasing clockwise
        ax.set_theta_zero_location("N")
        # plt.subplot(4,4,subdom+1)
        plt.title('sub domain #%s' % subdom)
        if part=='Re':
            crossSpectraRe = np.abs(allspecs_per_sub_domain[subdom]['cross-spectrum_2tau'].mean(dim='2tau').real)
        else:
            crossSpectraRe = allspecs_per_sub_domain[subdom]['cross-spectrum_2tau'].mean(dim='2tau').imag
        #crossSpectraRe = crossSpectraRe.rolling(kx=3,center=True).mean().rolling(ky=3,center=True).mean()
        logging.warning('trick to match a reference spectra on cartesian coords kx and ky')
        #cspRe_okGrid = crossSpectraRe
        # cspRe_okGrid = crossSpectraRe.assign_coords(kx=crossSpectraRe.kx.data / (1.1 * np.pi),
        #                                             ky=crossSpectraRe.ky.data / (1.1 * np.pi))
        crossSpectraRe = from_xCartesianSpectrum(crossSpectraRe,Nphi=72,ksampling='log',**{'Nk' : 60})
        crossSpectraRe = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(crossSpectraRe)
        crossSpectraRe = spectrum_rotation.apply_rotation(crossSpectraRe,
                                                             90.)  # This is for having origin at North
        crossSpectraRe = spectrum_rotation.apply_rotation(crossSpectraRe,heading)
        # crossSpectraRe_red = crossSpectraRe.where(
        #     np.logical_and(np.abs(crossSpectraRe.kx) <= 0.14,np.abs(crossSpectraRe.ky) <= 0.14),drop=True)
        # ax.set_rmax(0.8)
        #ax.set_ylim([0,0.2])
        # plt.contourf(np.radians(np.arange(0,360,5)),reference_oswK_1145m_60pts,np.sqrt(abs(crossSpectraRePol))) #,vmax=100000000,vmin=10000000

        # plt.pcolor(crossSpectraRePol/crossSpectraRePol.max())
        # plt.contourf(new_spec_Polar.phi,new_spec_Polar.k,crossSpectraRePol)
        crossSpectraRe_redpol = crossSpectraRe.where(np.abs(crossSpectraRe.k) <= 2.*np.pi/min_wavelength,drop=True)
        crossSpectraRe_redpol = crossSpectraRe_redpol.rolling(k=3,center=True).mean().rolling(phi=3,center=True).mean()
        if part=='Re':
            if vmin is None :
                vmin = 0
        else:
            vmin = crossSpectraRe_redpol.min()
        vmax = crossSpectraRe_redpol.max()
        levels = np.linspace(vmin,vmax,25)
        if part == 'Re' :
            crossSpectraRe_redpol.plot(cmap=cmap,levels=levels,vmin=0)
        else:
            crossSpectraRe_redpol.plot(cmap=PuOr,levels=levels)
        max_k = max(crossSpectraRe_redpol.k.values)
        plt.plot(np.radians([heading,heading]),(0.01,max_k),'r-')
        plt.plot(np.radians([heading + 180,heading + 180]),(0.01,max_k),'r-')
        plt.plot(np.radians([heading + 90,heading + 90]),(0.01,max_k),'r-')
        plt.plot(np.radians([heading + 270,heading + 270]),(0.01,max_k),'r-')
        ax.text(np.radians(heading),(0.65 + (np.cos(np.radians(heading)) < 0) * 0.25) * max_k,
                 ' Azimuth',size=18,color='red',rotation=-heading + 90 + (np.sin(np.radians(heading)) < 0) * 180,
                 ha='center')
        ax.text(np.radians(heading + 90),0.80 * max_k,' Range',size=18,color='red',
                 rotation=-heading + (np.cos(np.radians(heading)) < 0) * 180,ha='center')
        plt.grid(0,axis='y')
        r = [200,300,400,600,1000]
        circle_plot(ax,r,freq=0)
        #plt.axis('scaled')