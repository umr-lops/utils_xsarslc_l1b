
import os
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white","violet","mediumpurple","cyan","springgreen","yellow","red"])
import numpy as np
from slcl1butils import  conversion_polar_cartesian
from slcl1butils import spectrum_clockwise_to_trigo
try:
    import display_xspectra_grid
except:
    pass #TODO fix this dependency keep or let
from slcl1butils import spectrum_rotation
from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum
#import slcl1butils.plotting.add_azimuth_cutoff_lines_on_polar_spec_fig #wavetoolbox
reference_oswK_1145m_60pts = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377,
    0.007157583, 0.007619386, 0.008110984, 0.008634299, 0.009191379,
    0.0097844, 0.01041568, 0.0110877, 0.01180307, 0.01256459, 0.01337525,
    0.01423822, 0.01515686, 0.01613477, 0.01717577, 0.01828394, 0.01946361,
    0.02071939, 0.02205619, 0.02347924, 0.02499411, 0.02660671, 0.02832336,
    0.03015076, 0.03209607, 0.03416689, 0.03637131, 0.03871796, 0.04121602,
    0.04387525, 0.04670605, 0.0497195, 0.05292737, 0.05634221, 0.05997737,
    0.06384707, 0.06796645, 0.0723516, 0.07701967, 0.08198893, 0.08727881,
    0.09290998, 0.09890447, 0.1052857, 0.1120787, 0.1193099, 0.1270077,
    0.1352022, 0.1439253, 0.1532113, 0.1630964, 0.1736193, 0.1848211,
    0.1967456, 0.2094395])

def core_plot_part(crossSpectraRePol,heading,fig=None,ax=None,thecmap=None,limit_display_wv=100,title='x spec',
                   tauval=2,azimuth_cutoffSAR=None,vmin=None,vmax=None,add_colorbar=True):
    # partie display plot
    if fig is None :
        plt.figure(figsize=(20,8),dpi=100)
    if ax is None :
        ax = plt.subplot(1,1,1,polar=True)

        ax.set_theta_direction(-1)  # theta increasing clockwise
        ax.set_theta_zero_location("N")
    crossSpectraRe_redpol = crossSpectraRePol.where(np.abs(crossSpectraRePol.k) <= 2 * np.pi / limit_display_wv,
                                                    drop=True)

    # shift_phi = np.radians(heading-90)

    crossSpectraRe_redpol_phi_new = crossSpectraRe_redpol

    # crossSpectraRe_redpol_phi_new = crossSpectraRe_redpol.assign_coords({'phi':(crossSpectraRe_redpol.phi.values+shift_phi)[::-1]})
    # crossSpectraRe_redpol_phi_new = crossSpectraRe_redpol_phi_new/45000.
    im = crossSpectraRe_redpol_phi_new.plot(cmap=thecmap,alpha=0.8,vmin=vmin,vmax=vmax,add_colorbar=add_colorbar)
    #plt.pcolor(crossSpectraRe_redpol_phi_new.phi,crossSpectraRe_redpol_phi_new.k,crossSpectraRe_redpol_phi_new,cmap=thecmap)
    plt.grid(True)
    plt.plot(np.radians([heading,heading]),(0.01,max(crossSpectraRe_redpol.k)),'r-')
    plt.plot(np.radians([heading + 180,heading + 180]),(0.01,max(crossSpectraRe_redpol.k)),'r-')
    plt.plot(np.radians([heading + 90,heading + 90]),(0.01,max(crossSpectraRe_redpol.k)),'r-')
    plt.plot(np.radians([heading + 270,heading + 270]),(0.01,max(crossSpectraRe_redpol.k)),'r-')
    ax.text(np.radians(heading),(0.65 + (np.cos(np.radians(heading)) < 0) * 0.25) * max(crossSpectraRe_redpol.k),
            ' Azimuth',size=18,color='red',rotation=-heading + 90 + (np.sin(np.radians(heading)) < 0) * 180,
            ha='center')
    ax.text(np.radians(heading + 90),0.80 * max(crossSpectraRe_redpol.k),' Range',size=18,color='red',
            rotation=-heading + (np.cos(np.radians(heading)) < 0) * 180,ha='center')
    display_xspectra_grid.circle_plot(ax,r=[100,200,400])
    if azimuth_cutoffSAR:
        add_azimuth_cutoff_lines_on_polar_spec_fig.add_azimuth_cutoff_lines(ax=ax,
                                                                        tra=heading,
                                                                        limit_wl_plot=limit_display_wv,
                                                                        azc=azimuth_cutoffSAR)
    ax.set_rmax(2.0 * np.pi / limit_display_wv)
    ax.set_rmin(0) #security!!
    if title is None :
        plt.title('tau : %s' % tauval,fontsize=18)
    else :
        plt.title(title,fontsize=18)
    return im


def add_polar_direction_lines(deg_step=30):
    """
    grey lines to show different theta polar direction on a cartesian grid figure
    Parameters
    ----------
    deg_step
        float : stride of the polar lines displayed in degrees

    Returns
    -------

    """
    #
    maxx = 0.06
    offset_angle = -90
    for yy,theta in enumerate(np.arange(0,360,deg_step)):
        x1,y1 = pol2cart(maxx,np.radians(theta))
        x2,y2 = pol2cart(maxx,np.radians((theta+180)%360))
        x3,y3 = pol2cart(0.06,np.radians((theta)))
        x = np.array([x1,x2])
        y = np.array([y1,y2])
        x = np.array([x2,x1])
        y = np.array([y2,y1])
        #print(x)
        plt.plot(x,y,c='grey',alpha=0.3,lw=1)
        #rotn = np.degrees(np.arctan2(y[1:]-y[:-1], x[1:]-x[:-1]))[0]
        rotn = theta #label
        rotn = (offset_angle+rotn)%360
        #if rotn>0 and rotn<300:
        # if True:
        #print(rotn)
        if abs(rotn)>90: #on retourne le label si ca depasse 90 ou -90
            rotn  = (rotn+180)%180
        plt.annotate("%i°"%np.rint(rotn), xy=(x3, y3), rotation=rotn,fontsize=6,color='grey')
        # if yy==20:
        #     break


def display_polar_spectra(allspecs,heading,part='Re',limit_display_wv=100,title=None,outputfile=None,
                          fig=None,ax=None,interactive=False,tau_id=2,varname='cross-spectrum_%stau'):
    """

    :param allspecs:
    :param heading:
    :param part:
    :param limit_display_wv:
    :param title:
    :param outputfile:
    :param fig:
    :param ax:
    :param interactive:
    :param tau_id:
    :return:
    """

    for tauval in [tau_id] :
        if part == 'Im':
            thecmap = 'PuOr'
            #if isinstance(allspecs,xarray.DataArray):
            if '2tau' not in allspecs.dims:
                coS = allspecs
            else:
                if allspecs[varname  % tauval].dtype in [np.complex_, np.complex]:
                    coS = allspecs[varname % tauval].mean(
                        dim='%stau' % tauval).imag  # co spectrum = imag part of cross-spectrum
                else:
                    coS = allspecs[varname  % tauval].mean(
                        dim='%stau' % tauval)  # co spectrum = imag part of cross-spectrum
        else:
            thecmap = cmap
            coS = abs(allspecs[varname % tauval].mean(
                dim='%stau' % tauval).real)  # co spectrum = real part of cross-spectrum
        # new_spec_Polar = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(coS,Nphi=72,
        #                                                                             ksampling='log',
        #                                                                 **{'Nk' : 60,
        #                                                                    'kmin' :
        #                                                                        reference_oswK_1145m_60pts[
        #                                                                            0],
        #                                                                    'kmax' :
        #                                                                        reference_oswK_1145m_60pts[
        #                                                                            -1]})
        new_spec_Polar = conversion_polar_cartesian.from_xCartesianSpectrum(coS,Nphi=72,
                                                                                    ksampling='log',
                                                                                    **{'k' : reference_oswK_1145m_60pts})
        # new_spec_Polar = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(coS,Nphi=72,
        #                                                                             ksampling='log',
        #                                                                             **{'k' : reference_oswK_1145m_60pts})
        #new_spec_Polar.assign_coords({'k':reference_oswK_1145m_60pts}) # test agrouaze

        crossSpectraRePol = new_spec_Polar.squeeze()
        crossSpectraRePol = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(crossSpectraRePol)
        #crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,-90.)  # This is for having origin at North # 1dec21 change +90 to -90 on a case descending
        crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol, -270.) # sept 22: it seems consistent with what we see in the cart spectra, while -90 is opposite direction
        crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,heading)
        if part == 'Re':
            crossSpectraRePol = abs(crossSpectraRePol)
        im = core_plot_part(crossSpectraRePol,heading,fig=fig,ax=ax,thecmap=thecmap,
                       limit_display_wv=limit_display_wv,title=title,
                       tauval=tauval)
        if outputfile:
            plt.savefig(outputfile)
            logging.info('outputfile : %s',outputfile)
        else:
            if interactive:
                pass
                #plt.show()
    return im





def plot_a_single_xspec_cart_L1B_IW(ds,bursti,tile_line_i,tile_sample_i,title,fig,cat_xspec='intra',part='Re',
                                    rotation=False,orbit_pass=None,platform_heading=None,dpi=100,figsize=(8,6),
                                    outputfile=None):
    """

    Parameters
    ----------
    ds xarray.Dataset intra or inter from L1B IFREMER IW SLC product
    orbit_pass str 'Descending' or 'Ascending'
    bursti int
    tile_line_i int
    tile_sample_i int
    title str
    cat_xspec str intra or inter
    part str Re or Im
    rotation bool True -> rotation of the figure by platform heading, False -> keep image coordinates (ie line along azimuth axis and sample along range axis)
    outputfile str:
    Returns
    -------

    """
    import matplotlib
    from scipy import ndimage, misc
    nb_lines = 1
    nb_burst = 1
    nb_sample = 1
    cpt_spec = 1
    limit_display_wv = 50
    circles_wavelength = [50,100,200,300,400,500]
    tau_id = 2
    add_colorbar = True
    #fig = plt.figure()
    set_xspec = ds[{'burst':bursti,'tile_line':tile_line_i,'tile_sample':tile_sample_i}]
    if 'k_az' in  set_xspec.dims:
        print('symmetrize and swap_dims already done')
    else:
        set_xspec = set_xspec.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
        set_xspec = symmetrize_xspectrum(set_xspec, dim_range='k_rg', dim_azimuth='k_az')
    if cat_xspec == 'inter':
        #varname = cat_xspec+'burst_xspectra_'+part
        varname = cat_xspec + 'burst_xspectra'
    else:
        #varname = 'xspectra_%stau_'+part
        varname = 'xspectra_%stau' # it is mandatory to recombined Re+1j*Im dans the L1B
    # scales = (0, 5, 0, 5)
    # t = Affine2D().rotate_deg(25)

    # # Add floating axes
    # h = floating_axes.GridHelperCurveLinear(t, scales)
    # ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=h)

    plt.figure(dpi=dpi,figsize=figsize)
    #fig.add_subplot(ax)
    ax = plt.subplot(nb_lines * nb_burst, nb_sample, cpt_spec)
    im = display_cartesian_spectra(set_xspec,part=part,cat_xspec=cat_xspec,limit_display_wv=limit_display_wv,
                                   title=title,outputfile=outputfile,
          fig=fig,ax=ax,interactive=False,tau_id=tau_id,varname=varname,rolling=True,title_fontsize=8,
                circles_wavelength=circles_wavelength,add_colorbar=add_colorbar,kx_varname='k_rg',ky_varname='k_az',dpi=dpi) #
    if orbit_pass == 'Descending':
        ax.invert_xaxis()
        ax.invert_yaxis()
    plt.axis('scaled') # to make the circle looks like circles
    if rotation:
        #if True: # I need to save image on disk to rotate it
        outf = '/tmp/spectra_cart_iw_tiff_one_tile.png'
        plt.savefig(outf)
        img = matplotlib.pyplot.imread(outf)
        rot_ang = -(180 + platform_heading)
        print('rot_ang',rot_ang)
        #plt.figure()
        img_45 = ndimage.rotate(img, rot_ang, reshape=True)

        plt.figure()
        plt.imshow(img_45, cmap=plt.cm.gray)
        #plt.axvline(x=0.5)
        plt.quiver(50,200,0,60,color='k',scale=500,headwidth=10,headlength=12)
        plt.text(52,150,'North')
        plt.axis('off')
        plt.show()
    return set_xspec

def plot_a_single_xspec_cart_L1B_WV(ds,title,fig=None,ax=None,part='Re'
                                    ,orbit_pass=None,platform_heading=None,dpi=100,figsize=(8,6),
                                    outputfile=None,limit_display_wv = 50,
                                    circles_wavelength = [50, 100, 200, 300, 400, 500],vmax=None):
    """

    :param ds:
    :param title:
    :param fig:
    :param ax:
    :param part:
    :param orbit_pass:
    :param platform_heading:
    :param dpi:
    :param figsize:
    :param outputfile:
    :param limit_display_wv: float
    :return:
    """


    tau_id = 2
    add_colorbar = True
    # fig = plt.figure()
    varname = 'xspectra_%stau'  # it is mandatory to recombined Re+1j*Im dans the L1B
    if varname not in ds:
        logging.info('conjugate Re+Im')
        for tautau in range(3):
            ds['xspectra_%stau' % tautau] = ds['xspectra_%stau_Re' % tautau] + 1j * ds['xspectra_%stau_Im' % tautau]
            ds = ds.drop(['xspectra_%stau_Re' % tautau, 'xspectra_%stau_Im' % tautau])

    set_xspec = ds[varname%tau_id]
    if 'k_az' in set_xspec.dims:
        pass
    else:
        set_xspec = set_xspec.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
        set_xspec = symmetrize_xspectrum(set_xspec, dim_range='k_rg', dim_azimuth='k_az')
    ds[varname%tau_id] = set_xspec
    # scales = (0, 5, 0, 5)
    # t = Affine2D().rotate_deg(25)

    # # Add floating axes
    # h = floating_axes.GridHelperCurveLinear(t, scales)
    # ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=h)
    if fig is None:
        fig = plt.figure(dpi=dpi, figsize=figsize)
    # fig.add_subplot(ax)
    if ax is None:
        ax = plt.subplot(111)
    im = display_cartesian_spectra(ds, part=part, cat_xspec='intra', limit_display_wv=limit_display_wv,
                                   title=title, outputfile=outputfile,
                                   fig=fig, ax=ax, interactive=False, tau_id=tau_id, varname=varname, rolling=True,
                                   title_fontsize=8,
                                   circles_wavelength=circles_wavelength, add_colorbar=add_colorbar, kx_varname='k_rg',
                                   ky_varname='k_az', dpi=dpi,vmax=vmax)  #
    if orbit_pass == 'Descending':
        ax.invert_xaxis()
        ax.invert_yaxis()
    plt.axis('scaled')  # to make the circle looks like circles
    return set_xspec




def display_cartesian_spectra(allspecs,part='Re',cat_xspec='intra',limit_display_wv=100,title=None,outputfile=None,
                          fig=None,ax=None,interactive=False,tau_id=2,varname='cross-spectrum_%stau',title_fontsize=12,
                              kx_varname='kx',ky_varname='ky',rolling=False,dpi=100,circles_wavelength=[50,100,200,400],
                              add_colorbar=True,vmax=None):
    """

    Parameters
    ----------
    allspecs
    part
    cat_xspec intra or inter
    limit_display_wv
    title
    outputfile
    fig
    ax
    interactive
    tau_id
    varname
    kx_varname
    ky_varname
    rolling
    dpi
    circles_wavelength
    add_colorbar

    Returns
    -------

    """

    for tauval in [tau_id] :
        if cat_xspec == 'intra':
            varname = varname% tauval
            if part == 'Im':
                thecmap = 'PuOr'
                if allspecs[varname].dtype in [np.complex_, np.complex,np.complex64]:
                    coS = allspecs[varname].mean(
                        dim='%stau' % tauval).imag  # co spectrum = imag part of cross-spectrum
                else:
                    coS = allspecs[varname].mean(
                        dim='%stau' % tauval)  # co spectrum = imag part of cross-spectrum
            else:
                thecmap = cmap
                coS = abs(allspecs[varname].mean(
                    dim='%stau' % tauval).real)  # co spectrum = real part of cross-spectrum

        else:# inter burst case (without tau dimension)
            if part == 'Im':
                thecmap = 'PuOr'
                if allspecs[varname].dtype in [np.complex_, np.complex]:
                    coS = allspecs[varname].imag  # co spectrum = imag part of cross-spectrum
                else:
                    coS = allspecs[varname]
            else:
                thecmap = cmap
                coS = abs(allspecs[varname])
        crossSpectraRe_red = coS.where(
            np.logical_and(np.abs(coS[kx_varname]) <= 0.14,np.abs(coS[ky_varname]) <= 0.14),drop=True)
        #crossSpectraRe_red = crossSpectraRe_red.rolling(kx=3,center=True).mean().rolling(ky=3,center=True).mean()
        if rolling:
            crossSpectraRe_red = crossSpectraRe_red.rolling({kx_varname:3}, center=True).mean().rolling({ky_varname:3}, center=True).mean()

        # filter high wavelength (for test)
        # crossSpectraRe_red = crossSpectraRe_red.where(
        #     np.logical_and(np.abs(crossSpectraRe_red[kx_varname]) >= 2*np.pi/500., np.abs(crossSpectraRe_red[ky_varname]) >= 2*np.pi/500.), drop=True)
        # if True:
        #     lower_wl = 43.
        #     higher_wl = 250.
        #     from spectrum_momentum import filter_cartesian_with_wavelength_ring
        #     crossSpectraRe_red = filter_cartesian_with_wavelength_ring(lower_wl, higher_wl, crossSpectraRe_red)
        # partie display plot
        if fig is None:
            plt.figure(figsize=(10,8),dpi=dpi)
        if ax is None:

            ax = plt.subplot(1,1,1,polar=False)
        # add azimuth / range red axis
        # plt.axhline(y=0)
        plt.plot([0, 0], [-2.*np.pi/limit_display_wv, 2.*np.pi/limit_display_wv], 'c--',alpha=0.7,)
        plt.text(-0.001,0.05,'azimuth',color='c',rotation=90, va='center',fontsize=7)
        plt.plot([-2.*np.pi/limit_display_wv, 2.*np.pi/limit_display_wv], [0, 0], 'r--',alpha=0.7)
        plt.text(0.05, 0.004, 'range', color='r', rotation=0, va='center',fontsize=7)
        crossSpectraRe_red = crossSpectraRe_red.transpose("k_az", "k_rg")
        im = crossSpectraRe_red.plot(cmap=thecmap,alpha=0.8,add_colorbar=add_colorbar,vmax=vmax)
        add_cartesian_wavelength_circles(default_values=circles_wavelength)



        plt.legend(loc=1)
        plt.grid(True)

        #display_xspectra_grid.circle_plot(ax,r=[300,400,500,600])
        #ax.set_rmax(2.0*np.pi/limit_display_wv)
        if title is None:
            plt.title('tau : %s' % tauval,fontsize=title_fontsize)
        else:
            plt.title(title,fontsize=title_fontsize)
        if outputfile:
            plt.axis('scaled')  # to make the circle looks like circles
            filename, file_extension = os.path.splitext(outputfile)
            plt.savefig(outputfile,format=file_extension.replace('.',''))
            logging.info('outputfile : %s',outputfile)
        else:
            if interactive:
                #plt.show()
                pass
        return im
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
def add_cartesian_wavelength_circles(default_values=[100,300,600]):


    N_pts = 100
    phi = np.linspace(0, 2 * np.pi, N_pts)
    for rru in default_values:
        r100 = np.ones(N_pts) * 2 * np.pi / rru
        #r300 = np.ones(N_pts) * 2 * np.pi / 300
        # print(phi)
        # print(r)
        coords_cart_100 = []
        for uu in range(N_pts):
            coords_cart_100.append(pol2cart(r100[uu], phi[uu]))
        coords_cart_100 = np.array(coords_cart_100)

        # coords_cart_300 = []
        # for uu in range(N_pts):
        #     coords_cart_300.append(pol2cart(r300[uu], phi[uu]))
        # coords_cart_300 = np.array(coords_cart_300)
        # print(coords_cart_300.shape)
        # plt.plot(coords_cart_300[:, 0], coords_cart_300[:, 1], label='300 m')
        plt.plot(coords_cart_100[:, 0], coords_cart_100[:, 1],'--', label='%s m'%rru)

