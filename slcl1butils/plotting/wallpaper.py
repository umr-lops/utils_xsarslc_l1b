import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datatree
import os
import logging
from matplotlib import colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white","violet","mediumpurple","cyan","springgreen","yellow","red"])
PuOr = mcolors.LinearSegmentedColormap.from_list("", ["darkgoldenrod","white","purple"])

def plot_wallpaper(l1b_path, png_path='wallpaper_xspectra.png'):
    """

    Args:
        l1b_path: str
        png_path: str

    Returns:

    """
    root, ext = os.path.splitext(png_path)
    real_path = root+'_real'+ext
    imag_path = root+'_imag'+ext
    xsplot, orbit_pass = make_wallpaper(l1b_path)
    if orbit_pass.upper()=='ASCENDING':
        xsplot = xsplot.stack(trange=['tile_sample','range'])
        xsplot = xsplot.assign_coords({'burst':np.flip(xsplot['burst'].data), 'tile_line':np.flip(xsplot['tile_line'].data)}).sortby('burst', 'tile_line')
        xsplot = xsplot.stack(tazimuth=['burst','tile_line','azimuth'])
    elif orbit_pass.upper()=='DESCENDING':
        xsplot = xsplot.assign_coords({'tile_sample':np.flip(xsplot['tile_sample'].data)}).sortby('tile_sample')
        xsplot = xsplot.stack(trange=['tile_sample','range'])
        xsplot = xsplot.stack(tazimuth=['burst','tile_line','azimuth'])
    else:
        raise ValueError('Unknown orbit pass: {}'.format(orbit_pass))

    plt.figure(figsize=(320,180))
    plt.imshow(xsplot['xspectra_real'].transpose('tazimuth','trange','rgb').data)
    plt.savefig(real_path, dpi='figure',bbox_inches='tight')
    plt.close()
    logging.info('real path png: %s',real_path)
    plt.figure(figsize=(320,180))
    plt.imshow(xsplot['xspectra_imag'].transpose('tazimuth','trange','rgb').data)
    plt.savefig(imag_path, dpi='figure',bbox_inches='tight')
    plt.close()
    logging.info('imaginary path png: %s', imag_path)


def make_wallpaper(l1b_path):
    """

    Args:
        l1b_path: str

    Returns:

    """
    from slcl1butils.symmetrize_l1b_spectra import symmetrize_xspectrum
    from slcl1butils.utils import xndindex
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    l1b = datatree.open_datatree(l1b_path)
    xs2tau = l1b['intraburst']['xspectra_2tau_Re']+1j*l1b['intraburst']['xspectra_2tau_Im']
    xs2tau = xs2tau.assign_coords({'k_rg':xs2tau['k_rg'].mean(dim=set(xs2tau['k_rg'].dims)-set(['freq_sample']), keep_attrs=True)}).swap_dims({'freq_sample':'k_rg', 'freq_line':'k_az'})
    xs2tau = symmetrize_xspectrum(xs2tau).squeeze(dim='2tau')

    xsplot = list()
    for mytile in xndindex({'burst':xs2tau.sizes['burst'], 'tile_line':xs2tau.sizes['tile_line'], 'tile_sample':xs2tau.sizes['tile_sample']}):

        heading = np.radians(l1b['intraburst']['ground_heading'][mytile].data)
        incidence = np.round(l1b['intraburst']['incidence'][mytile].item(),2)
        tau = np.round(l1b['intraburst']['tau'][mytile].item(),3)
        try:
            cutoff = int(l1b['intraburst'].ds[mytile]['azimuth_cutoff'].data)
        except:
            cutoff = np.NaN
        lon = np.round(l1b['intraburst'].ds[mytile]['longitude'].item(), 2)
        lat = np.round(l1b['intraburst'].ds[mytile]['latitude'].item(), 2)
        keast = xs2tau['k_rg']*np.cos(heading)+xs2tau['k_az']*np.sin(heading)
        knorth = xs2tau['k_az']*np.cos(heading)-xs2tau['k_rg']*np.sin(heading)
        keast.attrs.update({'long_name':'wavenumber in East direction', 'units':'rad/m'})
        knorth.attrs.update({'long_name':'wavenumber in North direction', 'units':'rad/m'})
        xs2tau = xs2tau.assign_coords({'k_east':keast,'k_north':knorth})
        heading=np.arctan2(np.sin(heading), np.cos(heading))
        range_rotation = -np.degrees(heading) if np.abs(np.degrees(heading))<=90 else -np.degrees(heading)+180
        azimuth_rotation = -np.degrees(heading)+90 if np.degrees(heading)>=0 else -np.degrees(heading)-90

        figreal = plt.figure(figsize=(10,10), tight_layout=True)
        xs2tau[mytile].real.plot(cmap=cmap, vmin=0, x='k_east', y='k_north')
        ax = plt.gca()
        for r in [400,200,100,50]:
            circle = plt.Circle((0, 0), 2*np.pi/r, color='k', fill=False, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
            plt.text(-np.sqrt(2)*np.pi/r-0.002,np.sqrt(2)*np.pi/r+0.002,'{} m'.format(r), rotation=45.,horizontalalignment='center',verticalalignment='center')
        for a in [-60,-30,30,60]:
            plt.plot([-0.2*np.cos(np.radians(a)), 0.2*np.cos(np.radians(a))],[-0.2*np.sin(np.radians(a)), 0.2*np.sin(np.radians(a))], color='k', linestyle='--', linewidth=0.5)
        plt.vlines(0,-0.2,0.2, color='k', linestyle='--', linewidth=0.5)
        plt.hlines(0,-0.2,0.2, color='k', linestyle='--', linewidth=0.5)
        xp = 0.14*np.cos(heading)
        yp = 0.14*np.sin(heading)
        plt.plot([-xp,xp], [yp,-yp], color='r') # range line
        plt.plot([yp,-yp], [xp,-xp], color='r') # azimuth line
        if cutoff!=0:
            plt.plot(np.array([-xp,xp])+2*np.pi/cutoff*np.sin(heading), np.array([yp,-yp])+2*np.pi/cutoff*np.cos(heading), color='k', linestyle='--') # cutoff upper line
            plt.plot(np.array([-xp,xp])-2*np.pi/cutoff*np.sin(heading), np.array([yp,-yp])-2*np.pi/cutoff*np.cos(heading), color='k', linestyle='--') # cutoff lower line
        plt.axis('scaled')
        plt.xlim([-0.14,0.14])
        plt.ylim([-0.14,0.14])

        plt.text(0.9*xp,-0.9*yp+0.006,'range', color='r', rotation=range_rotation, fontsize=15,horizontalalignment='center',verticalalignment='center') # range name
        plt.text(0.85*yp+0.006,0.85*xp,'azimuth', color='r', rotation=azimuth_rotation, fontsize=15,horizontalalignment='center',verticalalignment='center') # azimuth name
        plt.text(-0.135,-0.135,'cuto-off : {} m'.format(cutoff))
        plt.text(-0.135,0.13,'incidence : {} deg'.format(incidence))
        plt.text(0.065,0.13,'longitude : {} deg'.format(lon))
        plt.text(0.065,0.12,'latitude : {} deg'.format(lat))
        plt.text(-0.135,0.12,'tau : {} s'.format(tau))
        plt.title('X-spectrum (real)')

        canvas = FigureCanvas(figreal)
        width, height = figreal.get_size_inches() * figreal.get_dpi()
        canvas.draw()
        imagereal = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)[146:854,102:810,:]
        plt.close()
        
        # -----------------------------------------

        figimag = plt.figure(figsize=(10,10), tight_layout=True)
        xs2tau[mytile].imag.plot(cmap=PuOr, x='k_east', y='k_north')
        ax = plt.gca()
        for r in [400,200,100,50]:
            circle = plt.Circle((0, 0), 2*np.pi/r, color='k', fill=False, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
            plt.text(-np.sqrt(2)*np.pi/r-0.002,np.sqrt(2)*np.pi/r+0.002,'{} m'.format(r), rotation=45.,horizontalalignment='center',verticalalignment='center')
        for a in [-60,-30,30,60]:
            plt.plot([-0.2*np.cos(np.radians(a)), 0.2*np.cos(np.radians(a))],[-0.2*np.sin(np.radians(a)), 0.2*np.sin(np.radians(a))], color='k', linestyle='--', linewidth=0.5)
        plt.vlines(0,-0.2,0.2, color='k', linestyle='--', linewidth=0.5)
        plt.hlines(0,-0.2,0.2, color='k', linestyle='--', linewidth=0.5)

        plt.plot([-xp,xp], [yp,-yp], color='r') # range line
        plt.plot([yp,-yp], [xp,-xp], color='r') # azimuth line
        if cutoff!=0:
            plt.plot(np.array([-xp,xp])+2*np.pi/cutoff*np.sin(heading), np.array([yp,-yp])+2*np.pi/cutoff*np.cos(heading), color='k', linestyle='--') # cutoff upper line
            plt.plot(np.array([-xp,xp])-2*np.pi/cutoff*np.sin(heading), np.array([yp,-yp])-2*np.pi/cutoff*np.cos(heading), color='k', linestyle='--') # cutoff lower line

        plt.axis('scaled')
        plt.xlim([-0.14,0.14])
        plt.ylim([-0.14,0.14])
        # plt.grid()
        plt.text(0.9*xp,-0.9*yp+0.006,'range', color='r', rotation=range_rotation, fontsize=15,horizontalalignment='center',verticalalignment='center') # range name
        plt.text(0.85*yp+0.006,0.85*xp,'azimuth', color='r', rotation=azimuth_rotation, fontsize=15,horizontalalignment='center',verticalalignment='center') # azimuth name
        plt.text(-0.135,-0.135,'cuto-off : {} m'.format(cutoff))
        plt.text(-0.135,0.13,'incidence : {} deg'.format(incidence))
        plt.text(-0.135,0.12,'tau : {} s'.format(tau))
        plt.text(0.065,0.13,'longitude : {} deg'.format(lon))
        plt.text(0.065,0.12,'latitude : {} deg'.format(lat))
        plt.title('X-spectrum (imaginary)')

        canvas = FigureCanvas(figimag)
        width, height = figimag.get_size_inches() * figimag.get_dpi()
        canvas.draw()
        imageimag = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)[146:854,102:810,:]
        plt.close()

        xsreal = xr.DataArray(imagereal, dims=('azimuth','range','rgb'), name='xspectra_real').assign_coords(mytile)
        xsimag = xr.DataArray(imageimag, dims=('azimuth','range','rgb'), name='xspectra_imag').assign_coords(mytile)
        xs = xr.merge([xsreal, xsimag])
        xsplot.append(xs)
    xsplot = xr.combine_by_coords([xs.expand_dims(['burst', 'tile_line','tile_sample']) for xs in xsplot])
    return xsplot, l1b['intraburst'].attrs['orbit_pass']