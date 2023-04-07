"""
A Grouazel + C Pouplin
May 2022, validated
"""
import numpy as np
def add_azimuth_cutoff_lines(ax,tra,limit_wl_plot,azc):
    """

    :param ax: pyplot axis for polar plot
    :param tra: float track angle or bearing angle of SAR image [degrees clockwise wrt from North] (could be negative -12 or -166 for instance)
    :param limit_wl_plot: limit of the figure in wavelength (2pi/wavenumber) [m]
    :param azc: float: azimuth cutoff computed on a cross spectrum [m]
    :return:
    """
    R = 2*np.pi/limit_wl_plot
    Azc = 2*np.pi/azc
    ttt = np.radians(tra-180.)
    theta0 = np.arctan(Azc/R)
    #theta0bis = np.arctan2(R,Azc) the pi/2 complementary angle
    distance = np.sqrt(R**2+Azc**2)
    theta1 = (np.pi/2.)-theta0
    theta3 = theta1 + ttt
    theta4 = -theta1 + ttt
    ax.plot([theta3,theta4],[distance,distance],zorder=100000,color='violet',lw=1.5,alpha=0.5,linestyle='--')
    theta1 = (np.pi/2.)+theta0
    theta3 = theta1 + ttt
    theta4 = -theta1 + ttt
    ax.plot([theta3,theta4],[distance,distance],zorder=100000,color='violet',lw=1.5,alpha=0.5,linestyle='--')