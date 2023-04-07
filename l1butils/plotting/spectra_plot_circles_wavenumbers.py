



import numpy as np
from matplotlib import pyplot as plt
def circle_plot ( ax,r,freq=0 ) :
    # Radial Circles and their label
    #theta = 2 * np.pi * np.arange(360) / 360.
    theta = np.radians(np.arange(360))
    labels = []
    if freq == 0 :
        for i in r :
            plt.plot(theta,np.arange(360) * 0 + 2 * np.pi / i,'--k')
            labels.append(str(i) + ' m')
        ax.set_rgrids([2 * np.pi / i for i in r],labels=labels,angle=45.)

    if freq == 1 :
        for i in r :
            plt.plot(theta,np.arange(360) * 0 + np.sqrt(2 * np.pi / i * 9.81 / (2 * np.pi) ** 2),'--k')
            labels.append(str(i) + ' m')
        ax.set_rgrids([np.sqrt(2 * np.pi / i * 9.81 / (2 * np.pi) ** 2) for i in r],labels=labels,angle=135.)