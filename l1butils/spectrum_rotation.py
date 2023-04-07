import numpy as np
def apply_rotation ( ds,angle ) :
    """
    Rotate spectrum. Angle is assumed to be in the same angle convention than ds

    Args:
        ds (xarray): xPolarSpectrum
        angle (float): rotation angle [deg]. Same convention than the spectrum)
    """
    angle_mpipi = lambda t : np.arctan2(np.sin(t),np.cos(t))
    ds = ds.roll(phi=int(np.rint(np.radians(angle) / ds.dphi)),roll_coords=False)
    if 'wd' in ds.attrs : ds.attrs.update(
        {'wd' : angle_mpipi(ds.wd + np.radians(angle))})  # wind direction has to be rotated too
    if 'curdir' in ds.attrs : ds.attrs.update(
        {'curdir' : angle_mpipi(ds.curdir + np.radians(angle))})  # current direction has to be rotated too
    return ds
