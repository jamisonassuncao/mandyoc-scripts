"""
Save velocity distributions to ASCII files to be read by MANDYOC
"""
import os
import numpy as np

HEADER = "V1 \n V2 \n V3 \n V4"
FNAME = "input_velocity_0.txt"


def save_velocity(velocity, path, fname=FNAME):
    """
    Save velocity grid as ASCII file ready to be read by MANDYOC

    The velocity grid values are saved on three column, following each axe on
    in crescent order, with the ``x`` indexes changing faster that the ``y``.

    Parameters
    ----------
    velocity : :class:`xarray.Dataset`
        Dataset containing the three velocity component distribution.
    path : str
        Path to save the velocity file.
    fname : str (optional)
       Filename of the output ASCII file. Default to ``input_velocity_0.txt``.
    """
    dimension = len(velocity.dims)
    shape = velocity.velocity_x.shape
    size = np.array(shape).prod()
    
    components = (velocity.velocity_x.T, velocity.velocity_z)
    velocity_to_save = np.zeros((dimension, size))
    
    for i, component in enumerate(components):
        velocity_to_save[i, :] = np.reshape(component.values, size)
    velocity_to_save = np.reshape(
        velocity_to_save.T, 
        (np.size(velocity_to_save))
        )
    # Will add a custom header required by MANDYOC
    return np.savetxt(os.path.join(path, fname), velocity_to_save, header=HEADER)
