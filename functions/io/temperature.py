"""
Save temperature distributions to ASCII files to be read by MANDYOC
"""
import os
import numpy as np

HEADER = "T1 \n T2 \n T3 \n T4"
FNAME = "input_temperature_0.txt"


def save_temperature(temperatures, path, fname=FNAME):
    """
    Save temperatures grid as ASCII file ready to be read by MANDYOC

    The temperatures grid values are saved on a single column, following each axe on
    in crescent order, with the ``x`` indexes changing faster that the ``z``.

    Parameters
    ----------
    temperatures : :class:`xarray.DataArray`
        Array containing a temperature distribution. Can be either 2D or 3D.
    path : str
        Path to save the temperature file.
    fname : str (optional)
       Filename of the output ASCII file. Deault to ``input_temperature_0.txt``.
    """
    expected_dims = ("x", "z")
    # Check if temperature dims are the right ones
    invalid_dims = [dim for dim in temperatures.dims if dim not in expected_dims]
    if invalid_dims:
        raise ValueError(
            "Found invalid temperature dimensions '{}': ".format(invalid_dims)
            + "must be '{}' for 2D temperatures.".format(expected_dims)
        )
    # Change order of temperature dimensions to ("x", "z") to ensure
    # right order of elements when the array is ravelled
    temperatures = temperatures.transpose(*expected_dims)
    # Ravel and save temperatures
    # We will use order "F" on numpy.ravel in order to make the first index to change
    # faster than the rest
    # Will add a custom header required by MANDYOC
    np.savetxt(
        os.path.join(path, fname), temperatures.values.ravel(order="F"), header=HEADER
    )
