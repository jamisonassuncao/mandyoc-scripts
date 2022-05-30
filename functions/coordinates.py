"""
Functions to create coordinates and initilize empty arrays
"""
import numpy as np
import xarray as xr


def grid_coordinates(region, shape):
    """
    Create grid coordinates for 2D models

    Parameters
    ----------
    region : tuple or list
        List containing the boundaries of the region of the grid. If the grid is 3D, the
        boundaries should be passed in the following order:
        ``x_min``, ``x_max``,``z_min``, ``z_max``.
        If the grid is 2D, the boundaries should be passed in the following order:
        ``x_min``, ``x_max``, ``z_min``, ``z_max``.
    shape : tuple
        Total number of grid points along each direction.
        If the grid is 3D, the tuple must be: ``n_x``, ``n_y``, ``n_z``.
        If the grid is 2D, the tuple must be: ``n_x``, ``n_z``.

    Returns
    -------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid.
    """
    # Sanity checks
    _check_region(region)
    _check_region_shape(region, shape)
    # Build coordinates according to
    if len(shape) == 2:
        nx, nz = shape[:]
        x_min, x_max, z_min, z_max = region[:]
        x = np.linspace(x_min, x_max, nx)
        z = np.linspace(z_min, z_max, nz)
        dims = ("x", "z")
        coords = {"x": x, "z": z}
    else:
        nx, ny, nz = shape[:]
        x_min, x_max, y_min, y_max, z_min, z_max = region[:]
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        dims = ("x", "y", "z")
        coords = {"x": x, "y": y, "z": z}
    da = xr.DataArray(np.zeros(shape), coords=coords, dims=dims)
    return da.coords


def create_grid(coordinates, fill_value=0):
    """
    Create an empty array for a set of coordinates

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid where the temperature distribution will be
        created. Must be in meters and can be either 2D or 3D.
    fill_value : float (optional) or None
        Value that will fill the initialized array. If None, the array will be filled
        with ``numpy.nan``s. Default to 0.
    """
    # Get shape of coordinates
    shape = get_shape(coordinates)
    if fill_value is None:
        fill_value = np.nan
    return xr.DataArray(fill_value * np.ones(shape), coords=coordinates)


def get_shape(coordinates):
    """
    Return shape of coordinates

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid.

    Return
    ------
    shape : tuple
        Tuple containing the shape of the coordinates
    """
    return tuple(coordinates[i].size for i in coordinates.dims)


def _check_region(region):
    """
    Sanity checks for region
    """
    if len(region) == 4:
        x_min, x_max, z_min, z_max = region
    elif len(region) == 6:
        x_min, x_max, y_min, y_max, z_min, z_max = region
        if x_min > x_max:
            raise ValueError(
                "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
                + "Must have y_min =< y_max. "
            )
    else:
        raise ValueError(
            "Invalid region '{}'. ".format(region)
            + "Only 4 or 6 values allowed for 2D and 3D dimensions, respectively."
        )
    if x_min > x_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have x_min =< x_max. "
        )
    if z_min > z_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have z_min =< z_max. "
        )


def _check_region_shape(region, shape):
    """
    Check shape lenght and if the region matches it
    """
    if len(shape) not in (2, 3):
        raise ValueError(
            "Invalid shape '{}'. ".format(shape) + "Shape must have 2 or 3 elements."
        )
    if len(shape) != len(region) // 2:
        raise ValueError(
            "Invalid region '{}' for shape '{}'. ".format(region, shape)
            + "Region must have twice the elements of shape."
        )
