"""
Function to create and merge interfaces
"""
import numpy as np
import xarray as xr

from .coordinates import get_shape


def create_interface(coordinates, fill_value=0):
    """
    Create an empty array to model a 2D or 3D interface

    The interface will be defined on horizontal coordinates, therefore its values will
    be the depth to the interface.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid that will be used to create the interface.
        Must be in meters and can be either 2D or 3D. If they are in 2D, the interface
        will be a curve, and if the coordinates are 3D, the interface will be a surface.
    fill_value : float (optional) or None
        Value that will fill the initialized array. If None, the array will be filled
        with ``numpy.nan``s. Default to 0.

    Returns
    -------
    da : :class:`xarray.DataArray`
        Array containing the interface filled with the same value.
    """
    # Get shape of coordinates
    shape = get_shape(coordinates)
    if fill_value is None:
        fill_value = np.nan
    # Remove the shape on z
    shape = shape[:-1]
    return xr.DataArray(
        fill_value * np.ones(shape),
        coords = [coordinates[i] for i in coordinates if i != "z"],
    )


def merge_interfaces(interfaces):
    """
    Merge a dictionary of interfaces into a single xarray.Dataset

    Parameters
    ----------
    interfaces : dict
        Dictionary containing a collection of interfaces.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        Dataset containing the interfaces.
    """
    ds = None
    for name, interface in interfaces.items():
        if ds:
            ds[name] = interface
        else:
            ds = interfaces[name].to_dataset(name=name)
    return ds


def interface_from_vertices(vertices, coordinates, direction="x"):
    """
    Create an interface by interpolating its vertices

    It works only for building interfaces in 2D or 3D from a 2D profiles.

    Parameters
    ----------
    vertices : list
        List of vertices of the interface.
    coordinates : :class:`xarray.DataArrayCoordinates`
        Two dimensional coordinates located on a regular grid.
    direction : string (optional)
        Direction of the subduction. If working in 3D it can be either *"x"* or *"y"*.
        When working in 2D, it must be *"x"*.

    Returns
    -------
    interface : :class:`xarray.DataArray`
        Array containing the interface.
    """
    h_min, h_max = coordinates[direction].min(), coordinates[direction].max()
    vertices = np.array(vertices)
    _check_boundary_vertices(vertices, h_min, h_max)
    interface = np.interp(coordinates[direction], vertices[:, 0], vertices[:, 1])
    da = xr.DataArray(interface, coords=[coordinates[direction]], dims=direction)
    if len(coordinates.dims) == 3:
        if direction == "x":
            missing_dim = "y"
        elif direction == "y":
            missing_dim == "x"
        da = da.expand_dims({missing_dim: coordinates[missing_dim].size})
        da.coords[missing_dim] = coordinates[missing_dim]
        da = da.transpose("x", "y")
    return da


def _check_boundary_vertices(vertices, x_min, x_max):
    """
    Check if boundary vertices matches the boundary of the coordinates
    """
    x = vertices[:, 0]
    if not np.allclose(x_min, x.min()) or not np.allclose(x_max, x.max()):
        raise ValueError(
            "Invalid vertices for creating the interfaces: {}. ".format(vertices)
            + "Remember to include boundary nodes that matches the coordinates "
            + "boundaries '{}.'".format((x_min, x_max))
        )
