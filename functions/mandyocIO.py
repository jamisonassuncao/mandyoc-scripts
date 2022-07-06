"""
Library to handle input and output from Mandyoc code 
"""
import os
import numpy as np
import xarray as xr

PARAMETERS_DICT = {
    "compositional_factor": "C",
    "density": "rho",
    "radiogenic_heat": "H",
    "pre-exponential_scale_factor": "A",
    "power_law_exponent": "n",
    "activation_energy": "Q",
    "activation_volume": "V",
}

TEMPERATURE_HEADER = "T1 \n T2 \n T3 \n T4"

def make_coordinates(region, shape):
    """
    Create grid coordinates for 2D and 3D models

    Parameters
    ----------
    region : tuple or list
        List containing the boundaries of the region of the grid. If the grid 
        is 2D, the boundaries should be passed in the following order:
        ``x_min``, ``x_max``,``z_min``, ``z_max``.
        If the grid is 3D, the boundaries should be passed in the following 
        order:
        ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, ``z_max``.
    shape : tuple
        Total number of grid nodes along each direction.
        If the grid is 2D, the tuple must be: ``n_x``, ``n_z``.
        If the grid is 3D, the tuple must be: ``n_x``, ``n_y``, ``n_z``.

    Returns
    -------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid.
    """
    # Sanity checks
    _check_region(region)
    _check_shape(region, shape)
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

def make_interface(coordinates, values=[0.0], direction='x'):
    """
    Create an array to represent a 2D or 3D interface.
    
    If a single values is given, creates a horizontal interface with that 
    value as depth. If a list of points is given, creates the interface by 
    linear iterpolation.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid that will be used to create the 
        interface. Must be in meters and can be either 2D or 3D. If they are 
        in 2D, the interface will be a curve, and if the coordinates are 3D, 
        the interface will be a surface.
    values : float (optional), None or list
        Value that will fill the initialized array. If None, the array will be
        filled with ``numpy.nan``s. If a list of vertices is given, it 
        interpolates them.
    direction : string (optional)
        Direction of the subduction. If working in 3D it can be either *"x"* or *"y"*.
        When working in 2D, it must be *"x"*.

    Returns
    -------
    arr : :class:`xarray.DataArray`
        Array containing the interface.
    """
    # Get shape of coordinates
    shape = _get_shape(coordinates)
    
    cond_1 = not isinstance(values, list)
    cond_2 = (isinstance(values, list)) and (len(values)==1)
    cond_3 = (isinstance(values, list)) and (len(values)>1)
    
    if (cond_1) or (cond_2):
        islist = False
    elif (cond_3):
        islist = True
    else:
        raise ValueError("Unrecognized values for making interface.")
        
    if (islist==False):
        if values is None:
            values = np.nan
        # Remove the shape on z
        shape = shape[:-1]
        arr = xr.DataArray(
            values * np.ones(shape),
            coords = [coordinates[i] for i in coordinates if i != "z"],
        )
    elif (islist==True):
        h_min, h_max = coordinates[direction].min(), coordinates[direction].max()
        values = np.array(values)
        # _check_boundary_vertices(values, h_min, h_max)
        interface = np.interp(coordinates[direction], values[:, 0], values[:, 1])
        arr = xr.DataArray(interface, coords=[coordinates[direction]], dims=direction)
        if len(coordinates.dims) == 3:
            if direction == "x":
                missing_dim = "y"
            elif direction == "y":
                missing_dim == "x"
            arr = arr.expand_dims({missing_dim: coordinates[missing_dim].size})
            arr.coords[missing_dim] = coordinates[missing_dim]
            arr = arr.transpose("x", "y")
    return arr

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

def save_interfaces(interfaces, parameters, path, fname='interfaces.txt'):
    """
    Save the interfaces and the rheological parameters as an ASCII file.

    Parameters
    ----------
    interfaces : :class:`xarray.Dataset`
        Dataset with the interfaces depth.
    parameters : dict
        Dictionary with the parameters values for each lithological unit.
        The necessary parameters are:
            - ``compositional factor``,
            - ``density``,
            - ``radiogenic heat``,
            - ``pre-exponential scale factor``,
            - ``power law exponent``,
            - ``activation energy``,
            - ``activation volume``
    path : str
        Path to save the file.
    fname : str (optional)
        Name to save the interface file. Default ``interface.txt``
    """
    # Check if givens parameters are consistent
    _check_necessary_parameters(parameters, interfaces)

    # Generate the header with the layers parameters
    header = []
    for parameter in parameters:
        header.append(
            PARAMETERS_DICT[parameter]
            + " "
            + " ".join(list(str(i) for i in parameters[parameter]))
        )
    header = "\n".join(header)
    dimension = len(interfaces.dims)
    expected_dims = "x"
    interfaces = interfaces.transpose(*expected_dims)
    # Stack and ravel the interfaces from the dataset
    # We will use order "F" on numpy.ravel in order to make the first index to change
    # faster than the rest
    stacked_interfaces = np.hstack(
        list(interfaces[i].values.ravel(order="F")[:, np.newaxis] for i in interfaces)
    )
    # Save the interface and the layers parameters
    np.savetxt(
        os.path.join(path, fname),
        stacked_interfaces,
        fmt="%f",
        header=header,
        comments="",
    )
    
def make_grid(coordinates, value=0):
    """
    Create an empty grid for a set of coordinates.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid where the temperature distribution will be
        created. Must be in meters and can be either 2D or 3D.
    value : float (optional) or None
        Value that will fill the initialized array. If None, the array will be filled
        with ``numpy.nan``s. Default to 0.
    """
    # Get shape of coordinates
    shape = _get_shape(coordinates)
    if value is None:
        value = np.nan
    return xr.DataArray(value * np.ones(shape), coords=coordinates)

def save_temperature(temperatures, path, fname="input_temperature_0"):
    """
    Save the temperature grid as an ASCII file.

    The temperatures grid values are saved on a single column, following each axis
    in increasing order, with the ``x`` indexes changing faster that the ``z``.

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
            "Invalid temperature dimensions '{}': ".format(invalid_dims)
            + "must be '{}' for a 2D temperature grid.".format(expected_dims)
        )
    # Change order of temperature dimensions to ("x", "z") to ensure
    # right order of elements when the array is ravelled
    temperatures = temperatures.transpose(*expected_dims)
    # Ravel and save temperatures
    # We will use order "F" on numpy.ravel in order to make the first index to change
    # faster than the rest
    # Will add a custom header required by MANDYOC
    np.savetxt(
        os.path.join(path, fname), temperatures.values.ravel(order="F"), header=TEMPERATURE_HEADER
    )

def _check_necessary_parameters(parameters, interfaces):
    """
    Check if there all parameters are given (not checking number).
    """
    for parameter in PARAMETERS_DICT:
        if parameter not in parameters:
            raise ValueError(
                "Parameter '{}' missing. ".format(parameter)
                + "All the following parameters must be included:"
                + "\n    "
                + "\n    ".join([str(i) for i in PARAMETERS_DICT.keys()])
            )
            
    """
    Check if the number of parameters is correct for each lithological unit.
    """
    sizes = list(len(i) for i in list(parameters.values()))
    if not np.allclose(sizes[0], sizes):
        raise ValueError(
            "Missing parameters for the lithological units. "
            + "Check if each lithological unit has all the parameters."
        )  
    """
    Check if the number of parameters is equal to the number of lithological units.
    """
    size = len(list(parameters.values())[0])
    if not np.allclose(size, len(interfaces) + 1):
        raise ValueError(
            "Invalid number of parameters ({}) for given number of lithological units ({}). ".format(
                size, len(interfaces)
            )
            + "The number of lithological units must be the number of interfaces plus one."
        )
    """
    Check if the interfaces do not cross each other.
    """
    inames = tuple(i for i in interfaces)
    for i in range(len(inames) - 1):
        if not (interfaces[inames[i + 1]] >= interfaces[inames[i]]).values.all():
            raise ValueError(
                "Interfaces are in the wrong order or crossing each other. "
                + "Check interfaces ({}) and ({}). ".format(
                    inames[i], inames[i + 1]
                )
            )


def _check_region(region):
    """
    Sanity checks for region
    """
    if len(region) == 4:
        x_min, x_max, z_min, z_max = region
    elif len(region) == 6:
        x_min, x_max, y_min, y_max, z_min, z_max = region
        if y_min >= y_max:
            raise ValueError(
                "Invalid region domain '{}' (x_min, x_max, z_min, z_max). ".format(region)
                + "Must have y_min =< y_max. "
            )
    else:
        raise ValueError(
            "Invalid number of region domain limits '{}'. ".format(region)
            + "Only 4 or 6 values allowed for 2D and 3D dimensions, respectively."
        )
    if x_min >= x_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have x_min =< x_max. "
        )
    if z_min >= z_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have z_min =< z_max. "
        )


def _check_shape(region, shape):
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
        
def _get_shape(coordinates):
    """
    Return the shape of ``coordinates``.

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

def _check_boundary_vertices(values, h_min, h_max):
    """
    Check if the boundary vertices match the boundary coordinates.
    """
    h = values[:, 0]
    if not np.allclose(h_min, h.min()) or not np.allclose(h_max, h.max()):
        raise ValueError(
            "Invalid vertices for creating the interfaces: {}. ".format(values)
            + "Remember to include boundary nodes that matches the coordinates "
            + "boundaries '{}.'".format((h_min, h_max))
        )