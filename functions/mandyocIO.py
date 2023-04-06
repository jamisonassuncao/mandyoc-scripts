"""
Libraries to handle input and output from Mandyoc code.
"""
import glob
import os
import gc
import numpy as np
import xarray as xr

SEG = 365. * 24. * 60. * 60.

PARAMETERS = {
    "compositional_factor": "C",
    "density": "rho",
    "radiogenic_heat": "H",
    "pre-exponential_scale_factor": "A",
    "power_law_exponent": "n",
    "activation_energy": "Q",
    "activation_volume": "V",
}

TEMPERATURE_HEADER = "T1\nT2\nT3\nT4"

OUTPUTS = {
    "temperature": "temperature",
    "density": "density",
    "radiogenic_heat": "heat",
    "viscosity": "viscosity",
    "strain": "strain",
    "strain_rate": "strain_rate",
    "pressure": "pressure",
    "surface": "surface",
    "velocity": "velocity",
}

OUTPUT_TIME = "time_"

PARAMETERS_FNAME = "param.txt"

# Define which datasets are scalars measured on the nodes of the grid, e.g.
# surface and velocity are not scalars.
SCALARS = tuple(OUTPUTS.keys())[:7]

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
            PARAMETERS[parameter]
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

def save_temperature(temperatures, path, fname="input_temperature_0.txt"):
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
    
def read_mandyoc_output(model_path, parameters_file=PARAMETERS_FNAME, datasets=tuple(OUTPUTS.keys()), steps_slice=None, save_big_dataset=False):
    """
    Read the files  generate by Mandyoc code
    Parameters
    ----------
    model_path : str
        Path to the folder where the Mandyoc files are located.
    parameters_file : str (optional)
        Name of the parameters file. It must be located inside the ``path``
        directory.
        Default to ``"param.txt"``.
    datasets : tuple (optional)
        Tuple containing the datasets that wants to be loaded.
        The available datasets are:
            - ``temperature``
            - ``density"``
            - ``radiogenic_heat``
            - ``strain``
            - ``strain_rate``
            - ``pressure``
            - ``viscosity``
            - ``velocity``
            - ``surface``
        By default, every dataset will be read.
    steps_slice : tuple
        Slice of steps to generate the step array. If it is None, it is taken
        from the folder where the Mandyoc files are located.
    save_big_dataset : bool
        Save all datasets in a single dataset. Recomended to small models
    filetype : str
        Files format to be read. Default to ``"ascii"``.
    Returns
    -------
    dataset :  :class:`xarray.Dataset`
        Dataset containing data generated by Mandyoc code.
    """
    # Read parameters
    parameters = _read_parameters(os.path.join(model_path, parameters_file))
    # Build coordinates
    shape = parameters["shape"]
    aux_coords = make_coordinates(region=parameters["region"], shape=shape)
    coordinates = np.array(aux_coords["x"]), np.array(aux_coords["z"])
    # Get array of times and steps
    steps, times = _read_times(model_path, parameters["print_step"], parameters["step_max"], steps_slice)
    end = np.size(times)
    # Create the coordinates dictionary containing the coordinates of the nodes
    # and the time and step arrays. Then create data_vars dictionary containing
    # the desired scalars datasets.
    coords = {"time": times, "step": ("time", steps)}
    dims = ("time", "x", "z")
    profile_dims = ("time", "x")
    coords["x"], coords["z"] = coordinates[:]

    print(f"Starting...")
    datasets_aux = []
    for scalar in SCALARS:
        if scalar in datasets:
            datasets_aux.append(scalar)
            scalars = _read_scalars(model_path, shape, steps, quantity=scalar)
            data_aux = {scalar: (dims, scalars)}
            xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_{scalar}.nc")
            print(f"{scalar.capitalize()} files saved.")
            del scalars
            del data_aux
            gc.collect()
    
    
    # Read surface if needed
    if "surface" in datasets:
        datasets_aux.append("surface")
        surface = _read_surface(model_path, shape[0], steps)
        data_aux = {"surface": (profile_dims, surface)}
        xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_surface.nc")
        print(f"Surface files saved.")
        del surface
        del data_aux
        gc.collect()

    # Read velocity if needed
    if "velocity" in datasets:
        datasets_aux.append("velocity")
        velocities = _read_velocity(model_path, shape, steps)
        data_aux = {}
        data_aux["velocity_x"] = (dims, velocities[0])
        data_aux["velocity_z"] = (dims, velocities[1])
        xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_velocity.nc")
        print(f"Velocity files saved.")
        del velocities
        del data_aux
        gc.collect()

    print(f"All files read and saved.")
    # return xr.Dataset(data_vars, coords=coords, attrs=parameters)  
    
#     empty_dataset = True
#     for item in datasets_aux:
#         dataset_aux = xr.open_dataset(f"{model_path}/_output_{item}.nc")
#         if (empty_dataset == True):
#             dataset = dataset_aux
#             empty_dataset = False
#         else:
#             dataset = dataset.merge(dataset_aux)
#         del dataset_aux
#     gc.collect()

#     dataset.to_netcdf(f"{model_path}/data.nc", format="NETCDF3_64BIT")
    # del dataset
    # gc.collect()
    
    return datasets_aux

def read_datasets(model_path, datasets, save_big_dataset=False):
    empty_dataset = True
    for item in datasets:
        dataset_aux = xr.open_dataset(f"{model_path}/_output_{item}.nc")
        if (empty_dataset == True):
            dataset = dataset_aux
            empty_dataset = False
        else:
            dataset = dataset.merge(dataset_aux)
        del dataset_aux
    gc.collect()

    if (save_big_dataset):
        print(f'Saving dataset with all Mandyoc data')
        dataset.to_netcdf(f"{model_path}/data.nc", format="NETCDF3_64BIT")
        print(f"Big dataset file saved.")
        gc.collect()
    
    return dataset

def diffuse_field(field, cond_air, kappa, dx, dz, t_max=1.0E6, fac=100):
    """
    Calculates the diffusion of a 2D field using finite difference.
    ----------
    field : numpy.ndarray
        2D field that will be diffused.
    conda_air : numpy.ndarray
        2D box where value will be constant.
    kappa : float
        Thermal diffusivity coefficient.
    dx: float
        Spacing in the x (horizontal) direction.
    dz: float
        Spacing in the z (vertical) direction.
    t_max: float
        Maximum diffusion time in years.
    fac: int
        Number of time steps to diffuse the field.
    Returns
    -------
    field :  numpy.ndarray
        2D array containing the diffused field.
    """
    dx_aux = np.min([np.abs(dx),np.abs(dz)])
    dt = np.min([dx_aux**2./(2.*kappa), t_max/fac])
    
    CTx = kappa * dt * SEG / (dx**2)
    CTz = kappa * dt * SEG / (dz**2)
    
    t = 0.0
    while (t<=t_max):
        auxX = field[2:,1:-1] + field[:-2,1:-1] - 2 * field[1:-1,1:-1]
        auxZ = field[1:-1,2:] + field[1:-1,:-2] - 2 * field[1:-1,1:-1]
        field[1:-1,1:-1] = field[1:-1,1:-1] + (CTx * auxX) + (CTz * auxZ)
        # boundary conditions
        field[:,cond_air] = 0.0
        field[0,:] = field[1,:]
        field[-1,:] = field[-2,:]
        # time increment
        t += dt
    return field

def _read_scalars(path, shape, steps, quantity):
    """
    Read Mandyoc scalar data
    Read ``temperature``, ``density``, ``radiogenic_heat``, ``viscosity``,
    ``strain``, ``strain_rate`` and ``pressure``.
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc files are located.
    shape: tuple
        Shape of the expected grid.
    steps : array
        Array containing the saved steps.
    quantity : str
        Type of scalar data to be read.
    Returns
    -------
    data: np.array
        Array containing the Mandyoc scalar data.
    """
    print(f"Reading {quantity} files...", end=" ")
    data = []
    for step in steps:
        filename = "{}_{}".format(OUTPUTS[quantity], step)
        data_step = np.loadtxt(
            os.path.join(path, filename + ".txt"),
            unpack=True,
            comments="P",
            skiprows=2,
        )
        # Convert very small numbers to zero
        data_step[np.abs(data_step) < 1.0e-200] = 0
        # Reshape data_step
        data_step = data_step.reshape(shape, order="F")
        # Append data_step to data
        data.append(data_step)
    data = np.array(data)
    print(f"{quantity.capitalize()} files read.", end=" ")
    return data

def _read_velocity(path, shape, steps):
    """
    Read velocity data generated by Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc output files are located.
    shape: tuple
        Shape of the expected grid.
    steps : array
        Array containing the saved steps.
    Returns
    -------
    data: tuple of arrays
        Tuple containing the components of the velocity vector.
    """
    print(f"Reading velocity files...", end=" ")
    # Determine the dimension of the velocity data
    dimension = len(shape)
    velocity_x, velocity_z = [], []
    for step in steps:
        filename = "{}_{}".format(OUTPUTS["velocity"], step)
        velocity = np.loadtxt(
            os.path.join(path, filename + ".txt"), comments="P", skiprows=2
        )
        # Convert very small numbers to zero
        velocity[np.abs(velocity) < 1.0e-200] = 0
        # Separate velocity into their three components
        velocity_x.append(velocity[0::dimension].reshape(shape, order="F"))
        velocity_z.append(velocity[1::dimension].reshape(shape, order="F"))
    # Transform the velocity_* lists to arrays
    velocity_x = np.array(velocity_x)
    velocity_z = np.array(velocity_z)
    print(f"Velocity files read.", end=" ")
    return (velocity_x, velocity_z)

def _read_surface(path, size, steps):
    """
    Read surface data generated by the Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc output files are located.
    size : int
        Size of the surface profile.
    steps : array
        Array containing the saved steps.
    Returns
    -------
    data : np.array
        Array containing the Mandyoc profile data.
    """
    print(f"Reading surface files...", end=" ")
    data = []
    for step in steps:
        filename = "sp_surface_global_{}".format(step)
        data_step = np.loadtxt(
            os.path.join(path, filename + ".txt"),
            unpack=True,
            comments="P",
            skiprows=2,
        )
        # Convert very small numbers to zero
        data_step[np.abs(data_step) < 1.0e-200] = 0
        # Reshape data_step
        # data_step = data_step.reshape(shape, order="F")
        # Append data_step to data
        data.append(data_step)
    data = np.array(data)
    print(f"Surface files read.", end=" ")
    return data
            
def _read_parameters(parameters_file):
    """
    Read parameters file
    .. warning :
        The parameters file contains the length of the region along each axe.
        While creating the region, we are assuming that the z axe points upwards
        and therefore all values beneath the surface are negative, and the x
        and y axes are all positive within the region.
    Parameters
    ----------
    parameters_file : str
        Path to the location of the parameters file.
    Returns
    -------
    parameters : dict
        Dictionary containing the parameters of Mandyoc files.
    """
    parameters = {}
    with open(parameters_file, "r") as params_file:
        for line in params_file:
            # Skip blank lines
            if not line.strip():
                continue
            if line[0] == "#":
                continue
            # Remove comments lines
            line = line.split("#")[0].split()
            var_name, var_value = line[0], line[2]
            parameters[var_name.strip()] = var_value.strip()
        # Add shape
        parameters["shape"] = (int(parameters["nx"]), int(parameters["nz"]))
        # Add dimension
        parameters["dimension"] = len(parameters["shape"])
        # Add region
        parameters["region"] = (
            0,
            float(parameters["lx"]),
            -float(parameters["lz"]),
            0,
        )
        parameters["step_max"] = int(parameters["step_max"])
        parameters["time_max"] = float(parameters["time_max"])
        parameters["print_step"] = int(parameters["step_print"])
        # Add units
        parameters["coords_units"] = "m"
        parameters["times_units"] = "Ma"
        parameters["temperature_units"] = "C"
        parameters["density_units"] = "kg/m^3"
        parameters["heat_units"] = "W/m^3"
        parameters["viscosity_units"] = "Pa s"
        parameters["strain_rate_units"] = "s^(-1)"
        parameters["pressure_units"] = "Pa"
    return parameters

def _read_times(path, print_step, max_steps, steps_slice):
    """
    Read the time files generated by Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc files are located.
    print_step : int
        Only steps multiple of ``print_step`` are saved by Mandyoc.
    max_steps : int
        Maximum number of steps. Mandyoc could break computation before the
        ``max_steps`` are run if the maximum time is reached. This quantity only
        bounds the number of time files.
    steps_slice : tuple
        Slice of steps (min_steps_slice, max_steps_slice). If it is None,
        min_step_slice = 0 and max_steps_slice = max_steps.
    Returns
    -------
    steps : numpy array
        Array containing the saved steps.
    times : numpy array
        Array containing the time of each step in Ma.
    """
    steps, times = [], []
    # Define the mininun and maximun step
    if steps_slice is not None:
        min_steps_slice, max_steps_slice = steps_slice[:]
    else:
        min_steps_slice, max_steps_slice = 0, max_steps
    for step in range(min_steps_slice, max_steps_slice + print_step, print_step):
        filename = os.path.join(path, "{}{}.txt".format(OUTPUT_TIME, step))
        if not os.path.isfile(filename):
            break
        time = np.loadtxt(filename, unpack=True, delimiter=":", usecols=(1))
        if time.shape == ():
            times.append(time)
        else:
            time = time[0]
            times.append(time)
        steps.append(step)

    # Transforms lists to arrays
    times = 1e-6 * np.array(times)  # convert time units into Ma
    steps = np.array(steps, dtype=int)
    return steps, times

def _check_necessary_parameters(parameters, interfaces):
    """
    Check if there all parameters are given (not checking number).
    """
    for parameter in PARAMETERS:
        if parameter not in parameters:
            raise ValueError(
                "Parameter '{}' missing. ".format(parameter)
                + "All the following parameters must be included:"
                + "\n    "
                + "\n    ".join([str(i) for i in PARAMETERS.keys()])
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

def read_particle_path(path, position, unit_number=np.nan, ncores=np.nan):
    """
    Follow a particle through time.
    
    Parameters
    ----------
    path : str
        Path to read the data.
    position : tuple
        (x, z) position in meters of the particle to be followed. 
        The closest particle will be used.
    unit_number : int
        Lithology number of the layer.
    ncores: int
        Number of cores used during the simulation, necessary to read the files properly.
    
    Return
    ------
    particle_path : array (legth N)
        Position of the particle in each time step.
    """
    
    # check ncores
    if (np.isnan(ncores)):
        aux = glob.glob(f"{path}/step_0_*.txt")
        ncores = np.size(aux)
    
    # read first step
    first_x, first_z, first_ID, first_lithology, first_strain = _read_step(path, "step_0_", ncores)
    
    # order closest points to <position> at the first step
    pos_x, pos_z = position
    dist = np.sqrt(((first_x - pos_x)**2) + ((first_z - pos_z)**2))
    clst = np.argsort(dist)
    
    # read last step
    parameters = _read_parameters(os.path.join(path, PARAMETERS_FNAME))
    nsteps = np.size(glob.glob(f"{path}/time_*.txt"))
    last_step_number = int(parameters["print_step"]*(nsteps-1))
    last_x, last_z, last_ID, last_lithology, last_strain = _read_step(path, f"step_{last_step_number}_", ncores)
    
    # loop through closest poinst while the closest point is not in the last step
    print(f'Finding closest point to x: {pos_x} [m] and z: {pos_z} [m]...')
    cont = 0
    point_in_sim = False
    point_in_lit = False
    while (point_in_sim == False):
        clst_ID = first_ID[clst[cont]]
        # check if closest point is within the desired lithology
        if (np.isnan(unit_number)):
            point_in_lit = True # lithology number does not matter
        else:
            if int(first_lithology[clst[cont]]) == unit_number:
                point_in_lit = True
            else:
                point_in_lit = False # line not necessary (this is for sanity)
                
        # check if closest point is in the last step or find another closer one
        if (clst_ID in last_ID) and (point_in_lit == True):
            print(f'Found point with ID: {first_ID[clst[cont]]}, x: {first_x[clst[cont]]} [m], z: {first_z[clst[cont]]} [m]')
            point_in_sim = True
            closest_ID = clst_ID                
        else:
            print(f'Found point with ID: {first_ID[clst[cont]]}, x: {first_x[clst[cont]]}, z: {first_z[clst[cont]]}')
            print(f'Point DOES NOT persist through the simulation. Finding another one...')
            cont += 1

    # read all steps storing the point position
    print("Reading step files...", end=" ")
    x, z = [], []
    for i in range(0, last_step_number, parameters["print_step"]):
        current_x, current_z, current_ID, current_lithology, current_strain = _read_step(path, f"step_{i}_", ncores)
        arg = np.where(current_ID == closest_ID)
        x = np.append(x, current_x[arg])
        z = np.append(z, current_z[arg])
    print("Step files read.")
        
    return x, z, closest_ID

def _read_step(path, filename, ncores):
    """
    Read a step file.
    
    Parameters
    ----------
    path : str
        Path to read the data.
    filename : str
        Auxiliary file name.
    ncores : int
        Number of cores the simulation used.
        
    Return
    ------
    data_x : array (Length N)
        Array containing the position x of the particle.
    data_z : array (Length N)
        Array containing the position z of the particle.
    data_ID : array (Length N)
        Array containing the ID of the particle.
    data_lithology : array (Length N)
        Array containing the number of the particle lithology of the particle.
    data_strain : array (Length N)
        Array containing the strain of the particle.
    """
    data_x, data_z, data_ID, data_lithology, data_strain = [], [], [], [], []
    for i in range(ncores):
        try:
            aux_x, aux_z, aux_ID, aux_lithology, aux_strain = np.loadtxt(os.path.join(path, filename + str(i) + ".txt"), unpack=True, comments="P")
        except:
            continue
        data_x = np.append(data_x, aux_x)
        data_z = np.append(data_z, aux_z)
        data_ID = np.append(data_ID, aux_ID)
        data_lithology = np.append(data_lithology, aux_lithology)
        data_strain = np.append(data_strain, aux_strain)
    return data_x, data_z, data_ID, data_lithology, data_strain