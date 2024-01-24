"""
Libraries to handle input and output from Mandyoc code.
"""
import glob
import os
import gc
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.mlab as mlab
import matplotlib.colors


SEG = 365. * 24. * 60. * 60.

PARAMETERS = {
    "compositional_factor": "C",
    "density": "rho",
    "radiogenic_heat": "H",
    "pre-exponential_scale_factor": "A",
    "power_law_exponent": "n",
    "activation_energy": "Q",
    "activation_volume": "V"
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

def save_interfaces(interfaces, parameters, path, strain_softening, fname='interfaces.txt'):
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
            - ``activation volume``,
            - ``weakening seed``
            - ``cohesion max``
            - ``cohesion min``
            - ``friction angle min``
            - ``friction angle max``
    path : str
        Path to save the file.
    fname : str (optional)
        Name to save the interface file. Default ``interface.txt``
    """
    # Check if givens parameters are consistent
    _check_necessary_parameters(parameters, interfaces, strain_softening)

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
    
def read_mandyoc_output(model_path, parameters_file=PARAMETERS_FNAME, datasets=tuple(OUTPUTS.keys()), skip=1, steps_slice=None, save_big_dataset=False):
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
    skip: int
        Reads files every <skip> value to save mamemory.
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
    steps = steps[::skip]
    times = times[::skip]
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
        # del dataset
        # gc.collect()
    
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

def build_parameter_file(**kwargs):
    """
    Creates a parameters dicitionary containing all the parameters that will be
    necessary fot the simulation.

    Parameters
    ----------
    **kwargs : dict
        Arguments used in the paramenter file.

    Returns
    -------
    params : dict
        Complete parameter file dictionary.
    """
    defaults = {
        'nx' : None,
        'nz' : None,
        'lx' : None,
        'lz' : None, 

        'aux00': '# Simulation options',
        'solver' : 'direct',
        'denok' : 1.0e-10,
        'rtol' : 1.0e-7,
        'RK4' : 'Euler',
        'Xi_min' : 1.0e-7,
        'random_initial_strain' : 0.0,
        'pressure_const' : -1.0,
        'initial_dynamic_range' : True,
        'periodic_boundary' : False,
        'high_kappa_in_asthenosphere' : False,
        'basal_heat' : -1.0,

        'aux01': '# Particles options',
        'particles_per_element' : 81,
        'particles_per_element_x' : 0,
        'particles_per_element_z' : 0,
        'particles_perturb_factor' : 0.7,

        'aux02': '# Surface processes',
        'sp_surface_tracking' : False,
        'sea_level' : 0.0,
        'sp_surface_processes' : False,
        'sp_dt' : 0,
        'a2l' : True,
        'sp_mode' : 1,
        'free_surface_stab' : True,
        'theta_FSSA' : 0.5,
        'sticky_blanket_air' : False,
        'precipitation_profile_from_ascii' : False,
        'climate_change_from_ascii' : False,

        'aux03': '# Time constrains',
        'step_max' : 7000,
        'time_max' : 10.0e6,
        'dt_max' : 10.0e3,
        'step_print' : 10,
        'sub_division_time_step' : 1.0,
        'initial_print_step' : 0,
        'initial_print_max_time' : 1.0e6,

        'aux04': '# Viscosity',
        'viscosity_reference' : None,
        'viscosity_max' : None,
        'viscosity_min' : None,
        'viscosity_per_element' : 'constant',
        'viscosity_mean_method' : 'arithmetic',
        'viscosity_dependence' : 'pressure',

        'aux05': '# External ASCII inputs/outputs',
        'interfaces_from_ascii' : True,
        'n_interfaces' : None,
        'temperature_from_ascii' : True,
        'velocity_from_ascii' : False,
        'variable_bcv' : False,
        'multi_velocity' : False,
        'binary_output' : False,
        'print_step_files' : True,

        'aux06': '# Physical parameters',
        'temperature_difference' : None,
        'thermal_expansion_coefficient' : None,
        'thermal_diffusivity_coefficient' : None,
        'gravity_acceleration' : None,
        'density_mantle' : 3300.,
        'heat_capacity' : None,
        'adiabatic_component' : None,
        'radiogenic_component' : None,

        'aux07': '# Strain softening',
        'non_linear_method' : 'on',
        'plasticity' : 'on',
        'weakening_min' : 0.05,
        'weakening_max' : 1.05,

        'aux08': '# Velocity boundary conditions',
        'top_normal_velocity' : 'fixed',
        'top_tangential_velocity' : 'free',
        'bot_normal_velocity' : 'fixed',
        'bot_tangential_velocity' : 'free',
        'left_normal_velocity' : 'fixed',
        'left_tangential_velocity' : 'free',
        'right_normal_velocity' : 'fixed ',
        'right_tangential_velocity' : 'free',

        'aux09': '# Temperature boundary conditions',
        'top_temperature' : 'fixed',
        'bot_temperature' : 'fixed',
        'left_temperature' : 'fixed',
        'right_temperature' : 'free',
        'rheology_model' : 19,
        'T_initial' : 0,
    }

    params = {}
    for key, value in defaults.items():
        param = kwargs.get(key, value)
        if param is None:
            raise ValueError(f"The parameter '{key}' is mandatory.")
        params[key] = str(param)
    return params

def save_parameter_file(params, run_dir):
    """
    Saves the parameter dictionary into a file called param.txt

    Parameters
    ----------

    params : dict
        Dictionary containing the parameters of the param.txt file.
    """
    # Create the parameter file
    with open(os.path.join(run_dir,"param.txt"), "w") as f:
        for key, value in params.items():
            if key[:3] == "aux":
                f.write(f"\n{value}\n")
            else:
                f.write('{:<32} = {}\n'.format(key, value))

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

def _check_necessary_parameters(parameters, interfaces, strain_softening):
    """
    Check if there all parameters are given (not checking number).
    """
    if (strain_softening):
        PARAMETERS['weakening_seed'] = 'weakening_seed'
        PARAMETERS['cohesion_min'] = 'cohesion_min'
        PARAMETERS['cohesion_max'] = 'cohesion_max'
        PARAMETERS['friction_angle_min'] = 'friction_angle_min'
        PARAMETERS['friction_angle_max'] = 'friction_angle_max'
    else:
        PARAMETERS.pop('weakening_seed', None)
        PARAMETERS.pop('cohesion_min', None)
        PARAMETERS.pop('cohesion_max', None)
        PARAMETERS.pop('friction_angle_min', None)
        PARAMETERS.pop('friction_angle_max', None)

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
            aux_x, aux_z, aux_ID, aux_lithology, aux_strain = np.loadtxt(os.path.join(path, f"{filename}{str(i)}.txt"), unpack=True, comments="P")
        except:
            print('didnt read')
            continue
        data_x = np.append(data_x, aux_x)
        data_z = np.append(data_z, aux_z)
        data_ID = np.append(data_ID, aux_ID)
        data_lithology = np.append(data_lithology, aux_lithology)
        data_strain = np.append(data_strain, aux_strain)
    return np.asarray(data_x), np.asarray(data_z), np.asarray(data_ID), np.asarray(data_lithology), np.asarray(data_strain)


def _extract_interface(z, Z, Nx, Rhoi, rho):
    '''
    Extract interface from Rhoi according to a given density (rho)

    Parameters
    ----------
    z: numpy array
        Array representing z direction.

    Z: numpy array
        Array representing z direction resampled with higher resolution.

    Nx: int
        Number of points in x direction.

    Rhoi: numpy array (Nz, Nx)
        Density field from mandyoc

    rho: int
        Value of density to be searched in Rhoi field
        
    Returns
    -------
    topo_aux: numpy array
        Array containing the extacted interface.
    '''

    topo_aux = []
    
    for j in np.arange(Nx):
        topoi = interp1d(z, Rhoi[:,j]) #return a "function" of interpolation to apply in other array
        idx = (np.abs(topoi(Z)-rho)).argmin()
        topo = Z[idx]
        topo_aux = np.append(topo_aux, topo)

    return topo_aux

def _log_fmt(x, pos):
    return "{:.0f}".format(np.log10(x))

def change_dataset(properties, datasets):
    '''
    Create new_datasets based on the properties that will be plotted
    
    Parameters
    ----------
    properties: list of strings
        Properties to plot.

    datasets: list of strings
        List of saved properties.

    Returns
    -------
    new_datasets: list of strings
        New list of properties that will be read.
    '''
    
    new_datasets = []
    for prop in properties:
        if (prop in datasets) and (prop not in new_datasets):
            new_datasets.append(prop)
        if (prop == "lithology") and ("strain" not in new_datasets):
            new_datasets.append("strain")
        if (prop == "temperature_anomaly") and ("temperature" not in new_datasets):
            new_datasets.append("temperature")
            
        if (prop == "lithology" or prop == 'temperature_anomlay') and ("density" not in new_datasets):
            new_datasets.append("density")
            
    return new_datasets
def _calc_melt_dry(To,Po):

    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1394 + 0.132899*P - 0.000005104*P**2
    cond = P>10000.0
    Tsol[cond] = 2212 + 0.030819*(P[cond] - 10000.0)

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq) #melt
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol) #partial melt
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)

def _calc_melt_wet(To,Po):
    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1240 + 49800/(P + 323)
    cond = P>2400.0
    Tsol[cond] = 1266 - 0.0118*P[cond] + 0.0000035*P[cond]**2

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq)
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol)
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)

def single_plot(dataset, prop, xlims, ylims, model_path, output_path, save_frames=True, plot_isotherms=True, plot_particles=False, isotherms = [400, 600, 800, 1000, 1300], plot_melt=False, melt_method='dry'):
    '''
    Plot and save data from mandyoc according to a given property and domain limits.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        Dataset containing mandyoc data for a single time step.
    
    prop: str
        Property from mandyoc.

    xlims: list
        List with the limits of x axis
        
    ylims: list
        List with the limits of y axis

    model_path: str
        Path to model

    output_path: str
        Path to save outputs
        
    save_frames: bool
        True to save frame by frames
        False to do not save the frames
    '''
    
    props_label = {'density':              r'$\mathrm{[kg/m^3]}$',
                   'radiogenic_heat':       'log(W/kg)',
                   'lithology':            r'log$(\epsilon_{II})$',
                   'pressure':              'P [GPa]',
                   'strain':               r'Accumulated strain [$\varepsilon$]',
                   'strain_rate':          r'log($\dot{\varepsilon}$)',
#                    'strain_rate':          r'$\dot{\varepsilon}$',
                   'temperature':          r'$^{\circ}\mathrm{[C]}$',
                   'temperature_anomaly':  r'Temperature anomaly $^{\circ}\mathrm{[C]}$',
                   'topography':            'Topography [km]',
                   'viscosity':             'log(Pa.s)'
                   }
    
    props_cmap = {'density': 'viridis',
                  'radiogenic_heat': 'inferno',
                  'lithology': 'viridis',
                  'pressure': 'viridis',
#                   'strain': 'viridis', #Default. Comment this line and uncomment one of the options bellow
#                   'strain': 'cividis',
#                   'strain': 'Greys',
                  'strain': 'inferno',
#                   'strain': 'magma',
                  'strain_rate': 'viridis',
                  'temperature': 'viridis',
                  'temperature_anomaly': 'RdBu_r',
                  'topography': '',
                  'viscosity': 'viridis'
                   }

    #limits of colorbars
    vals_minmax = {'density':             [0.0, 3378.],
                   'radiogenic_heat':     [1.0E-13, 1.0E-9],
                   'lithology':           [None, None],
                   'pressure':            [-1.0E-3, 1.0],
                   'strain':              [None, None],
                   'strain_rate':         [1.0E-19, 1.0E-14],
#                    'strain_rate':         [np.log10(1.0E-19), np.log10(1.0E-14)],
                   'temperature':         [0, 1600],
                   'temperature_anomaly': [-150, 150],
                   'surface':             [-6, 6],
                   'viscosity':           [1.0E18, 1.0E25],
#                    'viscosity':           [np.log10(1.0E18), np.log10(1.0E25)]
                  }

    model_name = model_path.split('/')[-1] #os.path.split(model_path)[0].split('/')[-1]

    Nx = int(dataset.nx)
    Nz = int(dataset.nz)
    Lx = float(dataset.lx)
    Lz = float(dataset.lz)
    instant = np.round(float(dataset.time), 2)
    
    xi = np.linspace(0, Lx/1000, Nx)
    zi = np.linspace(-Lz/1000+40, 0+40, Nz) #km, +40 to compensate the air layer above sea level
    xx, zz = np.meshgrid(xi, zi)
    
    #creating Canvas
    plt.close()
    label_size=12
    plt.rc('xtick', labelsize = label_size)
    plt.rc('ytick', labelsize = label_size)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12*(Lz/Lx)), constrained_layout = True)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout = True)
    #plot Time in Myr
    ax.text(0.8, 0.85, ' {:01} Myr'.format(instant), fontsize = 18, zorder=52, transform=ax.transAxes)
    
    val_minmax = vals_minmax[prop]
    
    if(plot_isotherms == True and prop != 'topography'): #Plot isotherms
        Temperi = dataset.temperature.T
        
        isot_colors = []
        for isotherm in isotherms:
            isot_colors.append('red')
            
        cs = ax.contour(xx, zz, Temperi, 100, levels=isotherms, colors=isot_colors)
        
        # if(instant == instants[0]):
        #     fmt = {}
        #     for level, isot in zip(cs.levels, isotherms):
        #         fmt[level] = str(level) + r'$^{\circ}$C'

        #     ax.clabel(cs, cs.levels, fmt=fmt, inline=True, use_clabeltext=True)
    if(plot_melt == True and prop != 'topography'):
        if(melt_method == 'dry'):
            melt = _calc_melt_dry(dataset.temperature, dataset.pressure)
        elif(melt_method == 'wet'):
            melt = _calc_melt_wet(dataset.temperature, dataset.pressure)

        levels = np.arange(0, 16, 1)
        extent=(0,
                Lx/1.0e3,
                -Lz/1.0e3 + 40,
                0 + 40)
        
        cs = ax.contour(melt.T*100,
                        levels,
                        origin='lower',
                        cmap='inferno',
                        extent=extent,
                        vmin=0, vmax=16,
                        linewidths=0.5,
                        # linewidths=30,
                        zorder=30)

        axmelt = inset_axes(ax,
                            width="20%",  # width: 30% of parent_bbox width
                            height="5%",  # height: 5%
                            bbox_to_anchor=(-0.78,
                                            -0.75,
                                            1,
                                            1),
                            bbox_transform=ax.transAxes,
                            )
        

        norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cs.cmap)
        sm.set_array([])

        cb = fig.colorbar(sm,
                    cax=axmelt,
                    label='melt content [%]',
                    orientation='horizontal',
                    fraction=0.008,
                    pad=0.02)
        cb.ax.tick_params(labelsize=12)
        
    #dealing with special data
    if(prop == 'lithology'):
        data = dataset['strain']
        
    elif(prop == 'temperature_anomaly'):
        #removing horizontal mean temperature
        A = dataset['temperature']
        B = A.T #shape: (Nz, Nx)
        C = np.mean(B, axis=1) #shape: 151==Nz
        D = (B.T - C) #B.T (Nx,Nz) para conseguir subtrair C
        data = D
        
    elif(prop == 'surface'):
        # print('Dealing with data')
#         topo_from_density = True
        topo_from_density = False
        
        if(topo_from_density == True):
            Rhoi = dataset.density.T
            interfaces=[2900, 3365]
            ##Extract layer topography
            z = np.linspace(Lz/1000.0, 0, Nz)
            Z = np.linspace(Lz/1000.0, 0, 8001) #zi
            x = np.linspace(Lx/1000.0, 0, Nx)

            topo_interface = _extract_interface(z, Z, Nx, Rhoi, 300.) #200 kg/m3 = air/crust interface
            
            condx = (xi >= 100) & (xi <= 600)
            z_mean = np.mean(topo_interface[condx])
            
            topo_interface -= np.abs(z_mean)
            topo_interface = -1.0*topo_interface

            data = topo_interface
        else:
            data = dataset.surface/1.0e3 + 40.0 #km + air layer correction
            
            
    elif(prop == 'pressure'):
        data = dataset[prop]/1.0E9 #GPa
        
    else:
        data = dataset[prop] 
        
    if(prop == 'strain_rate' or prop == 'radiogenic_heat' or prop == 'strain_rate' or prop == 'viscosity'): #properties that need a lognorm colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       norm = LogNorm(vmin=val_minmax[0], vmax=val_minmax[1]),
#                        vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        #creating colorbar
        axins1 = inset_axes(ax,
                            loc='lower right',
                            width="100%",  # respective to parent_bbox width
                            height="100%",  # respective to parent_bbox width
                            bbox_to_anchor=(0.7,#horizontal position respective to parent_bbox or "loc" position
                                            0.3,# vertical position
                                            0.25,# width
                                            0.05),# height
                            bbox_transform=ax.transAxes
                            )

        clb = fig.colorbar(im,
                           cax=axins1,
#                            ticks=ticks,
                           orientation='horizontal',
                           fraction=0.08,
                           pad=0.2,
                           format=_log_fmt)

        clb.set_label(props_label[prop], fontsize=12)
        clb.ax.tick_params(labelsize=12)
        clb.minorticks_off()
    
    elif (prop == 'density' or prop == 'pressure' or prop == 'temperature' or prop == 'temperature_anomaly'): #properties that need a regular colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        axins1 = inset_axes(ax,
                            loc='lower right',
                            width="100%",  # respective to parent_bbox width
                            height="100%",  # respective to parent_bbox width
                            bbox_to_anchor=(0.7,#horizontal position respective to parent_bbox or "loc" position
                                            0.3,# vertical position
                                            0.25,# width
                                            0.05),# height
                            bbox_transform=ax.transAxes
                            )
        
#         ticks = np.linspace(val_minmax[0], val_minmax[1], 6, endpoint=True)

        #precision of colorbar ticks
        if(prop == 'pressure'): 
            fmt = '%.2f'
        else:
            fmt = '%.0f'
            
        clb = fig.colorbar(im,
                           cax=axins1,
#                            ticks=ticks,
                           orientation='horizontal',
                           fraction=0.08,
                           pad=0.2,
                           format=fmt)

        clb.set_label(props_label[prop], fontsize=12)
        clb.ax.tick_params(labelsize=12)
        clb.minorticks_off()
        
    elif(prop == 'strain'):
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       vmin = float(data.min()),
                       vmax = float(data.max()),
                       aspect = 'auto')
        
    elif(prop == 'surface'):
        ax.plot(dataset.x/1.0e3, data, alpha = 1, linewidth = 2.0, color = "blueviolet")
        
    elif(prop == 'lithology'): #shaded lithology plot
        cr = 255.
        color_uc = (228. / cr, 156. / cr, 124. / cr)
        color_lc = (240. / cr, 209. / cr, 188. / cr)
        color_lit = (155. / cr, 194. / cr, 155. / cr)
        color_ast = (207. / cr, 226. / cr, 205. / cr)
        
        Rhoi = dataset.density.T
        interfaces=[2900, 3365]
        
        ##Extract layer topography
        z = np.linspace(Lz/1000.0, 0, Nz)
        Z = np.linspace(Lz/1000.0, 0, 8001) #zi
        x = np.linspace(Lx/1000.0, 0, Nx)
            
        topo_interface = _extract_interface(z, Z, Nx, Rhoi, 300.) #200 kg/m3 = air/crust interface
        condx = (xi >= 100) & (xi <= 600)
        z_mean = np.mean(topo_interface[condx])
        topo_interface -= np.abs(z_mean)
        topo_interface = -1.0*topo_interface
        
        ax.contourf(xx,
                    zz,
                    Rhoi,
                    levels = [200., 2750, 2900, 3365, 3900],
                    colors = [color_uc, color_lc, color_lit, color_ast])
        
        im=ax.imshow(data.T,
                     cmap = 'Greys',
                     origin = 'lower',
                     extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40,0 + 40),
                     zorder = 50,
                     alpha = 0.2, vmin=-0.5,
                     vmax = 0.7,
                     aspect = 'auto')
        #legend box
        bv1 = inset_axes(ax,
                        loc='lower right',
                        width="100%",  # respective to parent_bbox width
                        height="100%",  # respective to parent_bbox width
                        bbox_to_anchor=(0.9,#horizontal position respective to parent_bbox or "loc" position
                                        0.3,# vertical position
                                        0.085,# width
                                        0.35),# height
                        bbox_transform=ax.transAxes
                        )
        
        A = np.zeros((100, 10))

        A[:25, :] = 2700
        A[25:50, :] = 2800
        A[50:75, :] = 3300
        A[75:100, :] = 3400

        A = A[::-1, :]

        xA = np.linspace(-0.5, 0.9, 10)
        yA = np.linspace(0, 1.5, 100)

        xxA, yyA = np.meshgrid(xA, yA)
        air_threshold = 200
        bv1.contourf(
            xxA,
            yyA,
            A,
            levels=[air_threshold, 2750, 2900, 3365, 3900],
            colors=[color_uc, color_lc, color_lit, color_ast],
            extent=[-0.5, 0.9, 0, 1.5]
        )

        bv1.imshow(
            xxA[::-1, :],
            extent=[-0.5, 0.9, 0, 1.5],
            zorder=100,
            alpha=0.2,
            cmap=plt.get_cmap("Greys"),
            vmin=-0.5,
            vmax=0.9,
            aspect='auto'
        )

        bv1.set_yticklabels([])
        bv1.set_xlabel(r"log$(\varepsilon_{II})$", size=10)
        bv1.tick_params(axis='x', which='major', labelsize=10)
        bv1.set_xticks([-0.5, 0, 0.5])
        bv1.set_yticks([])
        bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
    
    if(plot_particles == True):
        if(prop != 'surface'):
            ncores = 20
            data_x, data_z, data_ID, data_lithology, data_strain = _read_step(model_path, f"step_{int(dataset.step)}_", ncores)
            # ax.scatter(data_x/1000, data_z/1000, 2, c='xkcd:black', marker='.', zorder=30)

            cond_litho = data_lithology > 0
            cond_ast = data_lithology == 0

            # if(prop=='lithology'):
            #     color_litho = 'xkcd:bright pink'
            #     color_ast = 'xkcd:black'
            # else:
            #     color_litho = 'xkcd:bright green'
            #     color_ast = 'xkcd:black'

            color_litho = 'xkcd:black'
            color_ast = 'xkcd:bright pink'

            ax.plot(data_x[cond_litho]/1000, data_z[cond_litho]/1000, "o", color=color_litho, markersize=0.2, alpha=1.0, zorder=30)
            ax.plot(data_x[cond_ast]/1000, data_z[cond_ast]/1000, "o", color=color_ast, markersize=0.2, alpha=1.0, zorder=30)
        # else:
        #     print('Error: You cannot print particles in the Surface plot!')
        #     return()
    
    if(prop != 'surface'):
        #Filling above topographic surface
        Rhoi = dataset.density.T
        interfaces=[2900, 3365]
        ##Extract layer topography
        z = np.linspace(Lz/1000.0, 0, Nz)
        Z = np.linspace(Lz/1000.0, 0, 8001) #zi
        x = np.linspace(Lx/1000.0, 0, Nx)

        topo_interface = _extract_interface(z, Z, Nx, Rhoi, 300.) #200 kg/m3 = air/crust interface
        condx = (xi >= 100) & (xi <= 600)
        z_mean = np.mean(topo_interface[condx])
        topo_interface -= np.abs(z_mean)
        topo_interface = -1.0*topo_interface

        xaux = xx[0]
        condaux = (xaux>xlims[0]) & (xaux<xlims[1])
        xaux = xaux[condaux]

        ax.fill_between(xaux, topo_interface[condaux], 39, color='white', alpha=1.0, zorder=51)
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("Distance (km)", fontsize = label_size)
        ax.set_ylabel("Depth (km)", fontsize = label_size)
        
    else:
        ax.grid('-k', alpha=0.7)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("Distance (km)", fontsize = label_size)
        ax.set_ylabel("Topography (km)", fontsize = label_size)
        
    if (save_frames == True):
        fig_name = f"{output_path}/{model_name}_{prop}"

        if(plot_melt==True):
                # fig_name = f"{output_path}/{model_name}_{prop}_MeltFrac_{melt_method}_{str(int(dataset.step)).zfill(6)}.png"
                fig_name = f"{fig_name}_MeltFrac_{melt_method}"
        
        if(plot_particles == True):
                fig_name = f"{fig_name}_particles"
        
        fig_name = f"{fig_name}_{str(int(dataset.step)).zfill(6)}.png"


        plt.savefig(fig_name, dpi=400)
