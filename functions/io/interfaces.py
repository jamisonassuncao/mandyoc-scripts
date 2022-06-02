"""
Save interfaces and layers data to ASCII files ready to be read by MANDYOC
"""
import numpy as np
import os

FLAGS = {
    "compositional_factor": "C",
    "density": "rho",
    "radiogenic_heat": "H",
    "pre-exponential_scale_factor": "A",
    "power_law_exponent": "n",
    "activation_energy": "Q",
    "activation_volume": "V",
}
FNAME = "interfaces.txt"


def save_interfaces(interfaces, layers_parameters, path, fname=FNAME):
    """
    Save the interface and the layer parameter as ASCII file

    Parameters
    ----------
    interfaces : :class:`xarray.Dataset`
        Dataset with the interfaces depth.
    layers_parameters : dict
        Dictionary with the parameters values for each layer.
        The needed parameters are:
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
    # Check if all the needed parameters are in layer_parameters
    _check_all_parameters(layers_parameters)
    # Check that the length of each parameter in layers_parameters are the same
    _check_length_parameters(layers_parameters)
    # Check that the length of the parameters is equal to the length of
    # interfaces plus one
    _check_length_interfaces(interfaces, layers_parameters)
    # Check if the interfaces are in the correct order inside the xr.Dataset
    # Deeper ones must be before the shallower ones.
    _check_order_of_interfaces(interfaces)
    # Generate the header with the layers parameters
    header = []
    for parameter in layers_parameters:
        header.append(
            FLAGS[parameter]
            + " "
            + " ".join(list(str(i) for i in layers_parameters[parameter]))
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


def _check_order_of_interfaces(interfaces):
    """
    Check if the interfaces are ordered from deep to shallow
    """
    inames = tuple(i for i in interfaces)
    for i in range(len(inames) - 1):
        if not (interfaces[inames[i + 1]] >= interfaces[inames[i]]).values.all():
            raise ValueError(
                "Wrong order of interfaces. "
                + "Interface {} must be located after {} on the xr.Dataset. ".format(
                    inames[i], inames[i + 1]
                )
                + "Deeper interfaces must be located before shallower ones."
            )


def _check_all_parameters(layers_parameters):
    """
    Check if all the needed parameters are in layers_parameters
    """
    for parameter in FLAGS:
        if parameter not in layers_parameters:
            raise ValueError(
                "Parameter '{}' not present in layer_parameters. ".format(parameter)
                + "All the following parameters must be included in layer_properties:"
                + "\n    "
                + "\n    ".join([str(i) for i in FLAGS.keys()])
            )


def _check_length_parameters(layers_parameters):
    """
    Check that the length of each parameter in layers_parameters are the same
    """
    sizes = list(len(i) for i in list(layers_parameters.values()))
    if not np.allclose(sizes[0], sizes):
        raise ValueError(
            "Invalid number of elements layer parameters. "
            + "All layer parameters must have the same number of elements."
        )


def _check_length_interfaces(interfaces, layers_parameters):
    """
    Check that the length of the parameters is equal to the length of interfaces
    plus one
    """
    size = len(list(layers_parameters.values())[0])
    if not np.allclose(size, len(interfaces) + 1):
        raise ValueError(
            "Invalid number of layers ({}) for given interfaces ({}). ".format(
                size, len(interfaces)
            )
            + "The number of layers must be one more than the number of interfaces."
        )
