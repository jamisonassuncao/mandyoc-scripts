import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_interfaces(interfaces, extent=[0, 1, -1, 0], meta_dir=None):
    """
    Makes 4 plots of interfaces and saves them.

    Parameters
    ----------
    interfaces : :class:`xarray.Dataset`
        Dataset containing the interfaces.
    extent : list
        List of 4 values containing x_min, x_max, z_min and z_max
    meta_dir : string
        Path to save the generated plot.
    """
    x_min, x_max, z_min, z_max = extent[0], extent[1], extent[2], extent[3]

    n_interfaces = len(interfaces.variables) - 1
    cmap = get_cmap('tab20b', n_interfaces)

    dx = 750
    fig, ax = plt.subplots(figsize=(25/1.5,15/1.5))
    # fig, ax = plt.subplots(figsize=(25,15))

    plt.subplot(2, 1, 1, aspect=1.)
    aux = 0
    for name, interface in interfaces.items():
        interface_new = interfaces[name]/1.0e3
        # plt.plot(interfaces.x/1.0e3, interfaces[name]/1.0e3, label=name, color=colors[aux])
        if (aux>0):
            plt.fill_between(interfaces.x/1.0e3, interface_old, interface_new, label=name[:-4], color=cmap(aux)[:3], linewidth=0)
        if (aux==0):
            plt.fill_between(interfaces.x/1.0e3, interface_new, z_min/1.0e3, label=name[:-4], color=cmap(aux)[:3], linewidth=0)
        plt.xlim((x_min/1.0e3, x_max/1.0e3))
        plt.ylim((z_min/1.0e3, z_max/1.0e3))
        plt.xlabel("Length [km]")
        plt.ylabel("Depth [km]")
        name_a = name
        interface_old = interfaces[name]/1.0e3
        plt.legend(loc=4)
        aux += 1
        
    plt.subplot(2, 5, 6)
    aux = 0
    for name, interface in interfaces.items():
        x0lim = x_min / 1.0e3
        x1lim = 500.0
        plt.plot(interfaces.x/1.0e3, interfaces[name]/1.0e3, label=name, color=cmap(aux)[:3])
        plt.xlim(x0lim, x1lim)
        plt.ylim((z_min/1.0e3, z_max/1.0e3))
        plt.xlabel("Length [km]")
        plt.ylabel("Depth [km]")
        aux += 1

    plt.subplot(2, 5, (7, 9), aspect=1)
    aux = 0
    for name, interface in interfaces.items():
        x0lim = (x_max - x_min) / 2.0 / 1.0e3 - 500.0
        x1lim = (x_max - x_min) / 2.0 / 1.0e3 + 500.0
        plt.plot(interfaces.x/1.0e3, interfaces[name]/1.0e3, label=name, color=cmap(aux)[:3])
        plt.xlim(x0lim, x1lim)
        plt.ylim((z_min/1.0e3, z_max/1.0e3))
        plt.ylim((-300, z_max/1.0e3))
        plt.xlabel("Length [km]")
        # plt.ylabel("Depth [km]")
        aux += 1
    plt.legend(loc=4)    
        
    plt.subplot(2, 5, 10)
    aux = 0
    for name, interface in interfaces.items():
        x0lim = 4000.0
        x1lim = x_max / 1.0e3
        plt.plot(interfaces.x/1.0e3, interfaces[name]/1.0e3, label=name, color=cmap(aux)[:3])
        plt.xlim(x0lim, x1lim)    
        plt.ylim((z_min/1.0e3, z_max/1.0e3))
        plt.xlabel("Length [km]")
        # plt.ylabel("Depth [km]")
        aux += 1


    # plt.savefig('out/initial-geometry.pdf')
    if (meta_dir != None):
        plt.savefig(f'{meta_dir}/initial-geometry.pdf')
    plt.show()

def plot_initial_temperature(temperature, lid, interfaces, h_air, extent=[0, 1, -1, 0], meta_dir=None):
    """
    Makes 5 temperature plots: one complete plot, three vertical profiles every 
    25% of the total horizontal length (excluding the borders), and one central
    profile with drawn interfaces.

    Parameters
    ----------
    temperature : :class:`xarray.Dataset`
        Dataset containing the temperature data
    lid : :class:`xarray.Dataset`
        Dataset containing the lithosphere-astenosphere interface boundary
    interfaces : :class:`xarray.Dataset`
        Dataset containg the differente interfaces
    h_air : float
        Depth of the air layer
    extent : list
        List of 4 values containing x_min, x_max, z_min and z_max
    meta_dir : string
        Path to save the generated plot
    """
    
    x_perc_1 = 0.25
    x_perc_2 = 0.50
    x_perc_3 = 0.75
    z_lim = -400
    T_min_lim, T_max_lim = -100, 1500.0

    x_min, x_max, z_min, z_max = extent[0], extent[1], extent[2], extent[3]
    extent = [x_min/1.0e3, x_max/1.0e3, z_min/1.0e3, z_max/1.0e3]

    cmap = get_cmap("coolwarm")

    plt.figure(figsize=(20,30))

    # complete teperature
    plt.subplot(6, 3, (1, 3))
    plt.title("Complete temperature field")
    x, y = np.meshgrid(temperature.coords["x"], temperature.coords["z"])
    # plt.imshow(temperature.T[:,::-1].T, extent=extent, aspect='equal')
    plt.imshow(temperature.T[::-1,:], extent=extent, aspect='equal', cmap=cmap)
    plt.vlines(x_perc_1 * x_max/1.0e3, z_max/1.0e3, z_min/1.0e3, linestyle="dashed", color="blue", label=str(x_perc_1*x_max/1.0e3)+" km")
    plt.vlines(x_perc_2 * x_max/1.0e3, z_max/1.0e3, z_min/1.0e3, linestyle="dashed", color="black", label=str(x_perc_2*x_max/1.0e3)+" km")
    plt.vlines(x_perc_3 * x_max/1.0e3, z_max/1.0e3, z_min/1.0e3, linestyle="dashed", color="red", label=str(x_perc_3*x_max/1.0e3)+" km")
    plt.legend(loc=3)

    # temperature gradient at (x_perc_1 * x_max)
    plt.subplot(6, 3, (4, 7))
    plt.title(f"Temperature gradiente at x={x_perc_1*x_max/1.0e3} km")
    y, x = temperature["z"]/1.0e3, temperature.sel(x=x_perc_1 * x_max, method="nearest")
    if (h_air < 0): plt.hlines(h_air/1.0e3, T_min_lim, T_max_lim, label="surface ({} km)".format(h_air/1.0e3), color="black", linestyle="dotted")
    z_lid = lid.sel(x=x_perc_1 * x_max, method="nearest")/1.0e3
    plt.hlines(z_lid, T_min_lim, T_max_lim, color="orange", linestyle="dotted", label="LAB depth ({} km)".format(z_lid.values))
    plt.vlines(1300, z_lim, extent[3], linestyles='dotted', color='green')
    plt.ylim(z_lim, extent[3])
    plt.plot(x, y)
    plt.xlim(T_min_lim, T_max_lim)
    plt.legend(loc=3)

    # temperature gradient at (x_perc_2 * x_max)
    plt.subplot(6, 3, (5, 8))
    plt.title(f"Temperature gradiente at x={x_perc_2*x_max/1.0e3} km")
    y, x = temperature["z"]/1.0e3, temperature.sel(x=x_perc_2 * x_max, method="nearest")
    if (h_air < 0): plt.hlines(h_air/1.0e3, T_min_lim, T_max_lim, label="surface ({} km)".format(h_air/1.0e3), color="black", linestyle="dotted")
    z_lid = lid.sel(x=x_perc_2 * x_max, method="nearest")/1.0e3
    plt.hlines(z_lid, T_min_lim, T_max_lim, color="orange", linestyle="dotted", label="LAB depth ({} km)".format(z_lid.values))
    z_lid = lid.sel(x=x_perc_2 * x_max, method="nearest").values
    plt.vlines(1300, z_lim, extent[3], linestyles='dotted', color='green')
    plt.ylim(z_lim, extent[3])
    plt.plot(x, y)
    plt.xlim(T_min_lim, T_max_lim)
    plt.legend(loc=3)

    # temperature gradient at (x_perc_3 * x_max)
    plt.subplot(6, 3, (6, 9))
    plt.title(f"Temperature gradiente at x={x_perc_3*x_max/1.0e3} km")
    y, x = temperature["z"]/1.0e3, temperature.sel(x=x_perc_3 * x_max, method="nearest")
    if (h_air < 0): plt.hlines(h_air/1.0e3, T_min_lim, T_max_lim, label="surface ({} km)".format(h_air/1.0e3), color="black", linestyle="dotted")
    z_lid = lid.sel(x=x_perc_3 * x_max, method="nearest")/1.0e3
    plt.hlines(z_lid, T_min_lim, T_max_lim, color="orange", linestyle="dotted", label="LAB depth ({} km)".format(z_lid.values))
    z_lid = lid.sel(x=x_perc_3 * x_max, method="nearest").values
    plt.vlines(1300, z_lim, extent[3], linestyles='dotted', color='green')
    plt.ylim(z_lim, extent[3])
    plt.plot(x, y)
    plt.xlim(T_min_lim, T_max_lim)
    plt.legend(loc=3)

    # zoomed teperature
    plt.subplot(6, 3, (10, 12))
    plt.title("Zoomed temperature field")
    x, y = np.meshgrid(temperature.coords["x"], temperature.coords["z"])
    # plt.imshow(temperature.T[:,::-1].T, extent=extent, aspect='equal')
    plt.imshow(temperature.T[::-1,:], extent=extent, aspect='equal', cmap=cmap)
    for name, interface in interfaces.items():
        plt.plot(interface["x"]/1.0e3, interface.values/1.0e3, color="white", linewidth=1.5)
    plt.xlim(2000, 4500)
    plt.ylim(z_lim, extent[3])
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label(r'Temperature, T [$^{\circ}$C]')

    if (meta_dir!=None):
        plt.savefig(f'{meta_dir}/initial-temperature.pdf')