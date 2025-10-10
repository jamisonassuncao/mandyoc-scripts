import os
import gc
import sys
import pymp
import shutil
import psutil
import subprocess
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

path = os.getcwd().split('/')
machine_path = '/'+path[1]+'/'+path[2] #cat the /home/user/ or /Users/user from system using path

sys.path.insert(0, f"{machine_path}/opt/mandyoc-scripts/functions")
from mandyocIO import read_mandyoc_output, read_datasets, read_particle_path, change_dataset, single_plot

path = os.getcwd().split('/') # Get local file
machine_path = '/'+path[1]+'/'+path[2] # Select home according to OS.

scenario = '/Doutorado/cenarios/mandyoc/keel/Lx4000km/cold_keel/thermal_bc_fixed/long_craton/mobile_belt/mb_dryol/keel_stable_DT200_HprodAst_Lx4000km_ck_lc_shallow_mbdo_Cmb01'

local = False
if(local==True):
    model_path = machine_path + scenario
else:
    # scenario = '/RFT_Clc1_DT200_PT1292oC_lit80km_1x1km2_NHK'
    external_media = 'Joao_Macedo'
    model_path = f"/media/{machine_path.split('/')[-1]}/{external_media}{scenario}"

read_ascii = True # if False, it will skip the ascii reading and saving processes and it will read the datasets from previous runs
save_images = True # Generate and save all the possible images
save_big_dataset = False#True
plot_isotherms = True
# plot_melt = True
plot_melt = False
melt_method = 'dry'
# melt_method = 'wet'

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

model_name = scenario.split('/')[-1]

# Create the output directory to save the dataset
output_path = os.path.join(model_path, "_output")
# print(output_path)

if not os.path.isdir(output_path):
    os.makedirs(output_path)

# model_name = os.path.split(model_path)[1]
# model_name = os.path.split(model_path)[0].split('/')[-1]
# model_name = scenario.split('/')[-1]

datasets = [#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            'surface',
            'temperature',
            'viscosity'
            ]# Read data and convert them to xarray.Dataset

properties = [#Properties from mandyoc. Comment/uncomment to select which ones you would like to plot
#              'density',
#              'radiogenic_heat',
             'lithology',
#              'pressure',
            #  'strain',
             'strain_rate',
             'temperature',
             'temperature_anomaly',
             'surface',
             'viscosity'
             ]

# Read ascii outputs and save them as xarray.Datasets

new_datasets = change_dataset(properties, datasets)
to_remove = []

remove_density=False
if ('density' not in properties): #used to plot air/curst interface
        properties.append('density')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('density')
        # remove_density=True

if (plot_isotherms): #add datasets needed to plot isotherms
    if ('temperature' not in properties):
        properties.append('temperature')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('temperature')
        
if (plot_melt): #add datasets needed to plot melt fraction
    if ('pressure' not in properties):
        properties.append('pressure')
    if ('temperature' not in properties):
        properties.append('temperature')
    new_datasets = change_dataset(properties, datasets)

    #removing the auxiliary datasets to not plot
    to_remove.append('pressure')
    to_remove.append('temperature')

if(clean_plot): #a clean plot
    new_datasets = change_dataset(properties, datasets)

for dataset in datasets:
    if not os.path.isfile(f"{model_path}/_output_{dataset}.nc"):
        print(f"Could not find dataset {dataset}. Creating missing dataset.")
        ds_data = read_mandyoc_output(
            model_path,
            datasets=dataset,
            parameters_file="param.txt"
        )     
for item in to_remove:
    properties.remove(item)

if (save_big_dataset):
    dataset = read_datasets(model_path, new_datasets, save_big_dataset = True) 
else:
    dataset = read_datasets(model_path, new_datasets)
# Normalize velocity values
if ("velocity_x" and "velocity_z") in dataset.data_vars:
    v_max = np.max((dataset.velocity_x**2 + dataset.velocity_z**2)**(0.5))    
    dataset.velocity_x[:] = dataset.velocity_x[:] / v_max
    dataset.velocity_z[:] = dataset.velocity_z[:] / v_max
# print(dataset.info)

# plot_particles = True 
plot_particles = False
unzip_steps = False

if(plot_particles):
    unzip_steps = True
    if(unzip_steps):
        comand = f"unzip -o {model_path}/{model_name}.zip step*.txt -d {model_path}"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)

properties = [#Properties from mandyoc. Comment/uncomment to select which ones you would like to plot
#              'density',
#              'radiogenic_heat',
             'lithology',
#              'pressure',
            #  'strain',
            #  'strain_rate',
            #  'temperature',
             'temperature_anomaly',
             'surface',
             'viscosity'
             ]

t0 = dataset.time[0]
t1 = dataset.time[1]
dt = int(t1 - t0)

start = int(t0)
end = int(dataset.time.size - 1)
step = 1#5

# start = 0
# end = 5
# step = 1

# step_initial = dataset.step[0]
# step_1 = dataset.step[1]
# step_final = dataset.step[-1]
# dstep = int(step_1 - step_initial)

with pymp.Parallel(20) as p:
    for i in p.range(start, end+step, step):
        # data = dataset.isel(time=i)
        per = np.round(100*(i+1-start)/(end-start), 2)
        text = f"Time: {np.round(float(dataset.isel(time=i).time), 2)} Myr; Step: {int(dataset.isel(time=i).step)}/{int(dataset.step.max())}, ({per:.2f}%)."
        
        # print(text, end='\r')
        data = dataset.isel(time=i)
        for prop in properties:
    #         print(f"Handeling {prop}.", end='\n')
            if(prop != 'surface'): # you can customize
                xlims = [0, float(dataset.isel(time=i).lx) / 1.0e3]
                ylims = [-float(dataset.isel(time=i).lz) / 1.0e3 + 40, 40]
            else:
                xmin = 0 #+ 200
                xmax = float(dataset.isel(time=i).lx) / 1.0E3 #- 200
                xlims = [xmin, xmax]
                ylims = [-2, 2]

            memory_info = psutil.virtual_memory()
            total_memory = memory_info.total
            used_memory = memory_info.used
            free_memory = memory_info.available

            
            print(f"Total Memory: {total_memory/(1024*1024)} Mbytes")
            print(f"Used Memory: {used_memory/(1024*1024)} Mbytes")
            print(f"Free Memory: {free_memory/(1024*1024)} Mbytes")
            print(80*"=") 
            single_plot(data, prop, xlims, ylims, model_path, output_path,
                        plot_isotherms = plot_isotherms,
                        plot_particles = plot_particles,
                        particle_size = 0.02,
                        # marker = ".",
                        ncores = 20,
                        # step_plot = 3,
                        isotherms = [500, 1300],
                        plot_melt = plot_melt,
                        melt_method = melt_method)
            
            memory_info = psutil.virtual_memory()
            total_memory = memory_info.total
            used_memory = memory_info.used
            free_memory = memory_info.available

            print(f"Total Memory after: {total_memory/(1024*1024)} Mbytes")
            print(f"Used Memory after: {used_memory/(1024*1024)} Mbytes")
            print(f"Free Memory after: {free_memory/(1024*1024)} Mbytes")
            print(80*"=")