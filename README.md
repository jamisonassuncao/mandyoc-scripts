# Simple scripts for Mandyoc #

This repository contains `.ipynb` scripts to be used together with the Mandyoc code ([`Sacek et al., 2022`](https://joss.theoj.org/papers/10.21105/joss.04070)). 

## .ipynb notebooks ##

* The [`plot-output.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/plot-output.ipynb) script both plots and creates a movie of any Mandyoc simulated scenario.

* The [`frames generator.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/frames_generator.ipynb) script both plot and creates a movie of the standard properties printed by Mandyoc and other other post-processing information like temperature anomaly. There is also options to plot additional information like isotherms or melt fraction.

* The [`plot_properties.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/plot_properties.ipynb) script plot in a single figure frames according to a given list of instants the standard properties printed by Mandyoc and other other post-processing information like temperature anomaly. There is also options to plot additional information like isotherms or melt fraction.

* The [`isotherm_evolution.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/isotherm_evolution.ipynb) script extract the mean depth of a given list of isotherms and plot the evolution of the mean depth over time.

* The [`calc_melt_volume.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/calc_melt_volume.ipynb) script calculates the evolution of melt fraction and melt volume over time and saves as a Xarray dataset. The estimate of melt fraction follow the models for a dry and wet mantle presented by Gerya (2019).

* The [`plot_melt_volume.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/calc_melt_volume.ipynb) script plot the evolution of melt volume calculated by [`calc_melt_volume.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/calc_melt_volume.ipynb) over time. As a bonus, this script also estimate and plot the rate of melt production and the respective thickness of the melt production.

* The [`slab2-temperature.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/slab2-temperature.ipynb) script (under development) creates a temperature field for the Slab2 subduction zone geometry model (Hayes et al., 2018).

* The [`subduction-initial.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/subduction-initial.ipynb) script creates an initial (and simple) subduction scenario for Mandyoc to simulate. The subduction scenario is based on the subduction simulation performed by Strak and Schelart (2021).

# Conda environment #

The scripts were built with [JupyterLab](https://jupyter.org/) and its environment `mpy` can be replicated with [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by using
```
conda env create --file=environment.yml
```

Once installed, the environment can be accessed by activating it with
```
conda activate mpy
```

The code is built on top of [`Agustina Pesces`](https://github.com/aguspesce)' codes.

