# Simple scripts for Mandyoc

This repository contains `.ipynb` scripts to be used together with the Mandyoc code (Sacek et al., 2022). 

The [`plot-output.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/plot-output.ipynb) script both plots and creates a movie of any Mandyoc simulated scenario.

The [`slab2-temperature.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/slab2-temperature.ipynb) script (under development) creates a temperature field for the Slab2 subduction zone geometry model (Hayes et al., 2018).

The [`subduction-initial.ipynb`](https://github.com/jamisonassuncao/mandyoc-scripts/blob/master/subduction-initial.ipynb) script creates an initial (and simple) subduction scenario for Mandyoc to simulate. The subduction scenario is based on the subduction simulation performed by Strak and Schelart (2021).

The scripts were built with [JupyterLab](https://jupyter.org/) and its environment `mpy` can be replicated with [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by using
```
conda env create --file=environment.yml
```

Once installed, the environment can be accessed by activating it with
```
conda activate mpy
```

The code is built on top of Agustina Pesces' codes.

