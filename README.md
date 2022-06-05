# Simple scripts for Mandyoc

This repository contains `.ipynb` scripts to be used together with the Mandyoc code. 

The `subduction-initial.ipynb` scriptcreates an initial subduction scenario for Mandyoc to simulate.

The `plot-output.ipynb` scriptplots and creates a movie of any Mandyoc simulated scenario.

The `slab2-temperature` script creates a temperature field for the Slab2 subduction zone geometry model (Hayes et al., 2018).

The scripts were built with Jupyter Lab and its environment can be replicated with `conda` by using
```
conda env create --file=environment.yml
```

Once installed, the environment can be acces by using
```
conda activate mpy
```

