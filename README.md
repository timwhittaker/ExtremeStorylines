# ExtremeStorylines

Sample code for the (...) paper. 

## Overview
This project optimizes a initial conditions to achieve a target temperature at a specific location (Lytton, BC). The model is initialized with ERA5 climate data and uses gradient-based optimization with `optax` and JAX. Some other small examples are included such as Lorenz96 and the Dinosaur dynamical core.

## Prerequisites
Before running the script, ensure you have the following dependencies installed:

- Python 3.x
- `jax`
- `jax.numpy`
- `optax`
- `xarray`
- `gcsfs`
- `yaml`
- `tqdm`
- `matplotlib`
- `neuralgcm`

## Configuration
Modify the `config.yaml` file to specify parameters such as:
```yaml
optimizer:
  name: "adam"
  learning_rate: 1e-9 # learning rate
  iteration_number: 50 # epochs
loss:
  lambda: 20 # overall weighing factor for the i.c. violation
  beta: 10 # strength of target term
evol_days: 11 # total number of evolution days
lytton_lat: 50.231111 # target coord
lytton_lon: 121.581389 # target coord
init_cond: "path_to_initial_conditions" # path to i.c. zarr file
output_dir: "path_to_output" # path to output folder
```

Initial conditions we use can be found [here](https://uqam-my.sharepoint.com/:u:/r/personal/cf891976_ens_uqam_ca/Documents/ExtremeStoryline/initial_cond.zip?csf=1&web=1&e=cxiunK)
