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
loss_threshold: 1e-3
output:
  losses_file: "losses.npy"
  final_optimized_state: "final_optimized.nc"
  initial_state_output: ""
