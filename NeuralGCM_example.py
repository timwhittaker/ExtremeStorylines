import jax
import jax.numpy as jnp
from jax import lax, tree_util as jtu
from functools import partial
import gcsfs
import jax
import numpy as np
import pickle
import xarray
from jax import grad, jit, checkpoint, value_and_grad, block_until_ready
import jax.numpy as jnp
import optax
from tqdm import tqdm
import sys

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

import matplotlib.pyplot as plt
import os
gcs = gcsfs.GCSFileSystem(token='anon')

sliced_era5 = xarray.open_zarr("hw_init.zarr", chunks=None)

model_name = 'v1/deterministic_2_8_deg.pkl'
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)
model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=sliced_era5.sizes['latitude'],
    longitude_nodes=sliced_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(sliced_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(sliced_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

inner_steps = 24  # save model outputs once every 24 hours
outer_steps = 1 * 24 // inner_steps  # total of 4 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

# initialize model state
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
rng_key = jax.random.key(42)  # optional for deterministic models
initial_state = model.encode(inputs, input_forcings, rng_key)

# use persistence for forcing variables (SST and sea ice cover)
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

# Extract and reconstruct differentiable parts of the state
def extract_non_diff(state):
    """Extract non-differentiable components (prng_key, prng_step, etc.)"""
    randomness = state.randomness
    inner_state = state.state
    memory = state.memory
    divergence = inner_state.divergence
    #vorticity = inner_state.vorticity
    temperature_variation = inner_state.temperature_variation
    tracers = inner_state.tracers
    sim_time = inner_state.sim_time
    #new_inner_state = inner_state.replace(temperature_variation=None, divergence=None, vorticity=None, tracers=None, sim_time=None)
    new_inner_state = inner_state.replace(temperature_variation=None, divergence=None, tracers=None, sim_time=None)
    # Remove non-differentiable components
    new_state = state.replace(state=new_inner_state, randomness=None, memory=None)

    #return new_state, (randomness, temperature_variation, divergence, vorticity, tracers, sim_time, memory)
    return new_state, (randomness, temperature_variation, divergence, tracers, sim_time, memory)


def reconstruct_full_state(state_without_non_diff, non_diff_components):
    """Reconstruct the full state by adding back non-differentiable components."""
    #randomness, temp_var, div, vort, tracers, sim_time, memory = non_diff_components
    randomness, temp_var, div, tracers, sim_time, memory = non_diff_components
    temp_var = jax.lax.stop_gradient(temp_var)
    div = jax.lax.stop_gradient(div)
    #vort = jax.lax.stop_gradient(vort)
    tracers = jax.lax.stop_gradient(tracers)
    sim_time = jax.lax.stop_gradient(sim_time)
    inner_state = state_without_non_diff.state
    #new_inner_state = inner_state.replace(temperature_variation=temp_var, divergence=div, vorticity=vort, tracers=tracers, sim_time=sim_time)
    new_inner_state = inner_state.replace(temperature_variation=temp_var, divergence=div, tracers=tracers, sim_time=sim_time)
    return state_without_non_diff.replace(randomness=randomness, state=new_inner_state, memory=memory)

lytton_lat = 50.231111
lytton_lon = 121.581389
# Convert Lytton longitude to 0-360 range
lytton_lon_positive = (360 - lytton_lon) % 360
# Find the closest latitude
closest_lat_index = np.abs(eval_era5.latitude.values - lytton_lat).argmin() + 1
# Find the closest longitude
closest_lon_index = np.abs(eval_era5.longitude.values - lytton_lon_positive).argmin() + 1
print(closest_lat_index, closest_lon_index)
# Configuration
TARGET_TEMP = 315.0
outer_steps = 14*24  # Total trajectory steps per iteration



@partial(jax.jit)
def compute_loss(diff_state, non_diff_components):
    """Checkpoint-friendly loss computation using your state structure"""
    # Reconstruct full state using your method
    full_state = reconstruct_full_state(diff_state, non_diff_components)
    
    # Memory-optimized unroll with checkpointing
    def scan_fn(carry, _):
        state = carry
        new_state, preds = model.unroll(
            state, all_forcings, steps=1, start_with_input=True
        )
        return new_state, preds['temperature']
    
    # Checkpointed trajectory computation
    _, temp_traj = lax.scan(
        jax.checkpoint(scan_fn),
        full_state,
        None,
        length=outer_steps
    )
    
    # loss calculation
    final_temp = jnp.mean(temp_traj[-4*24,0,-1,closest_lon_index:closest_lon_index+2, closest_lat_index:closest_lat_index+2])
    return jnp.mean(TARGET_TEMP - final_temp) ** 2 

@partial(jax.jit)
def update_step(diff_state, opt_state, non_diff_components):
    """Single optimization step preserving non-diff components"""
    # Gradient calculation using your state structure
    loss, grads = jax.value_and_grad(compute_loss)(diff_state, non_diff_components)
    
    # Apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    diff_state = optax.apply_updates(diff_state, updates)
    
    return diff_state, opt_state, loss

# Initialization using your exact state structure
initial_diff_state, initial_non_diff = extract_non_diff(initial_state)
optimizer = optax.adam(learning_rate=1e-5)
opt_state = optimizer.init(initial_diff_state)

# Training loop
current_diff = initial_diff_state
current_non_diff = initial_non_diff

pbar = tqdm(range(35), desc="Optimizing")  # Create a tqdm instance
for step in pbar:
    current_diff, opt_state, loss = update_step(
        current_diff,
        opt_state,
        current_non_diff
    )
    
    # Maintain non-diff components across steps
    full_state = reconstruct_full_state(current_diff, current_non_diff)
    _, current_non_diff = extract_non_diff(full_state)

    pbar.set_description(f"Loss: {loss:.4f}")  # Update description correctly

print("Training complete.")

times = np.arange(outer_steps)
optimized_state = reconstruct_full_state(current_diff, current_non_diff)
final_state, predictions = model.unroll(
    optimized_state,
    all_forcings,
    steps=outer_steps,
    #timedelta=timedelta,
    start_with_input=True,
)
predictions_ds = model.data_to_xarray(predictions, times=times)
predictions_ds.to_netcdf("optimized.nc")

final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    #timedelta=timedelta,
    start_with_input=True,
)
predictions_ds = model.data_to_xarray(predictions, times=times)
predictions_ds.to_netcdf("original.nc")
print("Trajectories saved!")
