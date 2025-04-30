import jax
import jax.numpy as jnp
from jax import lax, tree_util as jtu, grad, jit, checkpoint, value_and_grad, block_until_ready
import optax
import numpy as np
import pickle
import xarray
from functools import partial
import gcsfs
from tqdm import tqdm
import sys
import os
import yaml
import matplotlib.pyplot as plt
import argparse  

from dinosaur import (horizontal_interpolation, 
                     spherical_harmonic, 
                     xarray_utils)
import neuralgcm


gcs = gcsfs.GCSFileSystem(token='anon')

def create_output_directory(config):
    """Create parameterized output directory structure"""
    base_dir = config['output_dir']
    
    # Extract parameters
    opt = config['optimizer']
    loss = config['loss']
    evol_days = config['evol_days']
    lat = config['lytton_lat']
    lon = config['lytton_lon']
    init_cond = config['init_cond'].split('_')[-1].split('.')[0]

    # Format directory components
    components = [
        f"lr{float(opt['learning_rate']):.0e}",
        f"it{opt['iteration_number']}",
        f"lam{loss['lambda']}",
        f"b{loss['beta']}",
        f"d{evol_days}",
        f"init{init_cond}"
    ]

    # Clean special characters
    clean_components = [c.replace('.', 'p').replace('-', 'm') for c in components]
    
    # Create full path
    dir_name = "_".join(clean_components)
    full_path = os.path.join(base_dir, dir_name)
    
    # Create directory and return path
    os.makedirs(full_path, exist_ok=True)
    return full_path

def main(config_path):

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Load model
    model_name = 'v1_precip/stochastic_precip_2_8_deg.pkl'#'v1/deterministic_2_8_deg.pkl'
    with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
        ckpt = pickle.load(f)
    
    # Make sure we output surface pressure
    new_inputs_to_units_mapping = {
        'u': 'meter / second',
        'v': 'meter / second',
        't': 'kelvin',
        'z': 'm**2 s**-2',
        'sim_time': 'dimensionless',
        'tracers': {
            'specific_humidity': 'dimensionless',
            'specific_cloud_liquid_water_content': 'dimensionless',
            'specific_cloud_ice_water_content': 'dimensionless',
        },
        'diagnostics': {'surface_pressure': 'kg / (meter s**2)'},
    }

    new_model_config_str = '\n'.join([
        ckpt['model_config_str'],
        (
            'DimensionalLearnedPrimitiveToWeatherbenchDecoder.inputs_to_units_mapping'
            f' = {new_inputs_to_units_mapping}'
        ),
        (
            'DimensionalLearnedPrimitiveToWeatherbenchDecoder.diagnostics_module ='
            ' @NodalModelDiagnosticsDecoder'
        ),
        (
            'StochasticPhysicsParameterizationStep.diagnostics_module ='
            ' @SurfacePressureDiagnostics'
        ),
    ])
    ckpt['model_config_str'] = new_model_config_str

    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    
    output_dir = create_output_directory(config)

    # Load initial data
    sliced_era5 = xarray.open_zarr(config['init_cond'], chunks=None)
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

    # initialize model state
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
    rng_key = jax.random.key(42)  #
    initial_state = model.encode(inputs, input_forcings, rng_key)

    # use persistence for forcing variables (SST and sea ice cover)
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

    # Extract and reconstruct differentiable parts of the state
    def extract_non_diff(state):
        randomness = state.randomness
        inner_state = state.state
        components = (
            randomness,
            inner_state.sim_time,
            state.memory
        )
        new_inner_state = inner_state.replace(
            sim_time=None
        )
        return state.replace(state=new_inner_state, randomness=None, memory=None), components


    def reconstruct_full_state(state_without_non_diff, non_diff_components):
        """Reconstruct the full state by adding back non-differentiable components."""
        randomness, sim_time, memory = non_diff_components
        sim_time = jax.lax.stop_gradient(sim_time)
        inner_state = state_without_non_diff.state
        new_inner_state = inner_state.replace(sim_time=sim_time)
        return state_without_non_diff.replace(randomness=randomness, state=new_inner_state, memory=memory)

    # Target domain
    lytton_lat = float(config['lytton_lat'])
    lytton_lon = float(config['lytton_lon'])
    lytton_lon_positive = (360 - lytton_lon) % 360

    def find_closest_index(coords, target):
        return np.abs(coords - target).argmin()

    closest_lat_index = find_closest_index(eval_era5.latitude.values, lytton_lat)
    closest_lon_index = find_closest_index(eval_era5.longitude.values, lytton_lon_positive)

    outer_steps = float(config['evol_days'])*24  # Total trajectory steps per iteration

    @partial(jax.jit)
    def compute_loss(diff_state, non_diff_components, initial_diff_state):
        """Checkpoint loss computation using your state structure"""
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
        beta = float(config['loss']['beta'])
        lam0 = float(config['loss']['lambda'])


        T_ref = 298.15 
        p_ref = jnp.mean(initial_diff_state.state.log_surface_pressure)**2
        v_ref = jnp.mean(initial_diff_state.state.vorticity)**2
        d_ref = jnp.mean(initial_diff_state.state.divergence)**2
        T0_ref = jnp.mean(initial_diff_state.state.temperature_variation)**2
        sh_ref = jnp.mean(initial_diff_state.state.tracers['specific_humidity'])**2
        sciwc_ref = jnp.mean(initial_diff_state.state.tracers['specific_cloud_ice_water_content'])**2
        sclwc_ref = jnp.mean(initial_diff_state.state.tracers['specific_cloud_liquid_water_content'])**2

        # TODO: Make these parameters in config !!
        reg_term_p = jnp.mean((diff_state.state.log_surface_pressure - initial_diff_state.state.log_surface_pressure) ** 2)
        reg_term_d = 100*jnp.mean((diff_state.state.divergence - initial_diff_state.state.divergence) ** 2)
        reg_term_v = 100*jnp.mean((diff_state.state.vorticity - initial_diff_state.state.vorticity) ** 2)
        reg_term_T = 100*jnp.mean((diff_state.state.temperature_variation - initial_diff_state.state.temperature_variation) ** 2)
        reg_term_sh = 10*jnp.mean((diff_state.state.tracers['specific_humidity'] - initial_diff_state.state.tracers['specific_humidity']) ** 2)
        reg_term_sciwc = jnp.mean((diff_state.state.tracers['specific_cloud_ice_water_content'] - initial_diff_state.state.tracers['specific_cloud_ice_water_content']) ** 2)
        reg_term_sclwc = jnp.mean((diff_state.state.tracers['specific_cloud_ice_water_content'] - initial_diff_state.state.tracers['specific_cloud_ice_water_content']) ** 2)
        
        # TODO: Should add number of days to opt as a param in config!!
        final_temp = jnp.mean(temp_traj[-5*24:,0,-1,closest_lon_index-2:closest_lon_index+2, closest_lat_index-2:closest_lat_index+2])
        
        lam = lam0 
        
        return ((beta*T_ref)/jnp.sqrt(jnp.mean(final_temp)))+ (lam) * (reg_term_p/p_ref+reg_term_d/d_ref+reg_term_v/v_ref+reg_term_T/T0_ref + reg_term_sh/sh_ref + reg_term_sciwc/sciwc_ref + reg_term_sclwc/sclwc_ref), final_temp 

    @partial(jax.jit)
    def update_step(diff_state, opt_state, non_diff_components, initial_diff_state):
        """Single optimization step preserving non-diff components"""
        (loss, temp), grads = jax.value_and_grad(compute_loss,has_aux=True)(diff_state, non_diff_components, initial_diff_state)
        
        # Apply updates
        updates, opt_state = optimizer.update(grads, opt_state)
        diff_state = optax.apply_updates(diff_state, updates)
        
        return diff_state, opt_state, loss, temp

    # Initialization
    initial_diff_state, initial_non_diff = extract_non_diff(initial_state)
    optimizer = optax.adam(learning_rate=float(config['optimizer']['learning_rate']))
    opt_state = optimizer.init(initial_diff_state)
    # Training loop
    current_diff = initial_diff_state
    current_non_diff = initial_non_diff

    times = np.arange(outer_steps)
    losses = []

    pbar = tqdm(range(config['optimizer']['iteration_number']), desc="Optimizing")  
    for step in pbar:
        current_diff, opt_state, loss, temp = update_step(
            current_diff,
            opt_state,
            current_non_diff,
            initial_diff_state
        )
        losses.append(loss)
        # Maintain non-diff components across steps
        full_state = reconstruct_full_state(current_diff, current_non_diff)
        _, current_non_diff = extract_non_diff(full_state)

        pbar.set_description(f"Loss: {loss:.4f}, Mean temp: {temp:.4f}")  

        # Save intermediate trajectories
        if step%100==0: # TODO: make this a config param
            optimized_state = reconstruct_full_state(current_diff, current_non_diff)
            final_state, predictions = model.unroll(
                optimized_state,
                all_forcings,
                steps=outer_steps,
                start_with_input=True,
                )

            predictions_ds = model.data_to_xarray(predictions, times=times)
            # Only save temps and geoptential
            predictions_ds = predictions_ds[['temperature', 'geopotential']]
            predictions_ds.to_netcdf(f"{output_dir}/optimized_step{step}.nc")
            # save the log pressure
            sp = model.from_nondim_units(jnp.squeeze(jnp.exp(optimized_state.state.log_surface_pressure), axis=0), 'kg / (meter s**2)')
            np.save(f"{output_dir}/log_surface_pressure_{step}",sp)

            del predictions_ds, optimized_state, final_state, predictions

    print("Training complete.")

    # Save losses
    losses = np.array(losses)
    np.save(f"{output_dir}/losses", losses)

    #Save trajectories
    optimized_state = reconstruct_full_state(current_diff, current_non_diff)
    final_state, predictions = model.unroll(
        optimized_state,
        all_forcings,
        steps=outer_steps,
        #start_with_input=True,
    )
    predictions_ds = model.data_to_xarray(predictions, times=times)
    predictions_ds = predictions_ds
    predictions_ds.to_netcdf(f"{output_dir}/optimized.nc")
    nodal_pressure = model.model_coords.horizontal.to_nodal(optimized_state.state.log_surface_pressure)
    nodal_vort = model.model_coords.horizontal.to_nodal(optimized_state.state.vorticity)
    nodal_div = model.model_coords.horizontal.to_nodal(optimized_state.state.divergence)
    vort = model.from_nondim_units(nodal_vort, '1/s')
    div = model.from_nondim_units(nodal_div, '1/s')
    sp = model.from_nondim_units(jnp.squeeze(jnp.exp(nodal_pressure), axis=0), 'kg / (meter s**2)')
    np.save(f"{output_dir}/log_surface_pressure_opt",sp)
    np.save(f"{output_dir}/vorticity_opt",vort)
    np.save(f"{output_dir}/divergence_opt",div)

    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        #start_with_input=True,
    )
    predictions_ds = model.data_to_xarray(predictions, times=times)
    predictions_ds = predictions_ds
    predictions_ds.to_netcdf(f"{output_dir}/original.nc")
    nodal_pressure = model.model_coords.horizontal.to_nodal(initial_state.state.log_surface_pressure)
    nodal_vort = model.model_coords.horizontal.to_nodal(initial_state.state.vorticity)
    nodal_div = model.model_coords.horizontal.to_nodal(initial_state.state.divergence)
    vort = model.from_nondim_units(nodal_vort, '1/s')
    div = model.from_nondim_units(nodal_div, '1/s')
    sp = model.from_nondim_units(jnp.squeeze(jnp.exp(nodal_pressure), axis=0), 'kg / (meter s**2)')
    np.save(f"{output_dir}/log_surface_pressure_original",sp)
    np.save(f"{output_dir}/vorticity_original",vort)
    np.save(f"{output_dir}/divergence_original",div)
    print("Trajectories saved!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    args = parser.parse_args()
    
    main(args.config)
