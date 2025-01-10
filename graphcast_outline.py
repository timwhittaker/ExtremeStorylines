import dataclasses
import functools
from typing import Optional

import jax
import haiku as hk
import numpy as np
import xarray
import time

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_tree
from graphcast import xarray_jax


def load_model_from_checkpoint(checkpoint_path):
    """Load model checkpoint from a local file."""
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    return params, state, model_config, task_config

def load_data(data_path, task_config):
    """Load and preprocess weather data from a local file."""
    example_batch = xarray.load_dataset(data_path).compute()

    # Extract inputs, targets, and forcings
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", "12h"),  # Example lead time
        **dataclasses.asdict(task_config)
    )

    return inputs, targets, forcings

# 3. Load Normalization Stats 
def load_normalization_stats(stats_paths):
    """Load normalization stats from local files."""
    diffs_stddev_by_level = xarray.load_dataset(stats_paths['diffs_stddev']).compute()
    mean_by_level = xarray.load_dataset(stats_paths['mean']).compute()
    stddev_by_level = xarray.load_dataset(stats_paths['stddev']).compute()

    stats = {
        'diffs_stddev': diffs_stddev_by_level,
        'mean': mean_by_level,
        'stddev': stddev_by_level
    }
    return stats

# 4. Model Construction
def construct_wrapped_graphcast(model_config, task_config, stats):
    """Construct and wrap the GraphCast predictor."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats['diffs_stddev'],
        mean_by_level=stats['mean'],
        stddev_by_level=stats['stddev']
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

# 5. Define Model Functions 
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, stats)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, stats)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

# Modified grads_fn to compute gradients w.r.t inputs
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(inputs_): # Note the input is now the first argument
      (loss, diagnostics), next_state = loss_fn.apply(
          params, state, jax.random.PRNGKey(0), model_config, task_config,
          inputs_, targets, forcings)
      return loss, (diagnostics, next_state)
    
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(inputs) # Note: gradient w.r.t the inputs
    return loss, diagnostics, next_state, grads


# --- Main Execution ---
if __name__ == "__main__":
    # Paths configuration (Simplified)
    config = {
        'checkpoint_path': 'params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz',
        'stats_paths': {
            'diffs_stddev': 'stats/stats_diffs_stddev_by_level.nc',
            'mean': 'stats/stats_mean_by_level.nc',
            'stddev': 'stats/stats_stddev_by_level.nc'
        },
        'data_path': 'data/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc'
    }
    
    # Load model
    params, state, model_config, task_config = load_model_from_checkpoint(config['checkpoint_path'])
    print("Using Checkpoint Model")


    # Load data
    inputs, targets, forcings = load_data(config['data_path'], task_config)
    
    # Load stats
    stats = load_normalization_stats(config['stats_paths'])

    #Jit Functions (Similar to notebook)
    def with_configs(fn):
        return functools.partial(
            fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]
    
    init_jitted = jax.jit(with_configs(run_forward.init))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings)

    loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
    grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))


    # Make predictions (using the notebook's rollout)
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings
    )

    
    # Compute loss
    loss, diagnostics = loss_fn_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets=targets,
        forcings=forcings)
    print("Loss:", float(loss))
    
    # Compute gradients (using the notebook's gradients function)
    loss, diagnostics, next_state, grads = grads_fn_jitted(
        inputs=inputs,
        targets=targets,
        forcings=forcings)

    mean_grad = jax.tree_util.tree_map(
        lambda x: np.abs(xarray_jax.unwrap_data(x) if hasattr(x, 'data') else x).mean(), grads)    
    mean_grad_value = np.mean(jax.tree_util.tree_flatten(mean_grad)[0])
    print(f"Loss: {loss:.4f}, Mean |grad| of inputs: {mean_grad_value:.6f}")
    print("Shape of input gradients:", jax.tree_util.tree_flatten(grads)[0][0].shape)
