import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray

from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import vertical_interpolation

from utils import attach_data_array_units
from utils import attach_xarray_units
from utils import xarray_nondimensionalize
from utils import xarray_to_gcm_dict
from utils import slice_levels

import optax
from functools import partial
from jax.experimental import mesh_utils
from jax import grad, jit, checkpoint, value_and_grad

import sys

units = scales.units

# Create a device mesh
#devices = mesh_utils.create_device_mesh((2, 1, 1))  # (z, x, y) dimensions
#mesh = jax.sharding.Mesh(devices, ['z', 'x', 'y'])  # Define logical axis names

# Define the coordinate system with SPMD mesh
# simulation grid
layers = 8
ref_temp_si = 250 * units.degK
max_wavenumber=21
model_coords = coordinate_systems.CoordinateSystem(
    horizontal=spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=max_wavenumber + 1,
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonicsWithZeroImag,
    ),
    vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers),
    #spmd_mesh=mesh,
)

# Example: Shard model state
#from jax.sharding import NamedSharding, PartitionSpec
#sharding = NamedSharding(mesh, PartitionSpec('z', 'x', 'y'))


# timescales
dt_si = 5 * units.minute
save_every = 30 * units.minute
total_time = 10 * units.day + save_every
dfi_timescale = 6 * units.hour

# which levels to output
output_level_indices = [layers // 4, layers // 2, 3*layers // 4, -1]

ds_arco_era5 = xarray.open_zarr("ds_arco", chunks=None)

print(ds_arco_era5.time.values)

ds = ds_arco_era5[[
    'u_component_of_wind',
    'v_component_of_wind',
    'temperature',
    'specific_humidity',
    'specific_cloud_liquid_water_content',
    'specific_cloud_ice_water_content',
    'surface_pressure',
]]

raw_orography = ds_arco_era5.geopotential_at_surface

desired_lon = 180/np.pi * model_coords.horizontal.nodal_axes[0]
desired_lat = 180/np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])

ds_init = attach_xarray_units(ds.compute().interp(latitude=desired_lat, longitude=desired_lon))
ds_init['orography'] = attach_data_array_units(raw_orography.interp(latitude=desired_lat, longitude=desired_lon))
ds_init['orography'] /= scales.GRAVITY_ACCELERATION

source_vertical = vertical_interpolation.HybridCoordinates.ECMWF137()

# nondimensionalize
ds_nondim_init = xarray_nondimensionalize(ds_init)
model_level_inputs = xarray_to_gcm_dict(ds_nondim_init)

sp_nodal = model_level_inputs.pop('surface_pressure')
orography_input = model_level_inputs.pop('orography')

sp_init_hpa = ds_init.surface_pressure.transpose('longitude', 'latitude').data.to('hPa').magnitude

# build inputs
physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
print(physics_specs)

nodal_inputs = vertical_interpolation.regrid_hybrid_to_sigma(
    fields=model_level_inputs,
    hybrid_coords=source_vertical,
    sigma_coords=model_coords.vertical,
    surface_pressure=sp_init_hpa,
)
u_nodal = nodal_inputs['u_component_of_wind']
v_nodal = nodal_inputs['v_component_of_wind']
t_nodal = nodal_inputs['temperature']

# calculate vorticity & divergence
vorticity, divergence = spherical_harmonic.uv_nodal_to_vor_div_modal(
    model_coords.horizontal, u_nodal, v_nodal
)

# apply reference temperature
ref_temps = physics_specs.nondimensionalize(
    ref_temp_si * np.ones((model_coords.vertical.layers,))
)

assert ref_temps.shape == (model_coords.vertical.layers,)
temperature_variation = model_coords.horizontal.to_modal(
    t_nodal - ref_temps.reshape(-1, 1, 1)
)

log_sp = model_coords.horizontal.to_modal(np.log(sp_nodal))
tracers = model_coords.horizontal.to_modal(
    {
        'specific_humidity': nodal_inputs['specific_humidity'],
        'specific_cloud_liquid_water_content': nodal_inputs['specific_cloud_liquid_water_content'],
        'specific_cloud_ice_water_content': nodal_inputs['specific_cloud_ice_water_content'],
    }
)

# build initial state
raw_init_state = primitive_equations.State(
    vorticity=vorticity,
    divergence=divergence,
    temperature_variation=temperature_variation,
    log_surface_pressure=log_sp,
    tracers=tracers,
)

orography = model_coords.horizontal.to_modal(orography_input)
orography = filtering.exponential_filter(model_coords.horizontal, order=2)(orography)

# setup a simulation of the dry primitive equations
eq = primitive_equations.PrimitiveEquations(
    ref_temps, orography, model_coords, physics_specs
)

# setup hyper-spectral filter for running between dycore time-steps
res_factor = model_coords.horizontal.latitude_nodes / 64
dt = physics_specs.nondimensionalize(dt_si)
tau = physics_specs.nondimensionalize(8.6 / (2.4 ** np.log2(res_factor)) * units.hours)
hyperdiffusion_filter = time_integration.horizontal_diffusion_step_filter(
    model_coords.horizontal, dt=dt, tau=tau, order=2
)

# digital filter initialization
time_span = cutoff_period = physics_specs.nondimensionalize(dfi_timescale)
dfi = jax.jit(time_integration.digital_filter_initialization(
    equation=eq,
    ode_solver=time_integration.imex_rk_sil3,
    filters=[hyperdiffusion_filter],
    time_span=time_span,
    cutoff_period=cutoff_period,
    dt=dt,
))
dfi_init_state = jax.block_until_ready(dfi(raw_init_state))

def nodal_prognostics_and_diagnostics(state):
  coords = model_coords.horizontal
  u_nodal, v_nodal = spherical_harmonic.vor_div_to_uv_nodal(
      coords, state.vorticity, state.divergence)
  geopotential_nodal = coords.to_nodal(
      primitive_equations.get_geopotential(
          state.temperature_variation,
          eq.reference_temperature,
          orography,
          model_coords.vertical,
          ideal_gas_constant =0.00033225165572835946,
          gravity_acceleration=72.36408283456718,
      )
  )
  primitive_equations.get_geopotential
  vor_nodal = coords.to_nodal(state.vorticity)
  div_nodal = coords.to_nodal(state.divergence)
  sp_nodal = jnp.exp(coords.to_nodal(state.log_surface_pressure))
  tracers_nodal = {k: coords.to_nodal(v) for k, v in state.tracers.items()}
  t_nodal = (
      coords.to_nodal(state.temperature_variation)
      + ref_temps[:, np.newaxis, np.newaxis]
  )
  vertical_velocity_nodal = primitive_equations.compute_vertical_velocity(
      state, model_coords
  )
  state_nodal = {
      'u_component_of_wind': u_nodal,
      'v_component_of_wind': v_nodal,
      'temperature': t_nodal,
      'vorticity': vor_nodal,
      'divergence': div_nodal,
      'vertical_velocity': vertical_velocity_nodal,
      'geopotential': geopotential_nodal,
      'surface_pressure': sp_nodal,
      **tracers_nodal,
  }
  return slice_levels(state_nodal, output_level_indices)


def trajectory_to_xarray(trajectory):

  # convert units back to SI
  target_units = {k: v.data.units for k, v in ds_init.items()}
  target_units |= {
      'vorticity': units('1/s'),
      'divergence': units('1/s'),
      'geopotential': units('m^2/s^2'),
      'vertical_velocity': units('1/s'),
  }

  orography_nodal = jax.device_put(model_coords.horizontal.to_nodal(orography), device=jax.devices('cpu')[0])
  trajectory_cpu = jax.device_put(trajectory, device=jax.devices('cpu')[0])

  traj_nodal_si = {
      k: physics_specs.dimensionalize(v, target_units[k]).magnitude
      for k, v in trajectory_cpu.items()
  }

  # build xarray
  times = float(save_every / units.hour) * np.arange(outer_steps)
  lon = 180/np.pi * model_coords.horizontal.nodal_axes[0]
  lat = 180/np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])

  dims = ('time', 'sigma', 'longitude', 'latitude')
  ds_result = xarray.Dataset(
      data_vars={
          k: (dims, v) for k, v in traj_nodal_si.items() if k != 'surface_pressure'
      },
      coords={
          'longitude': lon,
          'latitude': lat,
          'sigma': model_coords.vertical.centers[output_level_indices],
          'time': times,
          'orography': (('longitude', 'latitude'), orography_nodal.squeeze()),
      },
  ).assign(
      surface_pressure=(
          ('time', 'longitude', 'latitude'),
          traj_nodal_si['surface_pressure'].squeeze(axis=-3),
      )
  )
  return ds_result

# temporal integration function
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
step_fn = time_integration.step_with_filters(
    time_integration.imex_rk_sil3(eq, dt),
    [hyperdiffusion_filter],
)
integrate_fn = jax.jit(time_integration.trajectory_from_step(
    step_fn,
    outer_steps=outer_steps,
    inner_steps=inner_steps,
    start_with_input=True,
    post_process_fn=nodal_prognostics_and_diagnostics,
))

# ===== TARGET CONFIGURATION =====
TARGET_TEMP = 310.0  # Kelvin
TARGET_IND = (20,30)
lytton_lat =  50.231111 
lytton_lon = 121.581389
# Convert Lytton longitude to 0-360 range
lytton_lon_positive = (360 - lytton_lon) % 360
# Find the closest latitude
closest_lat_index = np.abs(desired_lat - lytton_lat).argmin()
closest_lat = desired_lat[closest_lat_index]
# Find the closest longitude
closest_lon_index = np.abs(desired_lon - lytton_lon_positive).argmin()
closest_lon = desired_lon[closest_lon_index]
out_state, trajectory = jax.block_until_ready(integrate_fn(dfi_init_state))
ds = trajectory_to_xarray(trajectory)
flat_index = ds["temperature"].argmax().item()

# Convert the flattened index to multi-dimensional coordinates
coords = np.unravel_index(flat_index, ds["temperature"].shape)

# Map the coordinates to the actual dimension values
max_coords = {
    "time": ds["time"].values[coords[0]],
    "sigma": ds["sigma"].values[coords[1]],
    "longitude": ds["longitude"].values[coords[2]],
    "latitude": ds["latitude"].values[coords[3]],
}

print("Maximum coords:", max_coords)

# ===== LOSS WITH EXPLICIT PARAMETER =====
@jit
def loss_fn(log_surface_pressure, static_state):
    log_p = static_state.log_surface_pressure
    updated_state = static_state.replace(
        log_surface_pressure=log_surface_pressure
    )
    updated_state = jax.block_until_ready(dfi(updated_state))

    # Use checkpointed integration to save memory
    @checkpoint  # <-- Recomputation during backward pass
    def _run_model(state):
        _, trajectory = integrate_fn(state)
        return trajectory['temperature']

    lam = 0
    temp_traj = _run_model(updated_state)
    temp_final = jnp.mean(temp_traj[:-32, -1, closest_lon_index:closest_lon_index+4, closest_lat_index:closest_lat_index+4],axis=0)
    jax.debug.print("Measured temp {x}", x=jnp.mean(temp_final))
    jax.debug.print("I.C. MSE {x}", x=lam*jnp.mean(log_surface_pressure-log_p)**2)
    return -jnp.mean(temp_final)+ lam*jnp.mean((log_surface_pressure-log_p)**2) 

# Differentiate only w.r.t. log_surface_pressure
loss_and_grad_fn = jit(value_and_grad(loss_fn, argnums=0))  
# Initialize optimizer state with ONLY log_surface_pressure as parameter
optimizer = optax.adam(learning_rate=1e-4)
log_sp = dfi_init_state.log_surface_pressure  # Initial value
opt_state = optimizer.init(log_sp)

# Split state into trainable (log_sp) and static parts
static_state = raw_init_state #.replace(log_surface_pressure=None)

for step in range(50):
    loss_val, grad = loss_and_grad_fn(log_sp, static_state)  # Grad ≈ [∂loss/∂log_sp]
    updates, opt_state = optimizer.update(grad, opt_state)  # Update log_sp
    log_sp = optax.apply_updates(log_sp, updates)

    print(f"Step {step}: Loss = {loss_val:.3f}")


# Finalize: Rebuild full state with optimized log_sp
optimized_state = static_state.replace(log_surface_pressure=log_sp)

opt_stat_filter = jax.block_until_ready(dfi(optimized_state))
out_state, trajectory = jax.block_until_ready(integrate_fn(opt_stat_filter))
ds_out_opt = trajectory_to_xarray(trajectory)

ds_out_opt.to_netcdf("optimized.nc")

out_state, trajectory = jax.block_until_ready(integrate_fn(dfi_init_state))
ds_out_ogi = trajectory_to_xarray(trajectory)

ds_out_ogi.to_netcdf("original.nc")
