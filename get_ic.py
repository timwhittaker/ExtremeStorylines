import gcsfs
import jax
import numpy as np
import pickle
import xarray
import sys

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

import matplotlib.pyplot as plt
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

gcs = gcsfs.GCSFileSystem(token='anon')

# Select model
model_name = 'v1/deterministic_2_8_deg.pkl'  #@param ['v1/deterministic_0_7_deg.pkl', 'v1/deterministic_1_4_deg.pkl', 'v1/deterministic_2_8_deg.pkl', 'v1/stochastic_1_4_deg.pkl', 'v1_precip/stochastic_precip_2_8_deg.pkl', 'v1_precip/stochastic_evap_2_8_deg'] {type: "string"}

with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

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

# Access data on google storage
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)
demo_start_time = '2021-06-18T00:00:00'
demo_end_time = '2021-06-18T01:00:00'
data_inner_steps = 1  # process every 24th hour

sliced_era5 = (
    full_era5
    [model.input_variables + model.forcing_variables]
    .pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift='24 hours',
    )
    .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
    .compute()
)

# Save ic as zarr file
file_name="init_2021_06_18"
sliced_era5.to_zarr(f"{file_name}.zarr")
                                           
