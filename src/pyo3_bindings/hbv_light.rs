use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::hbv_light::params::Parameters;
use crate::hbv_light::routing::compute_triangular_weights;
use crate::hbv_light::run;
use crate::hbv_light::state::State;

/// Run HBV-Light over a timeseries.
///
/// Returns (fluxes_dict, optional_zone_outputs_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    params,
    precip,
    pet,
    temp,
    initial_state=None,
    n_zones=1,
    zone_elevations=None,
    zone_fractions=None,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
))]
fn hbv_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
    n_zones: usize,
    zone_elevations: Option<PyReadonlyArray1<'py, f64>>,
    zone_fractions: Option<PyReadonlyArray1<'py, f64>>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
) -> PyResult<(Bound<'py, PyDict>, Option<Bound<'py, PyDict>>)> {
    let p_slice = params.as_slice()?;
    let p = Parameters::from_array(p_slice)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let precip_slice = precip.as_slice()?;
    let pet_slice = pet.as_slice()?;
    let temp_slice = temp.as_slice()?;

    let state = match &initial_state {
        Some(s) => {
            let s_slice = s.as_slice()?;
            Some(State::from_array(s_slice, n_zones))
        }
        None => None,
    };

    // Extract zone arrays
    let ze_vec: Vec<f64>;
    let zone_elevs = match &zone_elevations {
        Some(ze) => {
            ze_vec = ze.as_slice()?.to_vec();
            Some(ze_vec.as_slice())
        }
        None => None,
    };

    let zf_vec: Vec<f64>;
    let zone_fracs = match &zone_fractions {
        Some(zf) => {
            zf_vec = zf.as_slice()?.to_vec();
            Some(zf_vec.as_slice())
        }
        None => None,
    };

    let (result, zone_outputs) = run::run(
        &p,
        precip_slice,
        pet_slice,
        temp_slice,
        state.as_ref(),
        n_zones,
        zone_elevs,
        zone_fracs,
        input_elevation,
        temp_gradient,
        precip_gradient,
    );

    // Build fluxes dict
    let dict = PyDict::new(py);
    dict.set_item("precip", PyArray1::from_vec(py, result.precip))?;
    dict.set_item("temp", PyArray1::from_vec(py, result.temp))?;
    dict.set_item("pet", PyArray1::from_vec(py, result.pet))?;
    dict.set_item("precip_rain", PyArray1::from_vec(py, result.precip_rain))?;
    dict.set_item("precip_snow", PyArray1::from_vec(py, result.precip_snow))?;
    dict.set_item("snow_pack", PyArray1::from_vec(py, result.snow_pack))?;
    dict.set_item("snow_melt", PyArray1::from_vec(py, result.snow_melt))?;
    dict.set_item("liquid_water_in_snow", PyArray1::from_vec(py, result.liquid_water_in_snow))?;
    dict.set_item("snow_input", PyArray1::from_vec(py, result.snow_input))?;
    dict.set_item("soil_moisture", PyArray1::from_vec(py, result.soil_moisture))?;
    dict.set_item("recharge", PyArray1::from_vec(py, result.recharge))?;
    dict.set_item("actual_et", PyArray1::from_vec(py, result.actual_et))?;
    dict.set_item("upper_zone", PyArray1::from_vec(py, result.upper_zone))?;
    dict.set_item("lower_zone", PyArray1::from_vec(py, result.lower_zone))?;
    dict.set_item("q0", PyArray1::from_vec(py, result.q0))?;
    dict.set_item("q1", PyArray1::from_vec(py, result.q1))?;
    dict.set_item("q2", PyArray1::from_vec(py, result.q2))?;
    dict.set_item("percolation", PyArray1::from_vec(py, result.percolation))?;
    dict.set_item("qgw", PyArray1::from_vec(py, result.qgw))?;
    dict.set_item("streamflow", PyArray1::from_vec(py, result.streamflow))?;

    // Build zone outputs dict if present
    let zone_dict = match zone_outputs {
        Some(zo) => {
            let zd = PyDict::new(py);
            zd.set_item("zone_elevations", PyArray1::from_vec(py, zo.zone_elevations))?;
            zd.set_item("zone_fractions", PyArray1::from_vec(py, zo.zone_fractions))?;

            // Convert Vec<Vec<f64>> to 2D: flatten to 1D and store shape
            let n_timesteps = zo.zone_temp.len();
            let n_z = if n_timesteps > 0 { zo.zone_temp[0].len() } else { 0 };

            // Helper to flatten 2D Vec<Vec<f64>> to contiguous Vec<f64> (row-major)
            let flatten_2d = |data: &[Vec<f64>]| -> Vec<f64> {
                let mut flat = Vec::with_capacity(n_timesteps * n_z);
                for row in data {
                    flat.extend_from_slice(row);
                }
                flat
            };

            // Store as flat arrays with shape info for Python to reshape
            zd.set_item("n_timesteps", n_timesteps)?;
            zd.set_item("n_zones", n_z)?;
            zd.set_item("zone_temp", PyArray1::from_vec(py, flatten_2d(&zo.zone_temp)))?;
            zd.set_item("zone_precip", PyArray1::from_vec(py, flatten_2d(&zo.zone_precip)))?;
            zd.set_item("snow_pack", PyArray1::from_vec(py, flatten_2d(&zo.snow_pack)))?;
            zd.set_item("liquid_water_in_snow", PyArray1::from_vec(py, flatten_2d(&zo.liquid_water_in_snow)))?;
            zd.set_item("snow_melt", PyArray1::from_vec(py, flatten_2d(&zo.snow_melt)))?;
            zd.set_item("snow_input", PyArray1::from_vec(py, flatten_2d(&zo.snow_input)))?;
            zd.set_item("soil_moisture", PyArray1::from_vec(py, flatten_2d(&zo.soil_moisture)))?;
            zd.set_item("recharge", PyArray1::from_vec(py, flatten_2d(&zo.recharge)))?;
            zd.set_item("actual_et", PyArray1::from_vec(py, flatten_2d(&zo.actual_et)))?;

            Some(zd)
        }
        None => None,
    };

    Ok((dict, zone_dict))
}

/// Execute one timestep of HBV-Light.
///
/// Returns (new_state_array, fluxes_dict).
#[pyfunction]
fn hbv_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    temp: f64,
    uh_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = params.as_slice()?;
    let p = Parameters::from_array(p_slice)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let s_slice = state.as_slice()?;
    let s = State::from_array(s_slice, 1);

    let uh_slice = uh_weights.as_slice()?;

    let (new_state, fluxes) = run::step(&s, &p, precip, pet, temp, uh_slice);

    let state_arr = PyArray1::from_vec(py, new_state.to_array());

    let dict = PyDict::new(py);
    dict.set_item("precip", fluxes.precip)?;
    dict.set_item("temp", fluxes.temp)?;
    dict.set_item("pet", fluxes.pet)?;
    dict.set_item("precip_rain", fluxes.precip_rain)?;
    dict.set_item("precip_snow", fluxes.precip_snow)?;
    dict.set_item("snow_pack", fluxes.snow_pack)?;
    dict.set_item("snow_melt", fluxes.snow_melt)?;
    dict.set_item("liquid_water_in_snow", fluxes.liquid_water_in_snow)?;
    dict.set_item("snow_input", fluxes.snow_input)?;
    dict.set_item("soil_moisture", fluxes.soil_moisture)?;
    dict.set_item("recharge", fluxes.recharge)?;
    dict.set_item("actual_et", fluxes.actual_et)?;
    dict.set_item("upper_zone", fluxes.upper_zone)?;
    dict.set_item("lower_zone", fluxes.lower_zone)?;
    dict.set_item("q0", fluxes.q0)?;
    dict.set_item("q1", fluxes.q1)?;
    dict.set_item("q2", fluxes.q2)?;
    dict.set_item("percolation", fluxes.percolation)?;
    dict.set_item("qgw", fluxes.qgw)?;
    dict.set_item("streamflow", fluxes.streamflow)?;

    Ok((state_arr, dict))
}

/// Compute triangular unit hydrograph weights.
#[pyfunction]
fn hbv_triangular_weights<'py>(
    py: Python<'py>,
    maxbas: f64,
) -> Bound<'py, PyArray1<f64>> {
    let weights = compute_triangular_weights(maxbas);
    PyArray1::from_vec(py, weights)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "hbv_light")?;
    m.add_function(wrap_pyfunction!(hbv_run, &m)?)?;
    m.add_function(wrap_pyfunction!(hbv_step, &m)?)?;
    m.add_function(wrap_pyfunction!(hbv_triangular_weights, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pydrology._core.hbv_light", &m)?;
    Ok(())
}
