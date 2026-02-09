use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cemaneige::constants::LAYER_STATE_SIZE;
use crate::cemaneige::coupled;
use crate::gr6j::constants::NH;
use crate::gr6j::params::Parameters as GR6JParameters;
use crate::gr6j::state::State as GR6JState;

/// Run the coupled GR6J-CemaNeige model over a timeseries.
///
/// Returns (snow_dict, gr6j_dict, layer_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    params,
    precip,
    pet,
    temp,
    initial_state=None,
    uh1_ordinates=None,
    uh2_ordinates=None,
    n_layers=1,
    layer_elevations=None,
    layer_fractions=None,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
    mean_annual_solid_precip=0.0,
))]
fn gr6j_cemaneige_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
    uh1_ordinates: Option<PyReadonlyArray1<'py, f64>>,
    uh2_ordinates: Option<PyReadonlyArray1<'py, f64>>,
    n_layers: usize,
    layer_elevations: Option<PyReadonlyArray1<'py, f64>>,
    layer_fractions: Option<PyReadonlyArray1<'py, f64>>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
    mean_annual_solid_precip: f64,
) -> PyResult<(
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
)> {
    let p_slice = params.as_slice()?;
    if p_slice.len() < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params must have at least 8 elements, got {}",
            p_slice.len()
        )));
    }

    let gr6j_params =
        GR6JParameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);
    let ctg = p_slice[6];
    let kf = p_slice[7];

    let precip_slice = precip.as_slice()?;
    let pet_slice = pet.as_slice()?;
    let temp_slice = temp.as_slice()?;

    // Build layer config
    let default_elevs = vec![0.0f64; n_layers];
    let default_fracs = vec![1.0 / n_layers as f64; n_layers];
    let layer_elevs: Vec<f64> = match &layer_elevations {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => default_elevs,
    };
    let layer_fracs: Vec<f64> = match &layer_fractions {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => default_fracs,
    };

    let input_elev = input_elevation.unwrap_or(f64::NAN);
    let t_grad = temp_gradient.unwrap_or(crate::cemaneige::constants::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(crate::cemaneige::constants::GRAD_P_DEFAULT);

    // Parse initial state
    let (init_gr6j, init_snow) = match &initial_state {
        Some(s) => {
            let s_slice = s.as_slice()?;
            let expected = 63 + n_layers * LAYER_STATE_SIZE;
            if s_slice.len() != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "initial_state must have {} elements (63 GR6J + {} * {} snow), got {}",
                    expected,
                    n_layers,
                    LAYER_STATE_SIZE,
                    s_slice.len()
                )));
            }
            let mut gr6j_arr = [0.0f64; 63];
            gr6j_arr.copy_from_slice(&s_slice[..63]);
            let gr6j_state = GR6JState::from_array(&gr6j_arr);

            let mut snow_states = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let base = 63 + i * LAYER_STATE_SIZE;
                let mut ls = [0.0f64; LAYER_STATE_SIZE];
                ls.copy_from_slice(&s_slice[base..base + LAYER_STATE_SIZE]);
                snow_states.push(ls);
            }
            (Some(gr6j_state), Some(snow_states))
        }
        None => (None, None),
    };

    // Handle pre-computed UH ordinates (ignore them — coupled_run computes internally)
    let _ = uh1_ordinates;
    let _ = uh2_ordinates;

    let result = coupled::coupled_run(
        &gr6j_params,
        ctg,
        kf,
        precip_slice,
        pet_slice,
        temp_slice,
        init_gr6j.as_ref(),
        init_snow.as_deref(),
        n_layers,
        &layer_elevs,
        &layer_fracs,
        input_elev,
        t_grad,
        p_grad,
        mean_annual_solid_precip,
    );

    // Build snow output dict
    let snow_dict = PyDict::new(py);
    snow_dict.set_item("snow_pliq", PyArray1::from_vec(py, result.snow.pliq))?;
    snow_dict.set_item("snow_psol", PyArray1::from_vec(py, result.snow.psol))?;
    snow_dict.set_item("snow_pack", PyArray1::from_vec(py, result.snow.snow_pack))?;
    snow_dict.set_item("snow_thermal_state", PyArray1::from_vec(py, result.snow.thermal_state))?;
    snow_dict.set_item("snow_gratio", PyArray1::from_vec(py, result.snow.gratio))?;
    snow_dict.set_item("snow_pot_melt", PyArray1::from_vec(py, result.snow.pot_melt))?;
    snow_dict.set_item("snow_melt", PyArray1::from_vec(py, result.snow.melt))?;
    snow_dict.set_item("snow_pliq_and_melt", PyArray1::from_vec(py, result.snow.pliq_and_melt))?;
    snow_dict.set_item("snow_temp", PyArray1::from_vec(py, result.snow.temp))?;

    // Build GR6J output dict
    let gr6j_dict = PyDict::new(py);
    gr6j_dict.set_item("pet", PyArray1::from_vec(py, result.gr6j.pet))?;
    gr6j_dict.set_item("precip", PyArray1::from_vec(py, result.gr6j.precip))?;
    gr6j_dict.set_item("production_store", PyArray1::from_vec(py, result.gr6j.production_store))?;
    gr6j_dict.set_item("net_rainfall", PyArray1::from_vec(py, result.gr6j.net_rainfall))?;
    gr6j_dict.set_item("storage_infiltration", PyArray1::from_vec(py, result.gr6j.storage_infiltration))?;
    gr6j_dict.set_item("actual_et", PyArray1::from_vec(py, result.gr6j.actual_et))?;
    gr6j_dict.set_item("percolation", PyArray1::from_vec(py, result.gr6j.percolation))?;
    gr6j_dict.set_item("effective_rainfall", PyArray1::from_vec(py, result.gr6j.effective_rainfall))?;
    gr6j_dict.set_item("q9", PyArray1::from_vec(py, result.gr6j.q9))?;
    gr6j_dict.set_item("q1", PyArray1::from_vec(py, result.gr6j.q1))?;
    gr6j_dict.set_item("routing_store", PyArray1::from_vec(py, result.gr6j.routing_store))?;
    gr6j_dict.set_item("exchange", PyArray1::from_vec(py, result.gr6j.exchange))?;
    gr6j_dict.set_item("actual_exchange_routing", PyArray1::from_vec(py, result.gr6j.actual_exchange_routing))?;
    gr6j_dict.set_item("actual_exchange_direct", PyArray1::from_vec(py, result.gr6j.actual_exchange_direct))?;
    gr6j_dict.set_item("actual_exchange_total", PyArray1::from_vec(py, result.gr6j.actual_exchange_total))?;
    gr6j_dict.set_item("qr", PyArray1::from_vec(py, result.gr6j.qr))?;
    gr6j_dict.set_item("qrexp", PyArray1::from_vec(py, result.gr6j.qrexp))?;
    gr6j_dict.set_item("exponential_store", PyArray1::from_vec(py, result.gr6j.exponential_store))?;
    gr6j_dict.set_item("qd", PyArray1::from_vec(py, result.gr6j.qd))?;
    gr6j_dict.set_item("streamflow", PyArray1::from_vec(py, result.gr6j.streamflow))?;

    // Build per-layer outputs dict
    let layer_dict = PyDict::new(py);
    let n_timesteps = precip_slice.len();

    // Build 2D arrays (n_timesteps x n_layers) for each layer field
    let mut layer_snow_pack = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_thermal = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_gratio = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_melt = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_pliq_and_melt = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_temp = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_precip_arr = vec![0.0f64; n_timesteps * n_layers];

    for t in 0..n_timesteps {
        for l in 0..n_layers {
            let idx = t * n_layers + l;
            let lf = &result.layers[t][l];
            layer_snow_pack[idx] = lf.snow_pack;
            layer_thermal[idx] = lf.thermal_state;
            layer_gratio[idx] = lf.gratio;
            layer_melt[idx] = lf.melt;
            layer_pliq_and_melt[idx] = lf.pliq_and_melt;
            layer_temp[idx] = lf.temp;
            layer_precip_arr[idx] = lf.precip;
        }
    }

    layer_dict.set_item(
        "snow_pack",
        PyArray2::from_vec2(py, &to_2d(&layer_snow_pack, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_thermal_state",
        PyArray2::from_vec2(py, &to_2d(&layer_thermal, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_gratio",
        PyArray2::from_vec2(py, &to_2d(&layer_gratio, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_melt",
        PyArray2::from_vec2(py, &to_2d(&layer_melt, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_pliq_and_melt",
        PyArray2::from_vec2(py, &to_2d(&layer_pliq_and_melt, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "layer_temp",
        PyArray2::from_vec2(py, &to_2d(&layer_temp, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "layer_precip",
        PyArray2::from_vec2(py, &to_2d(&layer_precip_arr, n_timesteps, n_layers))?,
    )?;

    Ok((snow_dict, gr6j_dict, layer_dict))
}

/// Convert a flat 1D array to a Vec<Vec<f64>> for PyArray2::from_vec2.
fn to_2d(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
        .collect()
}

/// Execute one timestep of the coupled GR6J-CemaNeige model.
///
/// Returns (new_state, fluxes_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    state,
    params,
    precip,
    pet,
    temp,
    uh1_ordinates,
    uh2_ordinates,
    layer_elevations,
    layer_fractions,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
))]
fn gr6j_cemaneige_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    temp: f64,
    uh1_ordinates: PyReadonlyArray1<'py, f64>,
    uh2_ordinates: PyReadonlyArray1<'py, f64>,
    layer_elevations: PyReadonlyArray1<'py, f64>,
    layer_fractions: PyReadonlyArray1<'py, f64>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = params.as_slice()?;
    if p_slice.len() < 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params must have at least 8 elements, got {}",
            p_slice.len()
        )));
    }

    let gr6j_params =
        GR6JParameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);
    let ctg = p_slice[6];
    let kf = p_slice[7];

    let s_slice = state.as_slice()?;
    let elevs = layer_elevations.as_slice()?;
    let fracs = layer_fractions.as_slice()?;
    let n_layers = elevs.len();

    let expected_state_size = 63 + n_layers * LAYER_STATE_SIZE;
    if s_slice.len() != expected_state_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state must have {} elements, got {}",
            expected_state_size,
            s_slice.len()
        )));
    }

    // Parse GR6J state
    let mut gr6j_arr = [0.0f64; 63];
    gr6j_arr.copy_from_slice(&s_slice[..63]);
    let gr6j_state = GR6JState::from_array(&gr6j_arr);

    // Parse snow layer states
    let mut snow_states = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let base = 63 + i * LAYER_STATE_SIZE;
        let mut ls = [0.0f64; LAYER_STATE_SIZE];
        ls.copy_from_slice(&s_slice[base..base + LAYER_STATE_SIZE]);
        snow_states.push(ls);
    }

    // Parse UH ordinates
    let uh1_slice = uh1_ordinates.as_slice()?;
    if uh1_slice.len() != NH {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "uh1_ordinates must have {} elements, got {}",
            NH,
            uh1_slice.len()
        )));
    }
    let mut uh1_arr = [0.0f64; NH];
    uh1_arr.copy_from_slice(uh1_slice);

    let uh2_slice = uh2_ordinates.as_slice()?;
    if uh2_slice.len() != 2 * NH {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "uh2_ordinates must have {} elements, got {}",
            2 * NH,
            uh2_slice.len()
        )));
    }
    let mut uh2_arr = [0.0f64; 2 * NH];
    uh2_arr.copy_from_slice(uh2_slice);

    let input_elev = input_elevation.unwrap_or(f64::NAN);
    let t_grad = temp_gradient.unwrap_or(crate::cemaneige::constants::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(crate::cemaneige::constants::GRAD_P_DEFAULT);
    let skip_extrap = input_elev.is_nan();

    let (new_gr6j, new_snow, snow_fluxes, gr6j_fluxes, _per_layer) = coupled::coupled_step(
        &gr6j_state,
        &snow_states,
        &gr6j_params,
        ctg,
        kf,
        precip,
        pet,
        temp,
        &uh1_arr,
        &uh2_arr,
        elevs,
        fracs,
        input_elev,
        t_grad,
        p_grad,
        skip_extrap,
    );

    // Build new state array
    let gr6j_state_arr = new_gr6j.to_array();
    let mut state_out = Vec::with_capacity(expected_state_size);
    state_out.extend_from_slice(&gr6j_state_arr);
    for ls in &new_snow {
        state_out.extend_from_slice(ls);
    }

    // Build fluxes dict
    let dict = PyDict::new(py);
    // Snow fluxes
    dict.set_item("precip_raw", precip)?;
    dict.set_item("snow_pliq", snow_fluxes.pliq)?;
    dict.set_item("snow_psol", snow_fluxes.psol)?;
    dict.set_item("snow_pack", snow_fluxes.snow_pack)?;
    dict.set_item("snow_thermal_state", snow_fluxes.thermal_state)?;
    dict.set_item("snow_gratio", snow_fluxes.gratio)?;
    dict.set_item("snow_pot_melt", snow_fluxes.pot_melt)?;
    dict.set_item("snow_melt", snow_fluxes.melt)?;
    dict.set_item("snow_pliq_and_melt", snow_fluxes.pliq_and_melt)?;
    dict.set_item("snow_temp", snow_fluxes.temp)?;
    // GR6J fluxes
    dict.set_item("pet", gr6j_fluxes.pet)?;
    dict.set_item("precip", gr6j_fluxes.precip)?;
    dict.set_item("production_store", gr6j_fluxes.production_store)?;
    dict.set_item("net_rainfall", gr6j_fluxes.net_rainfall)?;
    dict.set_item("storage_infiltration", gr6j_fluxes.storage_infiltration)?;
    dict.set_item("actual_et", gr6j_fluxes.actual_et)?;
    dict.set_item("percolation", gr6j_fluxes.percolation)?;
    dict.set_item("effective_rainfall", gr6j_fluxes.effective_rainfall)?;
    dict.set_item("q9", gr6j_fluxes.q9)?;
    dict.set_item("q1", gr6j_fluxes.q1)?;
    dict.set_item("routing_store", gr6j_fluxes.routing_store)?;
    dict.set_item("exchange", gr6j_fluxes.exchange)?;
    dict.set_item("actual_exchange_routing", gr6j_fluxes.actual_exchange_routing)?;
    dict.set_item("actual_exchange_direct", gr6j_fluxes.actual_exchange_direct)?;
    dict.set_item("actual_exchange_total", gr6j_fluxes.actual_exchange_total)?;
    dict.set_item("qr", gr6j_fluxes.qr)?;
    dict.set_item("qrexp", gr6j_fluxes.qrexp)?;
    dict.set_item("exponential_store", gr6j_fluxes.exponential_store)?;
    dict.set_item("qd", gr6j_fluxes.qd)?;
    dict.set_item("streamflow", gr6j_fluxes.streamflow)?;

    Ok((PyArray1::from_vec(py, state_out), dict))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "cemaneige")?;
    m.add_function(wrap_pyfunction!(gr6j_cemaneige_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_cemaneige_step, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
