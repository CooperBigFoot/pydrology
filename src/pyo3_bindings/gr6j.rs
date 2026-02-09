use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::gr6j::constants::NH;
use crate::gr6j::params::Parameters;
use crate::gr6j::run;
use crate::gr6j::state::State;
use crate::gr6j::unit_hydrographs::compute_uh_ordinates;

#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn gr6j_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p_slice = params.as_slice()?;
    let p = Parameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);

    let precip_slice = precip.as_slice()?;
    let pet_slice = pet.as_slice()?;

    let state = match &initial_state {
        Some(s) => {
            let s_slice = s.as_slice()?;
            if s_slice.len() != 63 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("initial_state must have 63 elements, got {}", s_slice.len()),
                ));
            }
            let mut arr = [0.0f64; 63];
            arr.copy_from_slice(s_slice);
            Some(State::from_array(&arr))
        }
        None => None,
    };

    let result = run::run(&p, precip_slice, pet_slice, state.as_ref());

    let dict = PyDict::new(py);
    dict.set_item("pet", PyArray1::from_vec(py, result.pet))?;
    dict.set_item("precip", PyArray1::from_vec(py, result.precip))?;
    dict.set_item("production_store", PyArray1::from_vec(py, result.production_store))?;
    dict.set_item("net_rainfall", PyArray1::from_vec(py, result.net_rainfall))?;
    dict.set_item("storage_infiltration", PyArray1::from_vec(py, result.storage_infiltration))?;
    dict.set_item("actual_et", PyArray1::from_vec(py, result.actual_et))?;
    dict.set_item("percolation", PyArray1::from_vec(py, result.percolation))?;
    dict.set_item("effective_rainfall", PyArray1::from_vec(py, result.effective_rainfall))?;
    dict.set_item("q9", PyArray1::from_vec(py, result.q9))?;
    dict.set_item("q1", PyArray1::from_vec(py, result.q1))?;
    dict.set_item("routing_store", PyArray1::from_vec(py, result.routing_store))?;
    dict.set_item("exchange", PyArray1::from_vec(py, result.exchange))?;
    dict.set_item("actual_exchange_routing", PyArray1::from_vec(py, result.actual_exchange_routing))?;
    dict.set_item("actual_exchange_direct", PyArray1::from_vec(py, result.actual_exchange_direct))?;
    dict.set_item("actual_exchange_total", PyArray1::from_vec(py, result.actual_exchange_total))?;
    dict.set_item("qr", PyArray1::from_vec(py, result.qr))?;
    dict.set_item("qrexp", PyArray1::from_vec(py, result.qrexp))?;
    dict.set_item("exponential_store", PyArray1::from_vec(py, result.exponential_store))?;
    dict.set_item("qd", PyArray1::from_vec(py, result.qd))?;
    dict.set_item("streamflow", PyArray1::from_vec(py, result.streamflow))?;
    Ok(dict)
}

#[pyfunction]
fn gr6j_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    uh1_ordinates: PyReadonlyArray1<'py, f64>,
    uh2_ordinates: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = params.as_slice()?;
    let p = Parameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);

    let s_slice = state.as_slice()?;
    if s_slice.len() != 63 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("state must have 63 elements, got {}", s_slice.len()),
        ));
    }
    let mut state_arr = [0.0f64; 63];
    state_arr.copy_from_slice(s_slice);
    let s = State::from_array(&state_arr);

    let uh1_slice = uh1_ordinates.as_slice()?;
    if uh1_slice.len() != NH {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("uh1_ordinates must have {} elements, got {}", NH, uh1_slice.len()),
        ));
    }
    let mut uh1_arr = [0.0f64; NH];
    uh1_arr.copy_from_slice(uh1_slice);

    let uh2_slice = uh2_ordinates.as_slice()?;
    if uh2_slice.len() != 2 * NH {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("uh2_ordinates must have {} elements, got {}", 2 * NH, uh2_slice.len()),
        ));
    }
    let mut uh2_arr = [0.0f64; 2 * NH];
    uh2_arr.copy_from_slice(uh2_slice);

    let (new_state, fluxes) = run::step(&s, &p, precip, pet, &uh1_arr, &uh2_arr);

    let new_state_arr = new_state.to_array();
    let state_out = PyArray1::from_vec(py, new_state_arr.to_vec());

    let dict = PyDict::new(py);
    dict.set_item("pet", fluxes.pet)?;
    dict.set_item("precip", fluxes.precip)?;
    dict.set_item("production_store", fluxes.production_store)?;
    dict.set_item("net_rainfall", fluxes.net_rainfall)?;
    dict.set_item("storage_infiltration", fluxes.storage_infiltration)?;
    dict.set_item("actual_et", fluxes.actual_et)?;
    dict.set_item("percolation", fluxes.percolation)?;
    dict.set_item("effective_rainfall", fluxes.effective_rainfall)?;
    dict.set_item("q9", fluxes.q9)?;
    dict.set_item("q1", fluxes.q1)?;
    dict.set_item("routing_store", fluxes.routing_store)?;
    dict.set_item("exchange", fluxes.exchange)?;
    dict.set_item("actual_exchange_routing", fluxes.actual_exchange_routing)?;
    dict.set_item("actual_exchange_direct", fluxes.actual_exchange_direct)?;
    dict.set_item("actual_exchange_total", fluxes.actual_exchange_total)?;
    dict.set_item("qr", fluxes.qr)?;
    dict.set_item("qrexp", fluxes.qrexp)?;
    dict.set_item("exponential_store", fluxes.exponential_store)?;
    dict.set_item("qd", fluxes.qd)?;
    dict.set_item("streamflow", fluxes.streamflow)?;
    Ok((state_out, dict))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn gr6j_compute_uh_ordinates<'py>(
    py: Python<'py>,
    x4: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (uh1, uh2) = compute_uh_ordinates(x4);
    Ok((
        PyArray1::from_slice(py, &uh1),
        PyArray1::from_slice(py, &uh2),
    ))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "gr6j")?;
    m.add_function(wrap_pyfunction!(gr6j_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_step, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_compute_uh_ordinates, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pydrology._core.gr6j", &m)?;
    Ok(())
}
