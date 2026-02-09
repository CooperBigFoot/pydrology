use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::gr2m::params::Parameters;
use crate::gr2m::run;
use crate::gr2m::state::State;

#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn gr2m_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p_slice = params.as_slice()?;
    let p = Parameters::new(p_slice[0], p_slice[1])
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let precip_slice = precip.as_slice()?;
    let pet_slice = pet.as_slice()?;

    let state = match &initial_state {
        Some(s) => {
            let s_slice = s.as_slice()?;
            Some(State {
                production_store: s_slice[0],
                routing_store: s_slice[1],
            })
        }
        None => None,
    };

    let result = run::run(&p, precip_slice, pet_slice, state.as_ref());

    let dict = PyDict::new(py);
    dict.set_item("pet", PyArray1::from_vec(py, result.pet))?;
    dict.set_item("precip", PyArray1::from_vec(py, result.precip))?;
    dict.set_item(
        "production_store",
        PyArray1::from_vec(py, result.production_store),
    )?;
    dict.set_item(
        "rainfall_excess",
        PyArray1::from_vec(py, result.rainfall_excess),
    )?;
    dict.set_item("storage_fill", PyArray1::from_vec(py, result.storage_fill))?;
    dict.set_item("actual_et", PyArray1::from_vec(py, result.actual_et))?;
    dict.set_item("percolation", PyArray1::from_vec(py, result.percolation))?;
    dict.set_item(
        "routing_input",
        PyArray1::from_vec(py, result.routing_input),
    )?;
    dict.set_item(
        "routing_store",
        PyArray1::from_vec(py, result.routing_store),
    )?;
    dict.set_item("exchange", PyArray1::from_vec(py, result.exchange))?;
    dict.set_item("streamflow", PyArray1::from_vec(py, result.streamflow))?;
    Ok(dict)
}

#[pyfunction]
fn gr2m_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = params.as_slice()?;
    let p = Parameters::new(p_slice[0], p_slice[1])
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let s_slice = state.as_slice()?;
    let s = State {
        production_store: s_slice[0],
        routing_store: s_slice[1],
    };

    let (new_state, fluxes) = run::step(&s, &p, precip, pet);

    let state_arr = PyArray1::from_vec(
        py,
        vec![new_state.production_store, new_state.routing_store],
    );

    let dict = PyDict::new(py);
    dict.set_item("pet", fluxes.pet)?;
    dict.set_item("precip", fluxes.precip)?;
    dict.set_item("production_store", fluxes.production_store)?;
    dict.set_item("rainfall_excess", fluxes.rainfall_excess)?;
    dict.set_item("storage_fill", fluxes.storage_fill)?;
    dict.set_item("actual_et", fluxes.actual_et)?;
    dict.set_item("percolation", fluxes.percolation)?;
    dict.set_item("routing_input", fluxes.routing_input)?;
    dict.set_item("routing_store", fluxes.routing_store)?;
    dict.set_item("exchange", fluxes.exchange)?;
    dict.set_item("streamflow", fluxes.streamflow)?;
    Ok((state_arr, dict))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "gr2m")?;
    m.add_function(wrap_pyfunction!(gr2m_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr2m_step, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pydrology._core.gr2m", &m)?;
    Ok(())
}
