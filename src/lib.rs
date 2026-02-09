/// pydrology — GR2M hydrological model in Rust.
///
/// A port of the GR2M model from pydrology, implementing the airGR
/// Fortran specification for monthly rainfall-runoff simulation.
pub mod cemaneige;
pub mod forcing;
pub mod gr2m;
pub mod gr6j;
pub mod hbv_light;
pub mod metrics;

#[cfg(feature = "python")]
mod pyo3_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_bindings::register(m)?;
    Ok(())
}
