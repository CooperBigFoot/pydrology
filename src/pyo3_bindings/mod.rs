#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Register the _core Python module.
#[cfg(feature = "python")]
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_version, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
