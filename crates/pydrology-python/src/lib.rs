#[macro_use]
mod macros;
mod convert;

mod cemaneige;
mod gr2m;
mod gr6j;
mod hbv_light;
mod metrics;

use pyo3::prelude::*;

/// Register a submodule in sys.modules so `from parent.child import ...` works.
fn register_submodule(py: Python<'_>, parent_name: &str, child: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_name = child.name()?;
    let full_name = format!("{}.{}", parent_name, child_name);
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item(full_name, child)?;
    Ok(())
}

#[pyfunction]
fn rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let parent_name = m.name()?.to_string();

    m.add_function(wrap_pyfunction!(rust_version, m)?)?;

    cemaneige::register(m)?;
    gr2m::register(m)?;
    gr6j::register(m)?;
    hbv_light::register(m)?;
    metrics::register(m)?;

    // Register submodules in sys.modules for `from pydrology._core.X import ...`
    for name in &["cemaneige", "gr2m", "gr6j", "hbv_light", "metrics"] {
        let sub = m.getattr(*name)?;
        register_submodule(py, &parent_name, sub.downcast::<PyModule>()?)?;
    }

    Ok(())
}
