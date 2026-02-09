/// Generate a frozen `#[pyclass]` struct where each field is a `Py<PyArray1<f64>>`.
///
/// Also generates a `from_timeseries()` method that converts from a
/// `FluxesTimeseries` (or `SnowTimeseries`) struct.
macro_rules! define_timeseries_result {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident from $core_type:ty {
            $($field:ident),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[pyo3::pyclass(frozen)]
        $vis struct $name {
            $(
                #[pyo3(get)]
                pub $field: Py<numpy::PyArray1<f64>>,
            )+
        }

        impl $name {
            pub fn from_timeseries(py: pyo3::Python<'_>, ts: $core_type) -> Self {
                Self {
                    $(
                        $field: numpy::PyArray1::from_vec(py, ts.$field).unbind(),
                    )+
                }
            }
        }
    };
}

/// Generate a frozen `#[pyclass]` struct where each field is `f64`.
///
/// Also generates a `from_fluxes()` method that copies values from the
/// corresponding Rust `Fluxes` struct.
macro_rules! define_step_result {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident from $core_type:ty {
            $($field:ident),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[pyo3::pyclass(frozen)]
        $vis struct $name {
            $(
                #[pyo3(get)]
                pub $field: f64,
            )+
        }

        impl $name {
            pub fn from_fluxes(f: &$core_type) -> Self {
                Self {
                    $(
                        $field: f.$field,
                    )+
                }
            }
        }
    };
}

/// Convert a `FluxesTimeseries` struct into a `PyDict`, preserving backward
/// compatibility with the existing dict-returning API.
macro_rules! timeseries_to_dict {
    ($py:expr, $ts:expr, $($field:ident),+ $(,)?) => {{
        let dict = pyo3::types::PyDict::new($py);
        $(
            dict.set_item(stringify!($field), numpy::PyArray1::from_vec($py, $ts.$field))?;
        )+
        dict
    }};
}

/// Convert a single-timestep `Fluxes` struct into a `PyDict`.
macro_rules! fluxes_to_dict {
    ($py:expr, $f:expr, $($field:ident),+ $(,)?) => {{
        let dict = pyo3::types::PyDict::new($py);
        $(
            dict.set_item(stringify!($field), $f.$field)?;
        )+
        dict
    }};
}
