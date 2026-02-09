//! CemaNeige numerical constants.
//!
//! Fixed values for the CemaNeige snow accumulation and melt model.

/// Melt threshold temperature [C].
pub const T_MELT: f64 = 0.0;

/// Minimum melt fraction [-]. Even with minimal snow cover,
/// melt proceeds at at least this fraction of potential.
pub const MIN_SPEED: f64 = 0.1;

/// All-snow threshold temperature [C] (USACE formula).
pub const T_SNOW: f64 = -1.0;

/// All-rain threshold temperature [C] (USACE formula).
pub const T_RAIN: f64 = 3.0;

/// Fraction of mean annual solid precip for initial gthreshold.
pub const GTHRESHOLD_FACTOR: f64 = 0.9;

/// Number of state variables per snow layer: [g, etg, gthreshold, glocalmax].
pub const LAYER_STATE_SIZE: usize = 4;

/// Number of CemaNeige parameters (ctg, kf).
pub const N_PARAMS: usize = 2;

/// Parameter names in canonical order.
pub const PARAM_NAMES: &[&str] = &["ctg", "kf"];

/// Parameter bounds as (min, max) tuples, in PARAM_NAMES order.
pub const PARAM_BOUNDS: &[(f64, f64)] = &[
    (0.0, 1.0),  // ctg
    (0.0, 10.0), // kf
];

/// Number of flux outputs per layer step.
pub const N_FLUXES: usize = 11;

/// Supported temporal resolutions for forcing data.
pub const SUPPORTED_RESOLUTIONS: &[crate::forcing::Resolution] = &[crate::forcing::Resolution::Daily];

/// Elevation constants (same as pydrology.utils.elevation).
pub const GRAD_T_DEFAULT: f64 = 0.6;
pub const GRAD_P_DEFAULT: f64 = 0.00041;
pub const ELEV_CAP_PRECIP: f64 = 4000.0;
