/// GR2M numerical constants and model contract.
///
/// Centralises all fixed values used throughout the GR2M model.
/// Values are derived from the original Fortran implementation (airGR).
use crate::forcing::Resolution;

// -- Numerical safeguards --

/// Maximum argument for tanh to prevent overflow in production store.
pub const MAX_TANH_ARG: f64 = 13.0;

/// Constant in quadratic routing equation: Q = R² / (R + 60).
pub const ROUTING_DENOMINATOR: f64 = 60.0;

// -- Model contract constants --

/// Parameter names in order.
pub const PARAM_NAMES: &[&str] = &["x1", "x2"];

/// Number of elements in state array representation.
pub const STATE_SIZE: usize = 2;

/// Supported temporal resolutions for forcing data.
pub const SUPPORTED_RESOLUTIONS: &[Resolution] = &[Resolution::Monthly];

// -- Parameter bounds --

/// Parameter bounds for calibration: (min, max).
pub struct Bounds {
    pub min: f64,
    pub max: f64,
}

/// Production store capacity [mm].
pub const X1_BOUNDS: Bounds = Bounds {
    min: 1.0,
    max: 2500.0,
};

/// Groundwater exchange coefficient [-].
pub const X2_BOUNDS: Bounds = Bounds { min: 0.2, max: 2.0 };

/// Parameter bounds as (min, max) tuples, in PARAM_NAMES order.
pub const PARAM_BOUNDS: &[(f64, f64)] = &[
    (1.0, 2500.0), // x1
    (0.2, 2.0),    // x2
];
