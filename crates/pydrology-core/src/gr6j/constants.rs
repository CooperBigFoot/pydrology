/// GR6J numerical constants and model contract.
///
/// Centralises all fixed values used throughout the GR6J model.
/// Values are derived from the original Fortran implementation (airGR).
use crate::forcing::Resolution;

// -- Routing split fractions --

/// Fraction of effective rainfall to UH1 (slow branch).
pub const B: f64 = 0.9;

/// Fraction of UH1 output to exponential store.
pub const C: f64 = 0.4;

// -- Unit hydrograph parameters --

/// S-curve exponent.
pub const D: f64 = 2.5;

/// UH1 length (days). UH2 is 2*NH = 40 days.
pub const NH: usize = 20;

// -- Percolation constant --

/// Percolation constant: (9/4)^4 = 2.25^4
pub const PERC_CONSTANT: f64 = 25.62890625;

// -- Numerical safeguards --

/// Maximum argument for tanh to prevent overflow in production store.
pub const MAX_TANH_ARG: f64 = 13.0;

/// Maximum AR for exponential store clipping.
pub const MAX_EXP_ARG: f64 = 33.0;

/// Threshold for exponential store branch equations.
pub const EXP_BRANCH_THRESHOLD: f64 = 7.0;

// -- Model contract constants --

/// Parameter names in order.
pub const PARAM_NAMES: &[&str] = &["x1", "x2", "x3", "x4", "x5", "x6"];

/// Number of elements in state array representation.
pub const STATE_SIZE: usize = 63;

/// Supported temporal resolutions for forcing data.
pub const SUPPORTED_RESOLUTIONS: &[Resolution] = &[Resolution::Daily];

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

/// Intercatchment exchange coefficient [mm/day].
pub const X2_BOUNDS: Bounds = Bounds {
    min: -5.0,
    max: 5.0,
};

/// Routing store capacity [mm].
pub const X3_BOUNDS: Bounds = Bounds {
    min: 1.0,
    max: 1000.0,
};

/// Unit hydrograph time constant [days].
pub const X4_BOUNDS: Bounds = Bounds {
    min: 0.5,
    max: 10.0,
};

/// Intercatchment exchange threshold [-].
pub const X5_BOUNDS: Bounds = Bounds {
    min: -4.0,
    max: 4.0,
};

/// Exponential store scale parameter [mm].
pub const X6_BOUNDS: Bounds = Bounds {
    min: 1.0,
    max: 50.0,
};
