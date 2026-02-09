/// HBV-Light numerical constants and parameter bounds.
///
/// Centralises all fixed values used throughout the HBV-Light model.
use crate::forcing::Resolution;

// -- Model contract constants --

/// Parameter names in canonical order.
pub const PARAM_NAMES: &[&str] = &[
    "tt", "cfmax", "sfcf", "cwh", "cfr", "fc", "lp", "beta", "k0", "k1", "k2", "perc", "uzl",
    "maxbas",
];

/// Number of parameters.
pub const N_PARAMS: usize = 14;

/// State size constants.
pub const ZONE_STATE_SIZE: usize = 3; // SP, LW, SM per zone
pub const LUMPED_STATE_SIZE: usize = 2; // SUZ, SLZ
pub const ROUTING_BUFFER_SIZE: usize = 7;

/// Supported temporal resolutions for forcing data.
pub const SUPPORTED_RESOLUTIONS: &[Resolution] = &[Resolution::Daily];

// -- Parameter bounds --

/// Parameter bounds for calibration.
pub struct Bounds {
    pub min: f64,
    pub max: f64,
}

/// Threshold temperature [C].
pub const TT_BOUNDS: Bounds = Bounds {
    min: -2.5,
    max: 2.5,
};

/// Degree-day factor [mm/C/d].
pub const CFMAX_BOUNDS: Bounds = Bounds {
    min: 0.5,
    max: 10.0,
};

/// Snowfall correction factor [-].
pub const SFCF_BOUNDS: Bounds = Bounds {
    min: 0.4,
    max: 1.4,
};

/// Water holding capacity of snow [-].
pub const CWH_BOUNDS: Bounds = Bounds {
    min: 0.0,
    max: 0.2,
};

/// Refreezing coefficient [-].
pub const CFR_BOUNDS: Bounds = Bounds {
    min: 0.0,
    max: 0.2,
};

/// Field capacity [mm].
pub const FC_BOUNDS: Bounds = Bounds {
    min: 50.0,
    max: 700.0,
};

/// Limit for potential ET [-].
pub const LP_BOUNDS: Bounds = Bounds {
    min: 0.3,
    max: 1.0,
};

/// Shape coefficient [-].
pub const BETA_BOUNDS: Bounds = Bounds {
    min: 1.0,
    max: 6.0,
};

/// Surface flow recession [1/d].
pub const K0_BOUNDS: Bounds = Bounds {
    min: 0.05,
    max: 0.99,
};

/// Interflow recession [1/d].
pub const K1_BOUNDS: Bounds = Bounds {
    min: 0.01,
    max: 0.5,
};

/// Baseflow recession [1/d].
pub const K2_BOUNDS: Bounds = Bounds {
    min: 0.001,
    max: 0.2,
};

/// Maximum percolation rate [mm/d].
pub const PERC_BOUNDS: Bounds = Bounds {
    min: 0.0,
    max: 6.0,
};

/// Upper zone threshold [mm].
pub const UZL_BOUNDS: Bounds = Bounds {
    min: 0.0,
    max: 100.0,
};

/// Routing time [d].
pub const MAXBAS_BOUNDS: Bounds = Bounds {
    min: 1.0,
    max: 7.0,
};

/// All bounds in parameter order.
pub const ALL_BOUNDS: [&Bounds; N_PARAMS] = [
    &TT_BOUNDS,
    &CFMAX_BOUNDS,
    &SFCF_BOUNDS,
    &CWH_BOUNDS,
    &CFR_BOUNDS,
    &FC_BOUNDS,
    &LP_BOUNDS,
    &BETA_BOUNDS,
    &K0_BOUNDS,
    &K1_BOUNDS,
    &K2_BOUNDS,
    &PERC_BOUNDS,
    &UZL_BOUNDS,
    &MAXBAS_BOUNDS,
];

// -- Elevation extrapolation defaults --

/// Default temperature gradient [C/100m].
pub const GRAD_T_DEFAULT: f64 = 0.6;

/// Default precipitation gradient [m^-1].
pub const GRAD_P_DEFAULT: f64 = 0.00041;

/// Elevation cap for precipitation extrapolation [m].
pub const ELEV_CAP_PRECIP: f64 = 4000.0;
