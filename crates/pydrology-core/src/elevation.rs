//! Shared elevation extrapolation utilities.
//!
//! Used by HBV-Light and CemaNeige for multi-zone/multi-layer runs.

/// Default temperature gradient [C/100m].
pub const GRAD_T_DEFAULT: f64 = 0.6;

/// Default precipitation gradient [m^-1].
pub const GRAD_P_DEFAULT: f64 = 0.00041;

/// Elevation cap for precipitation extrapolation [m].
pub const ELEV_CAP_PRECIP: f64 = 4000.0;

/// Extrapolate temperature to a different elevation using lapse rate.
#[inline]
pub fn extrapolate_temp(temp: f64, input_elev: f64, target_elev: f64, gradient: f64) -> f64 {
    temp - gradient * (target_elev - input_elev) / 100.0
}

/// Extrapolate precipitation to a different elevation using exponential gradient.
#[inline]
pub fn extrapolate_precip(precip: f64, input_elev: f64, target_elev: f64, gradient: f64) -> f64 {
    let eff_in = input_elev.min(ELEV_CAP_PRECIP);
    let eff_target = target_elev.min(ELEV_CAP_PRECIP);
    precip * (gradient * (eff_target - eff_in)).exp()
}

/// Extrapolate precipitation with a custom elevation cap.
#[inline]
pub fn extrapolate_precip_with_cap(
    precip: f64,
    input_elev: f64,
    target_elev: f64,
    gradient: f64,
    cap: f64,
) -> f64 {
    let eff_in = input_elev.min(cap);
    let eff_target = target_elev.min(cap);
    precip * (gradient * (eff_target - eff_in)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_same_elevation() {
        assert!((extrapolate_temp(10.0, 500.0, 500.0, 0.6) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn temp_higher_is_colder() {
        let t = extrapolate_temp(10.0, 500.0, 1000.0, 0.6);
        assert!((t - 7.0).abs() < 1e-10);
    }

    #[test]
    fn precip_same_elevation() {
        assert!((extrapolate_precip(10.0, 500.0, 500.0, 0.00041) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn precip_with_cap_same_elevation() {
        assert!(
            (extrapolate_precip_with_cap(10.0, 500.0, 500.0, 0.00041, 4000.0) - 10.0).abs()
                < 1e-10
        );
    }
}
