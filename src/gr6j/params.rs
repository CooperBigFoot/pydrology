/// GR6J calibrated parameters.
///
/// Six parameters that define model behavior:
/// - `x1`: Production store capacity [mm]
/// - `x2`: Intercatchment exchange coefficient [mm/day]
/// - `x3`: Routing store capacity [mm]
/// - `x4`: Unit hydrograph time constant [days]
/// - `x5`: Intercatchment exchange threshold [-]
/// - `x6`: Exponential store scale parameter [mm]
use super::constants::{X1_BOUNDS, X2_BOUNDS, X3_BOUNDS, X4_BOUNDS, X5_BOUNDS, X6_BOUNDS};

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
    pub x4: f64,
    pub x5: f64,
    pub x6: f64,
}

impl Parameters {
    /// Create new Parameters, returning an error if out of bounds.
    pub fn new(x1: f64, x2: f64, x3: f64, x4: f64, x5: f64, x6: f64) -> Result<Self, String> {
        if !(X1_BOUNDS.min..=X1_BOUNDS.max).contains(&x1) {
            return Err(format!(
                "x1 = {} is out of bounds [{}, {}]",
                x1, X1_BOUNDS.min, X1_BOUNDS.max
            ));
        }
        if !(X2_BOUNDS.min..=X2_BOUNDS.max).contains(&x2) {
            return Err(format!(
                "x2 = {} is out of bounds [{}, {}]",
                x2, X2_BOUNDS.min, X2_BOUNDS.max
            ));
        }
        if !(X3_BOUNDS.min..=X3_BOUNDS.max).contains(&x3) {
            return Err(format!(
                "x3 = {} is out of bounds [{}, {}]",
                x3, X3_BOUNDS.min, X3_BOUNDS.max
            ));
        }
        if !(X4_BOUNDS.min..=X4_BOUNDS.max).contains(&x4) {
            return Err(format!(
                "x4 = {} is out of bounds [{}, {}]",
                x4, X4_BOUNDS.min, X4_BOUNDS.max
            ));
        }
        if !(X5_BOUNDS.min..=X5_BOUNDS.max).contains(&x5) {
            return Err(format!(
                "x5 = {} is out of bounds [{}, {}]",
                x5, X5_BOUNDS.min, X5_BOUNDS.max
            ));
        }
        if !(X6_BOUNDS.min..=X6_BOUNDS.max).contains(&x6) {
            return Err(format!(
                "x6 = {} is out of bounds [{}, {}]",
                x6, X6_BOUNDS.min, X6_BOUNDS.max
            ));
        }
        Ok(Self {
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
        })
    }

    /// Create Parameters without bounds validation.
    ///
    /// Used by the PyO3 binding where bounds are enforced by the calibration
    /// layer, not the model core.
    pub fn new_unchecked(x1: f64, x2: f64, x3: f64, x4: f64, x5: f64, x6: f64) -> Self {
        Self {
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0).unwrap();
        assert_eq!(p.x1, 350.0);
        assert_eq!(p.x2, 0.0);
        assert_eq!(p.x3, 90.0);
        assert_eq!(p.x4, 1.7);
        assert_eq!(p.x5, 0.0);
        assert_eq!(p.x6, 5.0);
    }

    #[test]
    fn x1_too_low() {
        assert!(Parameters::new(0.5, 0.0, 90.0, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x1_too_high() {
        assert!(Parameters::new(3000.0, 0.0, 90.0, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x2_too_low() {
        assert!(Parameters::new(350.0, -6.0, 90.0, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x2_too_high() {
        assert!(Parameters::new(350.0, 6.0, 90.0, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x3_too_low() {
        assert!(Parameters::new(350.0, 0.0, 0.5, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x3_too_high() {
        assert!(Parameters::new(350.0, 0.0, 1500.0, 1.7, 0.0, 5.0).is_err());
    }

    #[test]
    fn x4_too_low() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 0.4, 0.0, 5.0).is_err());
    }

    #[test]
    fn x4_too_high() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 11.0, 0.0, 5.0).is_err());
    }

    #[test]
    fn x5_too_low() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 1.7, -5.0, 5.0).is_err());
    }

    #[test]
    fn x5_too_high() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 1.7, 5.0, 5.0).is_err());
    }

    #[test]
    fn x6_too_low() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 0.5).is_err());
    }

    #[test]
    fn x6_too_high() {
        assert!(Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 60.0).is_err());
    }

    #[test]
    fn boundary_values_are_valid() {
        assert!(Parameters::new(1.0, -5.0, 1.0, 0.5, -4.0, 1.0).is_ok());
        assert!(Parameters::new(2500.0, 5.0, 1000.0, 10.0, 4.0, 50.0).is_ok());
    }
}
