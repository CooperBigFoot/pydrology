/// GR2M calibrated parameters.
///
/// Two parameters that define model behavior. The struct is immutable
/// by default in Rust — no `frozen=True` needed like Python.
///
/// - `x1`: Production store capacity [mm]
/// - `x2`: Groundwater exchange coefficient [-]
use super::constants::{X1_BOUNDS, X2_BOUNDS};

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub x1: f64,
    pub x2: f64,
}

impl Parameters {
    /// Create new Parameters, returning an error if out of bounds.
    pub fn new(x1: f64, x2: f64) -> Result<Self, String> {
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
        Ok(Self { x1, x2 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(500.0, 1.0).unwrap();
        assert_eq!(p.x1, 500.0);
        assert_eq!(p.x2, 1.0);
    }

    #[test]
    fn x1_too_low() {
        assert!(Parameters::new(0.5, 1.0).is_err());
    }

    #[test]
    fn x1_too_high() {
        assert!(Parameters::new(3000.0, 1.0).is_err());
    }

    #[test]
    fn x2_too_low() {
        assert!(Parameters::new(500.0, 0.1).is_err());
    }

    #[test]
    fn x2_too_high() {
        assert!(Parameters::new(500.0, 3.0).is_err());
    }

    #[test]
    fn boundary_values_are_valid() {
        assert!(Parameters::new(1.0, 0.2).is_ok());
        assert!(Parameters::new(2500.0, 2.0).is_ok());
    }
}
