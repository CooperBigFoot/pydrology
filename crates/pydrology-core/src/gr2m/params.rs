/// GR2M calibrated parameters.
///
/// Two parameters that define model behavior. The struct is immutable
/// by default in Rust — no `frozen=True` needed like Python.
///
/// - `x1`: Production store capacity [mm]
/// - `x2`: Groundwater exchange coefficient [-]
use super::constants::{PARAM_BOUNDS, PARAM_NAMES};
use crate::traits::ModelParams;

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub x1: f64,
    pub x2: f64,
}

impl Parameters {
    /// Create new Parameters.
    pub fn new(x1: f64, x2: f64) -> Self {
        Self { x1, x2 }
    }
}

impl ModelParams for Parameters {
    const N_PARAMS: usize = 2;
    const PARAM_NAMES: &'static [&'static str] = PARAM_NAMES;
    const PARAM_BOUNDS: &'static [(f64, f64)] = PARAM_BOUNDS;

    fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != Self::N_PARAMS {
            return Err(format!(
                "expected {} parameters, got {}",
                Self::N_PARAMS,
                arr.len()
            ));
        }
        Ok(Self::new(arr[0], arr[1]))
    }

    fn to_array(&self) -> Vec<f64> {
        vec![self.x1, self.x2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(500.0, 1.0);
        assert_eq!(p.x1, 500.0);
        assert_eq!(p.x2, 1.0);
    }

    #[test]
    fn boundary_values_are_valid() {
        let _ = Parameters::new(1.0, 0.2);
        let _ = Parameters::new(2500.0, 2.0);
    }

    use crate::traits::ModelParams;

    #[test]
    fn from_array_valid() {
        let p = Parameters::from_array(&[500.0, 1.0]).unwrap();
        assert_eq!(p.x1, 500.0);
        assert_eq!(p.x2, 1.0);
    }

    #[test]
    fn from_array_wrong_length() {
        assert!(Parameters::from_array(&[500.0]).is_err());
        assert!(Parameters::from_array(&[500.0, 1.0, 2.0]).is_err());
    }

    #[test]
    fn to_array_roundtrip() {
        let p = Parameters::new(500.0, 1.0);
        let arr = p.to_array();
        let p2 = Parameters::from_array(&arr).unwrap();
        assert_eq!(p.x1, p2.x1);
        assert_eq!(p.x2, p2.x2);
    }
}
