/// GR6J calibrated parameters.
///
/// Six parameters that define model behavior:
/// - `x1`: Production store capacity [mm]
/// - `x2`: Intercatchment exchange coefficient [mm/day]
/// - `x3`: Routing store capacity [mm]
/// - `x4`: Unit hydrograph time constant [days]
/// - `x5`: Intercatchment exchange threshold [-]
/// - `x6`: Exponential store scale parameter [mm]
use super::constants::{N_PARAMS, PARAM_BOUNDS, PARAM_NAMES};
use crate::traits::ModelParams;

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
    /// Create new Parameters.
    pub fn new(x1: f64, x2: f64, x3: f64, x4: f64, x5: f64, x6: f64) -> Self {
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

impl ModelParams for Parameters {
    const N_PARAMS: usize = N_PARAMS;
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
        Ok(Self::new(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]))
    }

    fn to_array(&self) -> Vec<f64> {
        vec![self.x1, self.x2, self.x3, self.x4, self.x5, self.x6]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelParams;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0);
        assert_eq!(p.x1, 350.0);
        assert_eq!(p.x2, 0.0);
        assert_eq!(p.x3, 90.0);
        assert_eq!(p.x4, 1.7);
        assert_eq!(p.x5, 0.0);
        assert_eq!(p.x6, 5.0);
    }

    #[test]
    fn boundary_values_are_valid() {
        let _ = Parameters::new(1.0, -5.0, 1.0, 0.5, -4.0, 1.0);
        let _ = Parameters::new(2500.0, 5.0, 1000.0, 10.0, 4.0, 50.0);
    }

    #[test]
    fn from_array_valid() {
        let p =
            <Parameters as ModelParams>::from_array(&[350.0, 0.0, 90.0, 1.7, 0.0, 5.0]).unwrap();
        assert_eq!(p.x1, 350.0);
        assert_eq!(p.x6, 5.0);
    }

    #[test]
    fn from_array_wrong_length() {
        assert!(<Parameters as ModelParams>::from_array(&[350.0]).is_err());
        assert!(
            <Parameters as ModelParams>::from_array(&[350.0, 0.0, 90.0, 1.7, 0.0, 5.0, 1.0])
                .is_err()
        );
    }

    #[test]
    fn to_array_roundtrip() {
        let p = Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0);
        let arr = p.to_array();
        let p2 = <Parameters as ModelParams>::from_array(&arr).unwrap();
        assert_eq!(p.x1, p2.x1);
        assert_eq!(p.x6, p2.x6);
    }
}
