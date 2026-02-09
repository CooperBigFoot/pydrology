/// CemaNeige calibrated parameters.
///
/// Two parameters that define the snow model behavior.
use super::constants::{N_PARAMS, PARAM_BOUNDS, PARAM_NAMES};
use crate::traits::ModelParams;

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    /// Thermal state weighting coefficient [-]. Range [0, 1].
    pub ctg: f64,
    /// Degree-day melt factor [mm/C/day]. Range [0, 10].
    pub kf: f64,
}

impl Parameters {
    pub fn new(ctg: f64, kf: f64) -> Self {
        Self { ctg, kf }
    }

    pub fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != N_PARAMS {
            return Err(format!("expected {} parameters, got {}", N_PARAMS, arr.len()));
        }
        Ok(Self::new(arr[0], arr[1]))
    }

    pub fn to_array(&self) -> [f64; N_PARAMS] {
        [self.ctg, self.kf]
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
        Ok(Self::new(arr[0], arr[1]))
    }

    fn to_array(&self) -> Vec<f64> {
        vec![self.ctg, self.kf]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(0.97, 2.5);
        assert_eq!(p.ctg, 0.97);
        assert_eq!(p.kf, 2.5);
    }

    #[test]
    fn boundary_values_are_valid() {
        let _ = Parameters::new(0.0, 0.0);
        let _ = Parameters::new(1.0, 10.0);
    }

    #[test]
    fn from_array_roundtrip() {
        let p = Parameters::new(0.97, 2.5);
        let arr = p.to_array();
        let p2 = Parameters::from_array(&arr).unwrap();
        assert_eq!(p.ctg, p2.ctg);
        assert_eq!(p.kf, p2.kf);
    }

    #[test]
    fn model_params_roundtrip() {
        use crate::traits::ModelParams;

        let p = Parameters::new(0.97, 2.5);
        let arr = <Parameters as ModelParams>::to_array(&p);
        let p2 = <Parameters as ModelParams>::from_array(&arr).unwrap();
        assert_eq!(p.ctg, p2.ctg);
        assert_eq!(p.kf, p2.kf);
    }
}
