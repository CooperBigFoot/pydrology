/// CemaNeige calibrated parameters.
///
/// Two parameters that define the snow model behavior.
use super::constants::N_PARAMS;

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    /// Thermal state weighting coefficient [-]. Range [0, 1].
    pub ctg: f64,
    /// Degree-day melt factor [mm/C/day]. Range [0, 10].
    pub kf: f64,
}

impl Parameters {
    pub fn new(ctg: f64, kf: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&ctg) {
            return Err(format!("ctg = {} is out of bounds [0, 1]", ctg));
        }
        if !(0.0..=10.0).contains(&kf) {
            return Err(format!("kf = {} is out of bounds [0, 10]", kf));
        }
        Ok(Self { ctg, kf })
    }

    pub fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != N_PARAMS {
            return Err(format!("expected {} parameters, got {}", N_PARAMS, arr.len()));
        }
        Self::new(arr[0], arr[1])
    }

    pub fn to_array(&self) -> [f64; N_PARAMS] {
        [self.ctg, self.kf]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parameters() {
        let p = Parameters::new(0.97, 2.5).unwrap();
        assert_eq!(p.ctg, 0.97);
        assert_eq!(p.kf, 2.5);
    }

    #[test]
    fn ctg_out_of_bounds() {
        assert!(Parameters::new(-0.1, 2.5).is_err());
        assert!(Parameters::new(1.1, 2.5).is_err());
    }

    #[test]
    fn kf_out_of_bounds() {
        assert!(Parameters::new(0.5, -0.1).is_err());
        assert!(Parameters::new(0.5, 10.1).is_err());
    }

    #[test]
    fn boundary_values_are_valid() {
        assert!(Parameters::new(0.0, 0.0).is_ok());
        assert!(Parameters::new(1.0, 10.0).is_ok());
    }

    #[test]
    fn from_array_roundtrip() {
        let p = Parameters::new(0.97, 2.5).unwrap();
        let arr = p.to_array();
        let p2 = Parameters::from_array(&arr).unwrap();
        assert_eq!(p.ctg, p2.ctg);
        assert_eq!(p.kf, p2.kf);
    }
}
