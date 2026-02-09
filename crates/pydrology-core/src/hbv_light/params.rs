/// HBV-Light calibrated parameters.
///
/// 14 parameters that define model behavior.
use super::constants::{ALL_BOUNDS, N_PARAMS, PARAM_BOUNDS, PARAM_NAMES};
use crate::traits::ModelParams;

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub tt: f64,
    pub cfmax: f64,
    pub sfcf: f64,
    pub cwh: f64,
    pub cfr: f64,
    pub fc: f64,
    pub lp: f64,
    pub beta: f64,
    pub k0: f64,
    pub k1: f64,
    pub k2: f64,
    pub perc: f64,
    pub uzl: f64,
    pub maxbas: f64,
}

impl Parameters {
    /// Create new Parameters, returning an error if any value is out of bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tt: f64,
        cfmax: f64,
        sfcf: f64,
        cwh: f64,
        cfr: f64,
        fc: f64,
        lp: f64,
        beta: f64,
        k0: f64,
        k1: f64,
        k2: f64,
        perc: f64,
        uzl: f64,
        maxbas: f64,
    ) -> Result<Self, String> {
        let values = [
            tt, cfmax, sfcf, cwh, cfr, fc, lp, beta, k0, k1, k2, perc, uzl, maxbas,
        ];
        for (i, &val) in values.iter().enumerate() {
            let bounds = ALL_BOUNDS[i];
            if !(bounds.min..=bounds.max).contains(&val) {
                return Err(format!(
                    "{} = {} is out of bounds [{}, {}]",
                    PARAM_NAMES[i], val, bounds.min, bounds.max
                ));
            }
        }
        Ok(Self {
            tt,
            cfmax,
            sfcf,
            cwh,
            cfr,
            fc,
            lp,
            beta,
            k0,
            k1,
            k2,
            perc,
            uzl,
            maxbas,
        })
    }

    /// Create Parameters from a 14-element slice.
    pub fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != N_PARAMS {
            return Err(format!(
                "expected {} parameters, got {}",
                N_PARAMS,
                arr.len()
            ));
        }
        Self::new(
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
            arr[10], arr[11], arr[12], arr[13],
        )
    }

    /// Convert to a 14-element array.
    pub fn to_array(&self) -> [f64; N_PARAMS] {
        [
            self.tt, self.cfmax, self.sfcf, self.cwh, self.cfr, self.fc, self.lp, self.beta,
            self.k0, self.k1, self.k2, self.perc, self.uzl, self.maxbas,
        ]
    }
}

impl ModelParams for Parameters {
    const N_PARAMS: usize = N_PARAMS;
    const PARAM_NAMES: &'static [&'static str] = PARAM_NAMES;
    const PARAM_BOUNDS: &'static [(f64, f64)] = PARAM_BOUNDS;

    fn from_array(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != N_PARAMS {
            return Err(format!(
                "expected {} parameters, got {}",
                N_PARAMS,
                arr.len()
            ));
        }
        Self::new(
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
            arr[10], arr[11], arr[12], arr[13],
        )
    }

    fn to_array(&self) -> Vec<f64> {
        vec![
            self.tt, self.cfmax, self.sfcf, self.cwh, self.cfr, self.fc, self.lp, self.beta,
            self.k0, self.k1, self.k2, self.perc, self.uzl, self.maxbas,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_params() -> Parameters {
        Parameters::new(0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5)
            .unwrap()
    }

    #[test]
    fn valid_parameters() {
        let p = valid_params();
        assert_eq!(p.tt, 0.0);
        assert_eq!(p.fc, 250.0);
        assert_eq!(p.maxbas, 2.5);
    }

    #[test]
    fn from_array_roundtrip() {
        let p = valid_params();
        let arr = p.to_array();
        let p2 = Parameters::from_array(&arr).unwrap();
        assert_eq!(p.tt, p2.tt);
        assert_eq!(p.maxbas, p2.maxbas);
    }

    #[test]
    fn from_array_wrong_length() {
        assert!(Parameters::from_array(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn tt_out_of_bounds() {
        assert!(Parameters::new(
            -3.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5
        )
        .is_err());
    }

    #[test]
    fn fc_out_of_bounds() {
        assert!(Parameters::new(
            0.0, 3.0, 1.0, 0.1, 0.05, 10.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5
        )
        .is_err());
    }

    #[test]
    fn maxbas_out_of_bounds() {
        assert!(Parameters::new(
            0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 10.0
        )
        .is_err());
    }

    #[test]
    fn boundary_values_are_valid() {
        assert!(Parameters::new(
            -2.5, 0.5, 0.4, 0.0, 0.0, 50.0, 0.3, 1.0, 0.05, 0.01, 0.001, 0.0, 0.0, 1.0
        )
        .is_ok());
        assert!(Parameters::new(
            2.5, 10.0, 1.4, 0.2, 0.2, 700.0, 1.0, 6.0, 0.99, 0.5, 0.2, 6.0, 100.0, 7.0
        )
        .is_ok());
    }

    #[test]
    fn model_params_roundtrip() {
        use crate::traits::ModelParams;
        let p = valid_params();
        let arr = <Parameters as ModelParams>::to_array(&p);
        let p2 = <Parameters as ModelParams>::from_array(&arr).unwrap();
        assert_eq!(p.tt, p2.tt);
        assert_eq!(p.maxbas, p2.maxbas);
    }
}
