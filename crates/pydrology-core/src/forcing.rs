/// Temporal resolution of forcing data.
///
/// Mirrors pydrology's Resolution enum. Ordered from finest to coarsest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resolution {
    Hourly,
    Daily,
    Monthly,
    Annual,
}

impl Resolution {
    /// Average number of days per timestep.
    pub fn days_per_timestep(self) -> f64 {
        match self {
            Resolution::Hourly => 1.0 / 24.0,
            Resolution::Daily => 1.0,
            Resolution::Monthly => 30.4375,
            Resolution::Annual => 365.25,
        }
    }
}

/// Validated forcing data for the GR2M model.
///
/// All arrays must have the same length. NaN values are rejected.
/// This is the Rust equivalent of pydrology's Pydantic-validated ForcingData.
#[derive(Debug)]
pub struct ForcingData {
    pub precip: Vec<f64>,
    pub pet: Vec<f64>,
    pub resolution: Resolution,
}

impl ForcingData {
    /// Create new ForcingData with validation.
    ///
    /// Validates:
    /// - precip and pet have the same length
    /// - No NaN values in either array
    /// - Arrays are non-empty
    pub fn new(
        precip: Vec<f64>,
        pet: Vec<f64>,
        resolution: Resolution,
    ) -> Result<Self, String> {
        if precip.is_empty() {
            return Err("precip array is empty".to_string());
        }
        if precip.len() != pet.len() {
            return Err(format!(
                "precip length {} does not match pet length {}",
                precip.len(),
                pet.len()
            ));
        }
        if precip.iter().any(|v| v.is_nan()) {
            return Err("precip array contains NaN values".to_string());
        }
        if pet.iter().any(|v| v.is_nan()) {
            return Err("pet array contains NaN values".to_string());
        }
        Ok(Self {
            precip,
            pet,
            resolution,
        })
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.precip.len()
    }

    /// Returns `true` if there are no timesteps.
    pub fn is_empty(&self) -> bool {
        self.precip.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Resolution --

    #[test]
    fn resolution_ordering() {
        assert!(Resolution::Hourly < Resolution::Daily);
        assert!(Resolution::Daily < Resolution::Monthly);
        assert!(Resolution::Monthly < Resolution::Annual);
    }

    #[test]
    fn resolution_days_per_timestep() {
        assert_eq!(Resolution::Daily.days_per_timestep(), 1.0);
        assert_eq!(Resolution::Monthly.days_per_timestep(), 30.4375);
        assert_eq!(Resolution::Annual.days_per_timestep(), 365.25);
    }

    // -- ForcingData valid construction --

    #[test]
    fn valid_forcing_data() {
        let fd = ForcingData::new(
            vec![80.0, 70.0, 60.0],
            vec![20.0, 25.0, 30.0],
            Resolution::Monthly,
        );
        assert!(fd.is_ok());
        assert_eq!(fd.unwrap().len(), 3);
    }

    // -- Validation: length mismatch --

    #[test]
    fn rejects_length_mismatch() {
        let fd = ForcingData::new(
            vec![80.0, 70.0],
            vec![20.0],
            Resolution::Monthly,
        );
        assert!(fd.is_err());
        assert!(fd.unwrap_err().contains("does not match"));
    }

    // -- Validation: empty arrays --

    #[test]
    fn rejects_empty_arrays() {
        let fd = ForcingData::new(vec![], vec![], Resolution::Monthly);
        assert!(fd.is_err());
        assert!(fd.unwrap_err().contains("empty"));
    }

    // -- Validation: NaN rejection --

    #[test]
    fn rejects_nan_in_precip() {
        let fd = ForcingData::new(
            vec![10.0, f64::NAN, 5.0],
            vec![3.0, 4.0, 5.0],
            Resolution::Monthly,
        );
        assert!(fd.is_err());
        assert!(fd.unwrap_err().contains("NaN"));
    }

    #[test]
    fn rejects_nan_in_pet() {
        let fd = ForcingData::new(
            vec![10.0, 5.0, 0.0],
            vec![3.0, f64::NAN, 5.0],
            Resolution::Monthly,
        );
        assert!(fd.is_err());
        assert!(fd.unwrap_err().contains("NaN"));
    }
}
