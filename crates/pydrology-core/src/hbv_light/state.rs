/// HBV-Light model state variables.
///
/// Mutable state that evolves during simulation. Contains per-zone states
/// for snow and soil, lumped groundwater stores, and routing buffer.
use smallvec::SmallVec;

use super::constants::{LUMPED_STATE_SIZE, ROUTING_BUFFER_SIZE, ZONE_STATE_SIZE};
use super::params::Parameters;
use crate::traits::ModelState;

#[derive(Debug, Clone)]
pub struct State {
    /// Per-zone states: each inner array is [snow_pack, liquid_water, soil_moisture].
    pub zone_states: SmallVec<[[f64; ZONE_STATE_SIZE]; 4]>,
    /// Upper groundwater zone storage [mm].
    pub upper_zone: f64,
    /// Lower groundwater zone storage [mm].
    pub lower_zone: f64,
    /// Buffer for triangular unit hydrograph convolution.
    pub routing_buffer: [f64; ROUTING_BUFFER_SIZE],
}

impl State {
    /// Create initial state from parameters.
    ///
    /// Snow and liquid water start at zero, soil moisture at 50% FC,
    /// groundwater stores at zero, routing buffer at zero.
    pub fn initialize(params: &Parameters, n_zones: usize) -> Self {
        let mut zone_states: SmallVec<[[f64; ZONE_STATE_SIZE]; 4]> =
            smallvec::smallvec![[0.0f64; ZONE_STATE_SIZE]; n_zones];
        for zs in &mut zone_states {
            zs[2] = 0.5 * params.fc; // soil moisture at 50% FC
        }
        Self {
            zone_states,
            upper_zone: 0.0,
            lower_zone: 0.0,
            routing_buffer: [0.0; ROUTING_BUFFER_SIZE],
        }
    }

    /// Number of elevation zones.
    pub fn n_zones(&self) -> usize {
        self.zone_states.len()
    }

    /// Serialize state to a flat array.
    ///
    /// Layout: [zone_states.flatten(), upper_zone, lower_zone, routing_buffer]
    pub fn to_array(&self) -> Vec<f64> {
        let n_zones = self.n_zones();
        let size = n_zones * ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE;
        let mut arr = Vec::with_capacity(size);

        for zs in &self.zone_states {
            arr.extend_from_slice(zs);
        }
        arr.push(self.upper_zone);
        arr.push(self.lower_zone);
        arr.extend_from_slice(&self.routing_buffer);

        arr
    }

    /// Deserialize state from a flat array.
    pub fn from_array(arr: &[f64], n_zones: usize) -> Self {
        let expected = n_zones * ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE;
        assert_eq!(
            arr.len(),
            expected,
            "state array length {} does not match expected {} for {} zones",
            arr.len(),
            expected,
            n_zones
        );

        let mut zone_states: SmallVec<[[f64; ZONE_STATE_SIZE]; 4]> =
            SmallVec::with_capacity(n_zones);
        for z in 0..n_zones {
            let base = z * ZONE_STATE_SIZE;
            zone_states.push([arr[base], arr[base + 1], arr[base + 2]]);
        }

        let suz_idx = n_zones * ZONE_STATE_SIZE;
        let slz_idx = suz_idx + 1;
        let buf_start = slz_idx + 1;

        let mut routing_buffer = [0.0; ROUTING_BUFFER_SIZE];
        routing_buffer.copy_from_slice(&arr[buf_start..buf_start + ROUTING_BUFFER_SIZE]);

        Self {
            zone_states,
            upper_zone: arr[suz_idx],
            lower_zone: arr[slz_idx],
            routing_buffer,
        }
    }
}

impl ModelState for State {
    fn to_vec(&self) -> Vec<f64> {
        self.to_array()
    }

    fn from_slice(arr: &[f64]) -> Result<Self, String> {
        let fixed_part = LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE;
        if arr.len() < fixed_part {
            return Err(format!(
                "state array too short: {} < minimum {}",
                arr.len(),
                fixed_part
            ));
        }
        let zone_part = arr.len() - fixed_part;
        if zone_part % ZONE_STATE_SIZE != 0 {
            return Err(format!(
                "state array length {} does not align to zone size {} (fixed part {})",
                arr.len(),
                ZONE_STATE_SIZE,
                fixed_part
            ));
        }
        let n_zones = zone_part / ZONE_STATE_SIZE;
        if n_zones == 0 {
            return Err("state array implies 0 zones".to_string());
        }

        let mut zone_states: SmallVec<[[f64; ZONE_STATE_SIZE]; 4]> =
            SmallVec::with_capacity(n_zones);
        for z in 0..n_zones {
            let base = z * ZONE_STATE_SIZE;
            zone_states.push([arr[base], arr[base + 1], arr[base + 2]]);
        }

        let suz_idx = n_zones * ZONE_STATE_SIZE;
        let slz_idx = suz_idx + 1;
        let buf_start = slz_idx + 1;

        let mut routing_buffer = [0.0; ROUTING_BUFFER_SIZE];
        routing_buffer.copy_from_slice(&arr[buf_start..buf_start + ROUTING_BUFFER_SIZE]);

        Ok(Self {
            zone_states,
            upper_zone: arr[suz_idx],
            lower_zone: arr[slz_idx],
            routing_buffer,
        })
    }

    fn array_len(&self) -> usize {
        self.n_zones() * ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5)
            .unwrap()
    }

    #[test]
    fn initialize_single_zone() {
        let p = test_params();
        let s = State::initialize(&p, 1);
        assert_eq!(s.n_zones(), 1);
        assert_eq!(s.zone_states[0][0], 0.0); // SP
        assert_eq!(s.zone_states[0][1], 0.0); // LW
        assert_eq!(s.zone_states[0][2], 125.0); // 50% of FC=250
        assert_eq!(s.upper_zone, 0.0);
        assert_eq!(s.lower_zone, 0.0);
        assert_eq!(s.routing_buffer, [0.0; 7]);
    }

    #[test]
    fn initialize_multi_zone() {
        let p = test_params();
        let s = State::initialize(&p, 3);
        assert_eq!(s.n_zones(), 3);
        for z in 0..3 {
            assert_eq!(s.zone_states[z][2], 125.0);
        }
    }

    #[test]
    fn to_array_from_array_roundtrip() {
        let p = test_params();
        let mut s = State::initialize(&p, 2);
        s.upper_zone = 10.0;
        s.lower_zone = 5.0;
        s.routing_buffer[0] = 3.0;

        let arr = s.to_array();
        let s2 = State::from_array(&arr, 2);

        assert_eq!(s2.n_zones(), 2);
        assert_eq!(s2.upper_zone, 10.0);
        assert_eq!(s2.lower_zone, 5.0);
        assert_eq!(s2.routing_buffer[0], 3.0);
        assert_eq!(s2.zone_states[0][2], 125.0);
    }

    #[test]
    fn model_state_roundtrip() {
        let p = test_params();
        let mut s = State::initialize(&p, 2);
        s.upper_zone = 10.0;
        let v = s.to_vec();
        let s2 = State::from_slice(&v).unwrap();
        assert_eq!(s2.n_zones(), 2);
        assert_eq!(s2.upper_zone, 10.0);
    }

    #[test]
    fn model_state_wrong_length() {
        // Not enough for even fixed part (9)
        assert!(State::from_slice(&[1.0, 2.0]).is_err());
        // Misaligned: 11 - 9 = 2, 2 % 3 != 0
        assert!(State::from_slice(&[1.0; 11]).is_err());
    }
}
