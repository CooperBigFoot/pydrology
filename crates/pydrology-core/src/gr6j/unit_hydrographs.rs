/// GR6J unit hydrograph functions.
///
/// Implements the S-curve based unit hydrographs (UH1 and UH2) used
/// for temporal distribution of effective rainfall.
///
/// UH1 is faster (length NH=20 days), UH2 is slower (length 2*NH=40 days).
use super::constants::{D, NH};

/// Compute UH1 S-curve value at position i.
fn ss1(i: f64, x4: f64) -> f64 {
    if i <= 0.0 {
        0.0
    } else if i < x4 {
        (i / x4).powf(D)
    } else {
        1.0
    }
}

/// Compute UH2 S-curve value at position i.
fn ss2(i: f64, x4: f64) -> f64 {
    if i <= 0.0 {
        0.0
    } else if i <= x4 {
        0.5 * (i / x4).powf(D)
    } else if i < 2.0 * x4 {
        1.0 - 0.5 * (2.0 - i / x4).powf(D)
    } else {
        1.0
    }
}

/// Compute unit hydrograph ordinates for UH1 and UH2.
///
/// The ordinates are derived from S-curve functions. UH1 has a faster response
/// (base time X4) while UH2 has a slower response (base time 2*X4).
///
/// Returns (uh1_ordinates, uh2_ordinates) as fixed-size arrays.
pub fn compute_uh_ordinates(x4: f64) -> ([f64; NH], [f64; 2 * NH]) {
    let mut uh1_ordinates = [0.0; NH];
    let mut uh2_ordinates = [0.0; 2 * NH];

    for i in 1..=NH {
        let fi = i as f64;
        uh1_ordinates[i - 1] = ss1(fi, x4) - ss1(fi - 1.0, x4);
    }

    for i in 1..=(2 * NH) {
        let fi = i as f64;
        uh2_ordinates[i - 1] = ss2(fi, x4) - ss2(fi - 1.0, x4);
    }

    (uh1_ordinates, uh2_ordinates)
}

/// Perform unit hydrograph convolution for one time step.
///
/// Updates the unit hydrograph state array by shifting values and adding
/// the contribution of new rainfall input weighted by the UH ordinates.
///
/// Returns the UH output for this time step (the first element of states before update).
/// Modifies `states` in place.
pub fn convolve_uh(states: &mut [f64], ordinates: &[f64], input: f64) -> f64 {
    let output = states[0];
    let n = states.len();

    for k in 0..n - 1 {
        states[k] = states[k + 1] + ordinates[k] * input;
    }
    states[n - 1] = ordinates[n - 1] * input;

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected} ± {tol}, got {actual}"
        );
    }

    // -- S-curves --

    #[test]
    fn ss1_zero_at_origin() {
        assert_approx(ss1(0.0, 1.7), 0.0, 1e-10);
    }

    #[test]
    fn ss1_one_at_x4() {
        assert_approx(ss1(1.7, 1.7), 1.0, 1e-10);
    }

    #[test]
    fn ss1_one_beyond_x4() {
        assert_approx(ss1(5.0, 1.7), 1.0, 1e-10);
    }

    #[test]
    fn ss1_monotonically_increasing() {
        let prev = ss1(0.5, 1.7);
        let curr = ss1(1.0, 1.7);
        assert!(curr > prev);
    }

    #[test]
    fn ss2_zero_at_origin() {
        assert_approx(ss2(0.0, 1.7), 0.0, 1e-10);
    }

    #[test]
    fn ss2_half_at_x4() {
        assert_approx(ss2(1.7, 1.7), 0.5, 1e-10);
    }

    #[test]
    fn ss2_one_at_2x4() {
        assert_approx(ss2(3.4, 1.7), 1.0, 1e-10);
    }

    #[test]
    fn ss2_one_beyond_2x4() {
        assert_approx(ss2(10.0, 1.7), 1.0, 1e-10);
    }

    // -- UH ordinates --

    #[test]
    fn uh1_ordinates_sum_to_one() {
        let (uh1, _uh2) = compute_uh_ordinates(1.7);
        let sum: f64 = uh1.iter().sum();
        assert_approx(sum, 1.0, 1e-10);
    }

    #[test]
    fn uh2_ordinates_sum_to_one() {
        let (_uh1, uh2) = compute_uh_ordinates(1.7);
        let sum: f64 = uh2.iter().sum();
        assert_approx(sum, 1.0, 1e-10);
    }

    #[test]
    fn uh1_ordinates_non_negative() {
        let (uh1, _uh2) = compute_uh_ordinates(1.7);
        assert!(uh1.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn uh2_ordinates_non_negative() {
        let (_uh1, uh2) = compute_uh_ordinates(1.7);
        assert!(uh2.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn uh_ordinates_different_x4() {
        let (uh1_small, _) = compute_uh_ordinates(0.5);
        let (uh1_large, _) = compute_uh_ordinates(5.0);
        // Larger x4 should spread response more
        assert!(uh1_small[0] > uh1_large[0]);
    }

    // -- Convolution --

    #[test]
    fn convolve_returns_first_element() {
        let mut states = [10.0, 5.0, 2.0, 0.0];
        let ordinates = [0.4, 0.3, 0.2, 0.1];
        let output = convolve_uh(&mut states, &ordinates, 0.0);
        assert_approx(output, 10.0, 1e-10);
    }

    #[test]
    fn convolve_shifts_states() {
        let mut states = [10.0, 5.0, 2.0, 0.0];
        let ordinates = [0.4, 0.3, 0.2, 0.1];
        convolve_uh(&mut states, &ordinates, 0.0);
        // After convolution with zero input, states shift left
        assert_approx(states[0], 5.0, 1e-10);
        assert_approx(states[1], 2.0, 1e-10);
        assert_approx(states[2], 0.0, 1e-10);
        assert_approx(states[3], 0.0, 1e-10);
    }

    #[test]
    fn convolve_adds_input_contribution() {
        let mut states = [0.0; 4];
        let ordinates = [0.4, 0.3, 0.2, 0.1];
        let input = 10.0;
        convolve_uh(&mut states, &ordinates, input);
        assert_approx(states[0], 4.0, 1e-10);
        assert_approx(states[1], 3.0, 1e-10);
        assert_approx(states[2], 2.0, 1e-10);
        assert_approx(states[3], 1.0, 1e-10);
    }
}
