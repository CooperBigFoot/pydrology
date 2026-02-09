//! Hydrological metrics for calibration objectives.
//!
//! All metrics take observed and simulated slices and return a scalar score.

/// Nash-Sutcliffe Efficiency. Range: (-inf, 1], 1 = perfect.
pub fn nse(observed: &[f64], simulated: &[f64]) -> f64 {
    let n = observed.len();
    let mean_obs: f64 = observed.iter().sum::<f64>() / n as f64;
    let numerator: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    let denominator: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();
    if denominator == 0.0 {
        return f64::NEG_INFINITY;
    }
    1.0 - numerator / denominator
}

/// Log-transformed NSE. Uses log(x + 0.01) to avoid log(0).
pub fn log_nse(observed: &[f64], simulated: &[f64]) -> f64 {
    let log_obs: Vec<f64> = observed.iter().map(|o| (o + 0.01).ln()).collect();
    let log_sim: Vec<f64> = simulated.iter().map(|s| (s + 0.01).ln()).collect();
    nse(&log_obs, &log_sim)
}

/// Kling-Gupta Efficiency. Range: (-inf, 1], 1 = perfect.
pub fn kge(observed: &[f64], simulated: &[f64]) -> f64 {
    let n = observed.len() as f64;
    let mean_o = observed.iter().sum::<f64>() / n;
    let mean_s = simulated.iter().sum::<f64>() / n;
    let std_o = (observed
        .iter()
        .map(|o| (o - mean_o).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let std_s = (simulated
        .iter()
        .map(|s| (s - mean_s).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    let r = if std_o == 0.0 || std_s == 0.0 {
        0.0
    } else {
        observed
            .iter()
            .zip(simulated)
            .map(|(o, s)| (o - mean_o) * (s - mean_s))
            .sum::<f64>()
            / (n * std_o * std_s)
    };
    let alpha = if std_o == 0.0 { 0.0 } else { std_s / std_o };
    let beta = if mean_o == 0.0 { 0.0 } else { mean_s / mean_o };

    1.0 - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt()
}

/// Percent Bias. Optimal = 0. Positive = overestimation.
pub fn pbias(observed: &[f64], simulated: &[f64]) -> f64 {
    let sum_obs: f64 = observed.iter().sum();
    if sum_obs == 0.0 {
        return f64::INFINITY;
    }
    let diff_sum: f64 = simulated
        .iter()
        .zip(observed)
        .map(|(s, o)| s - o)
        .sum();
    100.0 * diff_sum / sum_obs
}

/// Root Mean Square Error. Range: [0, inf), 0 = perfect.
pub fn rmse(observed: &[f64], simulated: &[f64]) -> f64 {
    let n = observed.len() as f64;
    let mse: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

/// Mean Absolute Error. Range: [0, inf), 0 = perfect.
pub fn mae(observed: &[f64], simulated: &[f64]) -> f64 {
    let n = observed.len() as f64;
    observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).abs())
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- NSE tests ---

    #[test]
    fn nse_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(nse(&obs, &sim), 1.0);
    }

    #[test]
    fn nse_mean_simulation_gives_zero() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let sim = [mean; 5];
        assert_relative_eq!(nse(&obs, &sim), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn nse_constant_observed_returns_neg_inf() {
        let obs = [5.0; 5];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(nse(&obs, &sim), f64::NEG_INFINITY);
    }

    #[test]
    fn nse_poor_simulation_negative() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(nse(&obs, &sim) < 0.0);
    }

    #[test]
    fn nse_known_value() {
        // obs = [1,2,3,4,5], sim = [1.1, 2.2, 2.8, 4.1, 4.9]
        // mean_obs = 3.0
        // num = (0.1^2 + 0.2^2 + 0.2^2 + 0.1^2 + 0.1^2) = 0.01+0.04+0.04+0.01+0.01 = 0.11
        // den = (4+1+0+1+4) = 10
        // NSE = 1 - 0.11/10 = 0.989
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.1, 2.2, 2.8, 4.1, 4.9];
        assert_relative_eq!(nse(&obs, &sim), 0.989, epsilon = 1e-10);
    }

    // --- Log NSE tests ---

    #[test]
    fn log_nse_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(log_nse(&obs, &sim), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn log_nse_constant_observed_returns_neg_inf() {
        let obs = [1.0; 5];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(log_nse(&obs, &sim), f64::NEG_INFINITY);
    }

    #[test]
    fn log_nse_handles_zeros() {
        let obs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let sim = [0.0, 1.0, 2.0, 3.0, 4.0];
        let result = log_nse(&obs, &sim);
        assert!(result.is_finite());
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    // --- KGE tests ---

    #[test]
    fn kge_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(kge(&obs, &sim), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn kge_bias_reduces_score() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(kge(&obs, &sim) < 1.0);
    }

    #[test]
    fn kge_variability_reduces_score() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [2.0, 2.5, 3.0, 3.5, 4.0];
        assert!(kge(&obs, &sim) < 1.0);
    }

    #[test]
    fn kge_zero_variance_observed() {
        let obs = [3.0; 5];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kge(&obs, &sim);
        assert!(result.is_finite());
    }

    // --- PBIAS tests ---

    #[test]
    fn pbias_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(pbias(&obs, &sim), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn pbias_overestimation_positive() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(pbias(&obs, &sim) > 0.0);
    }

    #[test]
    fn pbias_underestimation_negative() {
        let obs = [2.0, 3.0, 4.0, 5.0, 6.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(pbias(&obs, &sim) < 0.0);
    }

    #[test]
    fn pbias_zero_observed_returns_inf() {
        let obs = [0.0; 5];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pbias(&obs, &sim), f64::INFINITY);
    }

    #[test]
    fn pbias_known_value() {
        // obs = [10, 20, 30], sim = [12, 22, 28]
        // diff = 2 + 2 - 2 = 2
        // sum_obs = 60
        // pbias = 100 * 2 / 60 = 3.333...
        let obs = [10.0, 20.0, 30.0];
        let sim = [12.0, 22.0, 28.0];
        assert_relative_eq!(pbias(&obs, &sim), 100.0 * 2.0 / 60.0, epsilon = 1e-10);
    }

    // --- RMSE tests ---

    #[test]
    fn rmse_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(rmse(&obs, &sim), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn rmse_constant_error() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [2.0, 3.0, 4.0, 5.0, 6.0];
        assert_relative_eq!(rmse(&obs, &sim), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn rmse_always_nonnegative() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [5.0, 1.0, 2.0];
        assert!(rmse(&obs, &sim) >= 0.0);
    }

    #[test]
    fn rmse_known_value() {
        // obs = [1,2,3], sim = [1,2,4] -> errors = [0,0,1] -> mse = 1/3 -> rmse = sqrt(1/3)
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.0, 2.0, 4.0];
        assert_relative_eq!(rmse(&obs, &sim), (1.0_f64 / 3.0).sqrt(), epsilon = 1e-10);
    }

    // --- MAE tests ---

    #[test]
    fn mae_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(mae(&obs, &sim), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn mae_constant_error() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [2.0, 3.0, 4.0, 5.0, 6.0];
        assert_relative_eq!(mae(&obs, &sim), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mae_symmetric_error() {
        let obs = [2.0, 2.0];
        let sim = [1.0, 3.0];
        assert_relative_eq!(mae(&obs, &sim), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mae_always_nonnegative() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [5.0, 1.0, 2.0];
        assert!(mae(&obs, &sim) >= 0.0);
    }
}
