/// Benchmark pydrology's GR2M: 1000 iterations of 1200 months (100 years).
use pydrology::gr2m::params::Parameters;
use pydrology::gr2m::run;
use std::time::Instant;

fn main() {
    // 100 years of monthly data (1200 timesteps)
    // Simple deterministic "random" data matching the Python benchmark range
    let n = 1200;
    let precip: Vec<f64> = (0..n)
        .map(|i| 10.0 + (i as f64 * 7.13).sin().abs() * 110.0)
        .collect();
    let pet: Vec<f64> = (0..n)
        .map(|i| 10.0 + (i as f64 * 3.77).sin().abs() * 70.0)
        .collect();

    let params = Parameters::new(500.0, 1.0).unwrap();

    // Warmup
    let _ = run::run(&params, &precip, &pet, None);

    // Benchmark
    let n_iters = 1000;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = run::run(&params, &precip, &pet, None);
    }
    let elapsed = start.elapsed();

    let total_timesteps = n * n_iters;
    let secs = elapsed.as_secs_f64();
    println!(
        "Rust:           {} runs x {} months = {} timesteps",
        n_iters, n, total_timesteps
    );
    println!("  Total time:  {:.3}s", secs);
    println!("  Per run:     {:.3}ms", secs / n_iters as f64 * 1000.0);
    println!(
        "  Throughput:  {:.0} timesteps/sec",
        total_timesteps as f64 / secs
    );
}
