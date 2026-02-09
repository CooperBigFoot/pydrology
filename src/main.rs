use pydrology::gr2m::params::Parameters;
use pydrology::gr2m::run;

fn main() {
    let p = Parameters::new(500.0, 1.0).unwrap();

    // 12 months of forcing data (mm/month)
    let precip = [
        80.0, 70.0, 60.0, 40.0, 20.0, 10.0, 5.0, 10.0, 30.0, 50.0, 70.0, 90.0,
    ];
    let pet = [
        20.0, 25.0, 40.0, 55.0, 60.0, 65.0, 60.0, 50.0, 35.0, 25.0, 20.0, 15.0,
    ];

    // Run the model
    let result = run::run(&p, &precip, &pet, None);

    // Print results
    println!("Month | Precip |  PET  |  Q (streamflow) | Prod Store | Rout Store");
    println!("------|--------|-------|-----------------|------------|----------");
    for t in 0..result.len() {
        println!(
            "  {:>2}  | {:>6.1} | {:>5.1} | {:>15.2} | {:>10.2} | {:>10.2}",
            t + 1,
            result.precip[t],
            result.pet[t],
            result.streamflow[t],
            result.production_store[t],
            result.routing_store[t],
        );
    }

    // Water balance check
    let total_p: f64 = result.precip.iter().sum();
    let total_q: f64 = result.streamflow.iter().sum();
    let total_ae: f64 = result.actual_et.iter().sum();
    println!(
        "\nTotals: P={:.1}, Q={:.1}, AE={:.1}",
        total_p, total_q, total_ae
    );
}
