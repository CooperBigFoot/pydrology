use pydrology_macros::Fluxes;

#[derive(Debug, Clone, Copy, Fluxes)]
#[fluxes(timeseries_name = "SnowTimeseries")]
pub struct SnowFluxes {
    pub melt: f64,
    pub snow_pack: f64,
}

fn main() {
    let f = SnowFluxes { melt: 1.5, snow_pack: 50.0 };
    let mut ts = SnowTimeseries::with_capacity(5);
    ts.push(&f);
    assert_eq!(ts.len(), 1);
    assert_eq!(SnowFluxes::field_names(), &["melt", "snow_pack"]);
}
