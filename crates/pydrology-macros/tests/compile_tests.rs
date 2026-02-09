#[test]
fn compile_tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/basic.rs");
    t.pass("tests/pass/custom_name.rs");
    t.compile_fail("tests/fail/non_f64_field.rs");
}
