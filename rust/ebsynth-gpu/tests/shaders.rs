//! Every WGSL module parses and validates (catches shader errors without a GPU).

fn validate(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
}

#[test]
fn ebsynth_shader_validates() {
    validate("ebsynth.wgsl", include_str!("../src/shaders/ebsynth.wgsl"));
}

#[test]
fn fastblend_shaders_validate() {
    let common = include_str!("../src/shaders/fastblend_common.wgsl");
    for (name, body) in [
        (
            "fastblend_image.wgsl",
            include_str!("../src/shaders/fastblend_image.wgsl"),
        ),
        (
            "fastblend_match.wgsl",
            include_str!("../src/shaders/fastblend_match.wgsl"),
        ),
        (
            "fastblend_init.wgsl",
            include_str!("../src/shaders/fastblend_init.wgsl"),
        ),
    ] {
        validate(name, &format!("{common}\n{body}"));
    }
}
