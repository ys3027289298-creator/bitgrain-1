// Build script: compile the C SIMD kernels (DCT/quantization) so `cargo test`
// and the static/cdylib outputs are self-contained. The top-level Makefile
// still drives release packaging; this only makes cargo builds link cleanly.
fn main() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let c_dir = manifest.parent().expect("repo root").join("c");

    // Re-run when the kernels or their headers change.
    for src in ["dct.c", "quant.c", "dct.h", "quant.h"] {
        println!("cargo:rerun-if-changed={}", c_dir.join(src).display());
    }

    let mut build = cc::Build::new();
    build
        .file(c_dir.join("dct.c"))
        .file(c_dir.join("quant.c"))
        .include(&c_dir)
        .opt_level(3);

    // Mirror the Makefile's portable baseline unless native is explicitly
    // requested via CFLAGS-style target features. Keep this conservative so
    // distributed builds run on any x86-64/ARM host.
    if cfg!(target_arch = "x86_64") {
        build.flag("-msse2");
    }

    build.compile("bitgrain_simd");

    // dct.c uses libm on some targets (the Makefile links -lm).
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=m");
    }
}
