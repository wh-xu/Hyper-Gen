extern crate bindgen;
extern crate cc;

use std::{env, path::PathBuf, process::Command};

use bindgen::CargoCallbacks;
use regex::Regex;

fn main() {
    println!("cargo:rerun-if-changed={}", "cuda");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_src = PathBuf::from("src/cuda_kernel.cu");
    let ptx_file = out_dir.join("cuda_kmer_hash.ptx");

    // Determine architecture based on feature flags
    let (arch, code) = if cfg!(feature = "cuda-sketch-ada-lovelace") {
        ("compute_89", "sm_89") // Ada Lovelace architecture for NIVIDA RTX 4090 series
    } else if cfg!(feature = "cuda-sketch-ampere") {
        ("compute_80", "sm_80") // Ampere architecture for NIVIDA A100 series
    } else if cfg!(feature = "cuda-sketch-hopper") {
        ("compute_90", "sm_90") // // Hopper architecture for NIVIDA H100 series
    } else {
        panic!("Unsupported GPU architecture feature flag!");
    };

    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cuda_src)
        .arg(format!("-arch={}", arch))
        .arg(format!("-code={}", code))
        .status()
        .unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );

    let bindings = bindgen::Builder::default()
        .header("src/cuda_kernel.h")
        .parse_callbacks(Box::new(CargoCallbacks))
        .no_copy("*")
        .no_debug("*")
        .generate()
        .expect("Unable to generate bindings");

    let generated_bindings = bindings.to_string();

    let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}
