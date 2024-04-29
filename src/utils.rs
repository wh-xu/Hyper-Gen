use std::fs;
use std::path::{Path, PathBuf};

use clap::{arg, value_parser, Command};

use crate::{params, types::*};

use crate::hd;

pub fn create_cli() -> CliParams {
    let cmd = Command::new("hyper-gen")
        .bin_name("hyper-gen")
        .subcommand_required(true)
        .version(params::VERSION)
        .about(
            "HyperGen: Fast and memory-efficient genome sketching\n\n
        1. Genome sketching using hyperdimensional computing (HDC):\n
        hyper-gen-rust sketch -p {fna_path} -o {output_sketch_file} \n\n
        2. ANI estimation and database search:\n
        hyper-gen-rust dist -r {ref_sketch} -q {query_sketch} -o {output_ANI_results}",
        )
        .subcommand(
            // sketch command
            clap::command!(params::CMD_SKETCH).args(&[
                arg!(-p --path <PATH> "Input folder path to sketch")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-r --path_r <PATH_R> "Path to ref sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-q --path_q <PATH_Q> "Path to query sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-o --out [OUT] "Output path ").value_parser(value_parser!(PathBuf)),
                arg!(-t --thread <THREAD> "# of threads used for computation")
                    .default_value("16")
                    .value_parser(value_parser!(u8)),
                arg!(-S --sketch_method <METHOD> "Sketch method")
                    .default_value("fracminhash")
                    .value_parser(value_parser!(String)),
                arg!(-k --ksize <KSIZE> "k-mer size for sketching")
                    .default_value("21")
                    .value_parser(value_parser!(u8)),
                arg!(-m --min_kmer <MIN_KMER_CNT> "Minimum valid k-mer count")
                    .default_value("1")
                    .value_parser(value_parser!(u32)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-th --ani_th <ANI_TH> "ANI threshold")
                    .default_value("85.0")
                    .value_parser(value_parser!(f32)),
            ]),
        )
        .subcommand(
            // dist command
            clap::command!(params::CMD_DIST).args(&[
                arg!(-p --path <PATH> "Path to sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-r --path_r <PATH_R> "Path to ref sketch file")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-q --path_q <PATH_Q> "Path to query sketch file")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-o --out [OUT] "Output path ").value_parser(value_parser!(PathBuf)),
                arg!(-t --thread <THREAD> "# of threads used for computation")
                    .default_value("16")
                    .value_parser(value_parser!(u8)),
                arg!(-k --ksize <KSIZE> "k-mer size for sketching")
                    .default_value("21")
                    .value_parser(value_parser!(u8)),
                arg!(-S --sketch_method <METHOD> "Sketch method")
                    .default_value("fracminhash")
                    .value_parser(value_parser!(String)),
                arg!(-m --min_kmer <MIN_KMER_CNT> "Minimum valid k-mer count")
                    .default_value("1")
                    .value_parser(value_parser!(u32)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-th --ani_th <ANI_TH> "ANI threshold")
                    .default_value("85.0")
                    .value_parser(value_parser!(f32)),
            ]),
        )
        .subcommand(
            // search command
            clap::command!(params::CMD_SEARCH).args(&[
                arg!(-p --path <PATH> "Path to sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-r --path_r <PATH_R> "Path to ref sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-q --path_q <PATH_Q> "Path to query sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-o --out [OUT] "Output path ").value_parser(value_parser!(PathBuf)),
                arg!(-t --thread <THREAD> "# of threads used for computation")
                    .default_value("16")
                    .value_parser(value_parser!(u8)),
                arg!(-k --ksize <KSIZE> "k-mer size for sketching")
                    .default_value("21")
                    .value_parser(value_parser!(u8)),
                arg!(-S --sketch_method <METHOD> "Sketch method")
                    .default_value("fracminhash")
                    .value_parser(value_parser!(String)),
                arg!(-m --min_kmer <MIN_KMER_CNT> "Minimum valid k-mer count")
                    .default_value("1")
                    .value_parser(value_parser!(u32)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-th --ani_th <ANI_TH> "ANI threshold")
                    .default_value("85.0")
                    .value_parser(value_parser!(f32)),
            ]),
        );

    parse_cmd(cmd)
}

pub fn parse_cmd(cmd: Command) -> CliParams {
    let matches = cmd.get_matches();

    let (mode, matches) = match matches.subcommand() {
        Some((params::CMD_SKETCH, matches)) => (params::CMD_SKETCH, matches),
        Some((params::CMD_DIST, matches)) => (params::CMD_DIST, matches),
        Some((params::CMD_SEARCH, matches)) => (params::CMD_SEARCH, matches),
        _ => unreachable!("clap should ensure we don't get here"),
    };

    let cli_params = CliParams {
        mode: mode.to_string(),
        path: matches.get_one::<PathBuf>("path").expect("").clone(),
        path_ref_sketch: matches.get_one::<PathBuf>("path_r").expect("").clone(),
        path_query_sketch: matches.get_one::<PathBuf>("path_q").expect("").clone(),
        out_file: {
            if matches.contains_id("out") {
                matches.get_one::<PathBuf>("out").expect("").clone()
            } else {
                PathBuf::new()
            }
        },
        ksize: matches.get_one::<u8>("ksize").expect("").clone(),
        sketch_method: matches
            .get_one::<String>("sketch_method")
            .expect("")
            .clone(),
        min_kmer_cnt: matches.get_one::<u32>("min_kmer").expect("").clone(),
        scaled: matches.get_one::<u64>("scaled").expect("").clone(),
        hv_d: matches.get_one::<usize>("hv_d").expect("").clone(),
        ani_threshold: matches.get_one::<f32>("ani_th").expect("").clone(),
        if_compressed: true, // TODO
        threads: matches.get_one::<u8>("thread").expect("").clone(),
    };

    cli_params
}

pub fn dump_sketch(file_sketch: &Vec<Sketch>, params: &SketchParams) {
    let out_filename = params.out_file.to_str().unwrap();

    assert!(
        out_filename.ends_with(".sketch"),
        "The output sketch file should have an extension of .sketch"
    );

    // Serialization
    let all_sketch = FileSketch {
        ksize: params.ksize,
        scaled: params.scaled,
        hv_d: params.hv_d,
        hv_norm_2: file_sketch
            .into_iter()
            .map(|s| s.hv_l2_norm_sq.clone())
            .collect::<Vec<i32>>(),
        hv_quant_bits_vec: file_sketch
            .into_iter()
            .map(|s| s.hv_quant_bits.clone())
            .collect::<Vec<u8>>(),
        file_vec: file_sketch
            .into_iter()
            .map(|s| s.file_name.clone())
            .collect::<Vec<String>>(),
        hv_vec: file_sketch
            .into_iter()
            .map(|s| s.hv.clone())
            .collect::<Vec<Vec<i16>>>(),
    };

    // Dump sketch file
    let serialized = bincode::serialize::<FileSketch>(&all_sketch).unwrap();
    fs::write(params.out_file.to_str().unwrap(), &serialized).expect("Dump sketch file failed!");

    println!("Dump sketch file to {}", out_filename);
}

pub fn load_sketch(path: &Path) -> FileSketch {
    let serialized = fs::read(path).expect("Opening sketch file failed!");
    let file_sketch = bincode::deserialize::<FileSketch>(&serialized[..]).unwrap();

    file_sketch
}

pub fn dump_ani_file(sketch_dist: &SketchDist) {
    let mut csv_str = String::new();
    for i in 0..sketch_dist.files.len() {
        if sketch_dist.ani[i] >= sketch_dist.ani_threshold {
            csv_str.push_str(&format!(
                "{}\t{}\t{:.3}\n",
                sketch_dist.files[i].0, sketch_dist.files[i].1, sketch_dist.ani[i]
            ));
        }
    }

    fs::write(sketch_dist.out_file.to_str().unwrap(), &csv_str.as_bytes())
        .expect("Dump ANI file failed!");
}

use std::collections::HashMap;

pub fn dump_sketch_to_txt(path: &Path) {
    let mut file_sketch = load_sketch(path);

    hd::decompress_file_sketch(&mut file_sketch);

    // Write to files
    let data = file_sketch.hv_vec;

    // Create a histogram
    let mut hist: HashMap<i16, u32> = HashMap::new();
    for i in 0..data.len() {
        for j in &data[i] {
            if hist.get(j) == None {
                hist.insert(*j, 1);
            } else if let Some(c) = hist.get_mut(&j) {
                *c += 1;
            }
        }
    }

    for kv in hist {
        println!("{}\t{}", kv.0, kv.1);
    }
}
