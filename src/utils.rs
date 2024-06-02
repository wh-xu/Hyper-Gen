use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};

use log::{info, warn};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::Local;
use clap::{arg, value_parser, Command};
use env_logger::{Builder, Target};
use log::LevelFilter;
use std::io::Write;

use crate::{hd, params, types::*};

pub fn create_cli() -> CliParams {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d-%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, LevelFilter::Info)
        .target(Target::Stdout)
        .init();

    let cmd = Command::new("hyper-gen")
        .bin_name("hyper-gen")
        .subcommand_required(true)
        .version(params::VERSION)
        .about(
            "HyperGen: Fast and memory-efficient genome sketching in hyperdimensional space\n\n
        1. Genome sketching using FracMinhash and hyperdimensional computing (HDC). Three file types (.fna .fa .fasta) are supported:\n
        hyper-gen-rust sketch -p {fna_path} -o {output_sketch_file} \n\n
        2. ANI estimation and database search:\n
        hyper-gen-rust dist -r {ref_sketch} -q {query_sketch} -o {output_ANI_results}",
        )
        .subcommand(
            // sketch command
            clap::command!(params::CMD_SKETCH).args(&[
                arg!(-p --path <PATH> "Input folder path to sketch").required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(-r --path_r <PATH_R> "Path to ref sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-q --path_q <PATH_Q> "Path to query sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-o --out [OUT] "Output path ").required(true).value_parser(value_parser!(PathBuf)),
                arg!(-t --thread <THREAD> "# of threads used for computation")
                    .default_value("16")
                    .value_parser(value_parser!(u8)),
                arg!(-m --sketch_method <METHOD> "Sketch method")
                    .default_value("t1ha2")
                    .value_parser(value_parser!(String)),
                arg!(-C --canonical <CANONICAL> "If use canonical kmer")
                    .default_value("true")
                    .value_parser(value_parser!(bool)),
                arg!(-k --ksize <KSIZE> "k-mer size for sketching")
                    .default_value("21")
                    .value_parser(value_parser!(u8)),
                arg!(-S --seed <SEED> "Hash seed")
                    .default_value("123")
                    .value_parser(value_parser!(u64)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-Q --quant_scale <HD_D> "Scaling factor for HV quantization")
                    .default_value("1.0")
                    .value_parser(value_parser!(f32)),
                arg!(-a --ani_th <ANI_TH> "ANI threshold")
                    .default_value("85.0")
                    .value_parser(value_parser!(f32)),
                arg!(-D --device <DEVICE> "Device to run")
                    .default_value("cpu")
                    .value_parser(value_parser!(String)),
            ]),
        )
        .subcommand(
            // dist command
            clap::command!(params::CMD_DIST).args(&[
                arg!(-p --path <PATH> "Path to sketch file")
                    .default_value("1")
                    .value_parser(value_parser!(PathBuf)),
                arg!(-r --path_r <PATH_R> "Path to ref sketch file").required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(-q --path_q <PATH_Q> "Path to query sketch file").required(true)
                    .value_parser(value_parser!(PathBuf)),
                arg!(-o --out [OUT] "Output path ").required(true).value_parser(value_parser!(PathBuf)),
                arg!(-t --thread <THREAD> "# of threads used for computation")
                    .default_value("16")
                    .value_parser(value_parser!(u8)),
                arg!(-m --sketch_method <METHOD> "Sketch method")
                    .default_value("fracminhash")
                    .value_parser(value_parser!(String)),
                arg!(-C --canonical <CANONICAL> "If use canonical kmer")
                    .default_value("true")
                    .value_parser(value_parser!(bool)),
                arg!(-k --ksize <KSIZE> "k-mer size for sketching")
                    .default_value("21")
                    .value_parser(value_parser!(u8)),
                arg!(-S --seed <SEED> "Hash seed")
                    .default_value("123")
                    .value_parser(value_parser!(u64)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-Q --quant_scale <HD_D> "Scaling factor for HV quantization")
                    .default_value("1.0")
                    .value_parser(value_parser!(f32)),
                arg!(-a --ani_th <ANI_TH> "ANI threshold")
                    .default_value("85.0")
                    .value_parser(value_parser!(f32)),
                arg!(-D --device <DEVICE> "Device to run")
                    .default_value("cpu")
                    .value_parser(value_parser!(String)),
            ]),
        )
        .subcommand(
            // search command
            clap::command!(params::CMD_SEARCH).args(&[
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
                arg!(-m --sketch_method <METHOD> "Sketch method")
                    .default_value("fracminhash")
                    .value_parser(value_parser!(String)),
                arg!(-s --scaled <SCALED> "Scaled factor for FracMinHash")
                    .default_value("1500")
                    .value_parser(value_parser!(u64)),
                arg!(-d --hv_d <HD_D> "Dimension for hypervector")
                    .default_value("4096")
                    .value_parser(value_parser!(usize)),
                arg!(-Q --quant_scale <HD_D> "Scaling factor for HV quantization")
                    .default_value("1.0")
                    .value_parser(value_parser!(f32)),
                arg!(-a --ani_th <ANI_TH> "ANI threshold")
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
        canonical: matches.get_one::<bool>("canonical").expect("").clone(),
        seed: matches.get_one::<u64>("seed").expect("").clone(),
        scaled: matches.get_one::<u64>("scaled").expect("").clone(),
        hv_d: matches.get_one::<usize>("hv_d").expect("").clone(),
        hv_quant_scale: matches.get_one::<f32>("quant_scale").expect("").clone(),
        ani_threshold: matches.get_one::<f32>("ani_th").expect("").clone(),
        if_compressed: true, // TODO
        threads: matches.get_one::<u8>("thread").expect("").clone(),
        device: matches.get_one::<String>("device").expect("").clone(),
    };

    cli_params
}

pub fn get_fasta_files(path: &PathBuf) -> Vec<PathBuf> {
    // pub fn get_fasta_files(path: PathBuf) -> Vec<Result<PathBuf, GlobError>> {
    let mut all_files = Vec::new();
    for t in ["*.fna", "*.fa", "*.fasta"] {
        let mut files: Vec<_> = glob(path.join(t).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .map(|f| f.unwrap())
            .collect();

        all_files.append(&mut files);
    }

    all_files
}

pub fn get_progress_bar(n_file: usize) -> ProgressBar {
    let pb = ProgressBar::new(n_file as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} ({percent}%) - Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
    );

    pb
}

pub fn dump_sketch(file_sketch: &Vec<FileSketch>, out_file_path: &PathBuf) {
    let out_filename = out_file_path.to_str().unwrap();

    // Serialization
    let serialized = bincode::serialize::<Vec<FileSketch>>(&file_sketch).unwrap();

    // Dump sketch file
    fs::write(out_filename, &serialized).expect("Dump sketch file failed!");

    let sketch_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    info!(
        "Dump sketch file to {} with size {:.2} MB",
        out_filename, sketch_size_mb
    );
}

pub fn load_sketch(path: &Path) -> Vec<FileSketch> {
    info!("Loading sketch from {}", path.to_str().unwrap());
    let serialized = fs::read(path).expect("Opening sketch file failed!");
    let file_sketch = bincode::deserialize::<Vec<FileSketch>>(&serialized[..]).unwrap();

    file_sketch
}

pub fn dump_ani_file(sketch_dist: &SketchDist) {
    // Sort based on ANIs
    let mut indices = (0..sketch_dist.file_ani.len()).collect::<Vec<_>>();
    indices.sort_by(|&i1, &i2| {
        sketch_dist.file_ani[i1]
            .1
            .partial_cmp(&sketch_dist.file_ani[i2].1)
            .unwrap()
    });
    indices.reverse();

    // Dump in order
    let mut csv_str = String::new();
    let mut cnt: f32 = 0.0;
    for i in 0..sketch_dist.file_ani.len() {
        if sketch_dist.file_ani[indices[i]].1 >= sketch_dist.ani_threshold {
            csv_str.push_str(&format!(
                "{}\t{}\t{:.3}\n",
                sketch_dist.file_ani[indices[i]].0 .0,
                sketch_dist.file_ani[indices[i]].0 .1,
                sketch_dist.file_ani[indices[i]].1
            ));
            cnt += 1.0;
        } else {
            break;
        }
    }

    fs::write(sketch_dist.out_file.to_str().unwrap(), &csv_str.as_bytes())
        .expect("Dump ANI file failed!");

    // Warning if output ANIs are too sparse
    let total_dist = sketch_dist.file_ani.len() as f32;
    let perc = cnt / total_dist * 100.0;
    if perc < 5.0 {
        warn!(
            "Output ANIs with threshold {:.1} are too divergent: {} of {} ({:.2}%) ANIs are reported",
            sketch_dist.ani_threshold, cnt, total_dist, perc
        );
    } else {
        info!(
            "Output {} of {} ANIs above threshold {:.1} to file {}",
            cnt,
            total_dist,
            sketch_dist.ani_threshold,
            sketch_dist.out_file.to_str().unwrap()
        )
    }
}

use std::collections::HashMap;

pub fn dump_distribution_to_txt(path: &Path) {
    let mut file_sketch = load_sketch(path);

    hd::decompress_file_sketch(&mut file_sketch);

    // Write to files
    let data: Vec<Vec<i16>> = (0..file_sketch.len())
        .map(|i| file_sketch[i].hv.clone())
        .collect();

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
