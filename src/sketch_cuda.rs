use std::cmp::max;
use std::path::Path;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use crate::types::*;
use crate::{dist, fastx_reader, hd, utils};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use std::path::PathBuf;

use needletail::{parse_fastx_file, Sequence};

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

const SEQ_NT4_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

pub fn sketch_cpu_parallel(path_fna: &String, ksize: usize, scaled: u64) -> Vec<HashSet<u64>> {
    // get files
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // start sketching
    let threshold = u64::MAX / scaled;

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| {
            let mut fastx_reader = parse_fastx_file(&files[i]).expect("Opening .fna files failed");

            let mut sketch_kmer_set = HashSet::<u64>::default();

            while let Some(record) = fastx_reader.next() {
                let seqrec = record.expect("invalid record");
                let norm_seq = seqrec.normalize(false);

                for (_, (kmer_u64, _), _) in norm_seq.bit_kmers(ksize as u8, true) {
                    let h = mm_hash64(kmer_u64);
                    if h < threshold {
                        sketch_kmer_set.insert(h);
                    }
                }
            }
            pb.inc(1);
            pb.eta();
            sketch_kmer_set
        })
        .collect();

    pb.finish_and_clear();

    sketch_kmer_sets
}

pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Vec<HashSet<u64>> {
    // get files
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // setup GPU device
    let gpu = CudaDevice::new(0).unwrap();

    // compile ptx
    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "cuda_kernel", &["cuda_kmer_bit_pack_mmhash"])
        .unwrap();
    let gpu = &gpu;

    // start sketching
    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| {
            // NOTE: this is the important call to have
            // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
            gpu.bind_to_thread().unwrap();

            // let now = Instant::now();

            let fna_seqs = fastx_reader::read_merge_seq(&files[i]);

            // println!("Time taken to extract seq: {:.2?}", now.elapsed());

            let n_bps = fna_seqs.len();
            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 512;
            let n_threads = (n_kmers + bp_per_thread - 1) / bp_per_thread;

            let gpu_seq = gpu.htod_copy(fna_seqs).unwrap();
            let gpu_seq_nt4_table = gpu.htod_copy(SEQ_NT4_TABLE.to_vec()).unwrap();
            // allocate 4x more space that expected
            let n_hash_per_thread = max(bp_per_thread / scaled as usize * 3, 8);
            let n_hash_array = n_hash_per_thread * n_threads;
            let gpu_kmer_bit_hash = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

            let config = LaunchConfig::for_num_elems(n_threads as u32);
            let params = (
                &gpu_seq,
                n_bps,
                bp_per_thread,
                n_hash_per_thread,
                ksize,
                u64::MAX / scaled,
                canonical,
                &gpu_seq_nt4_table,
                &gpu_kmer_bit_hash,
            );
            let f = gpu
                .get_func("cuda_kernel", "cuda_kmer_bit_pack_mmhash")
                .unwrap();
            unsafe { f.clone().launch(config, params) }.unwrap();

            let host_kmer_bit_hash = gpu.sync_reclaim(gpu_kmer_bit_hash).unwrap();

            let mut sketch_kmer_set = HashSet::<u64>::default();
            for h in host_kmer_bit_hash {
                if h != 0 {
                    sketch_kmer_set.insert(h);
                }
            }

            pb.inc(1);
            pb.eta();
            sketch_kmer_set
        })
        .collect();

    pb.finish_and_clear();

    sketch_kmer_sets
}

pub fn cuda_t1ha2_hash_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
    seed: u64,
) -> Vec<HashSet<u64>> {
    // get files
    let files: Vec<_> = glob(Path::new(&path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // setup GPU device
    let gpu = CudaDevice::new(0).unwrap();

    // compile ptx
    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "cuda_kernel", &["cuda_kmer_t1ha2"])
        .unwrap();
    let gpu = &gpu;

    // start sketching
    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| {
            // NOTE: this is the important call to have
            // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
            gpu.bind_to_thread().unwrap();

            let now = Instant::now();

            let fna_seqs = fastx_reader::read_merge_seq(files[*i].as_ref().unwrap());

            println!("Time taken to extract seq: {:.2?}", now.elapsed());

            let n_bps = fna_seqs.len();
            let n_kmers = n_bps - ksize + 1;
            let kmer_per_thread = 512;
            let n_threads = (n_kmers + kmer_per_thread - 1) / kmer_per_thread;

            // copy to GPU
            let now = Instant::now();
            let gpu_seq = gpu.htod_copy(fna_seqs).unwrap();
            // allocate 4x more space that expected
            let n_hash_per_thread = max(kmer_per_thread / scaled as usize * 3, 8);
            let n_hash_array = n_hash_per_thread * n_threads;
            let gpu_kmer_hash = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

            println!("Time taken to copy to gpu: {:.2?}", now.elapsed());

            // execute kernel
            let now = Instant::now();

            let config = LaunchConfig::for_num_elems(n_threads as u32);
            let params = (
                &gpu_seq,
                n_bps,
                kmer_per_thread,
                n_hash_per_thread,
                ksize,
                u64::MAX / scaled,
                seed,
                canonical,
                &gpu_kmer_hash,
            );
            let f = gpu.get_func("cuda_kernel", "cuda_kmer_t1ha2").unwrap();
            unsafe { f.launch(config, params) }.unwrap();

            let host_kmer_hash = gpu.sync_reclaim(gpu_kmer_hash).unwrap();

            println!("Time taken to run kernel: {:.2?}", now.elapsed());
            let now = Instant::now();

            let mut sketch_kmer_set = HashSet::<u64>::default();
            for h in host_kmer_hash {
                if h != 0 {
                    sketch_kmer_set.insert(h);
                }
            }

            println!("Time taken to postprocess: {:.2?}", now.elapsed());
            pb.inc(1);
            pb.eta();
            sketch_kmer_set
        })
        .collect();

    pb.finish_and_clear();

    sketch_kmer_sets
}

//
fn extract_kmer_mmhash_bitpack_cuda(file: &PathBuf, sketch: &mut Sketch, gpu: &Arc<CudaDevice>) {
    let fna_seqs = fastx_reader::read_merge_seq(file);

    let n_bps = fna_seqs.len();
    let ksize = sketch.ksize as usize;
    let scaled = sketch.scaled;
    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512;
    let n_threads = (n_kmers + kmer_per_thread - 1) / kmer_per_thread;

    // copy to GPU
    let gpu_seq = gpu.htod_copy(fna_seqs).unwrap();
    let gpu_seq_nt4_table = gpu.htod_copy(SEQ_NT4_TABLE.to_vec()).unwrap();
    // allocate 4x more space that expected
    let n_hash_per_thread = max(kmer_per_thread / sketch.scaled as usize * 4, 8);
    let n_hash_array = n_hash_per_thread * n_threads;
    let gpu_kmer_bit_hash = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

    // execute kernel
    let config = LaunchConfig::for_num_elems(n_threads as u32);
    let params = (
        &gpu_seq,
        n_bps,
        kmer_per_thread,
        n_hash_per_thread,
        ksize,
        u64::MAX / scaled,
        sketch.canonical,
        &gpu_seq_nt4_table,
        &gpu_kmer_bit_hash,
    );
    let f = gpu
        .get_func("cuda_kernel", "cuda_kmer_bit_pack_mmhash")
        .unwrap();
    unsafe { f.clone().launch(config, params) }.unwrap();

    gpu.synchronize().unwrap();

    // let host_seq = gpu.sync_reclaim(gpu_seq).unwrap();
    let host_kmer_bit_hash = gpu.sync_reclaim(gpu_kmer_bit_hash).unwrap();
    for h in host_kmer_bit_hash {
        if h != 0 {
            sketch.hash_set.insert(h);
        }
    }
}

fn extract_kmer_t1ha2_cuda(file: &PathBuf, sketch: &mut Sketch, gpu: &Arc<CudaDevice>) {
    let fna_seqs = fastx_reader::read_merge_seq(file);

    let n_bps = fna_seqs.len();
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let scaled = sketch.scaled;
    let seed = sketch.seed;
    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512;
    let n_threads = (n_kmers + kmer_per_thread - 1) / kmer_per_thread;

    // copy to GPU
    let gpu_seq = gpu.htod_copy(fna_seqs).unwrap();
    // allocate 4x more space that expected
    let n_hash_per_thread = max(kmer_per_thread / sketch.scaled as usize * 4, 8);
    let n_hash_array = n_hash_per_thread * n_threads;
    let gpu_kmer_hash = gpu.alloc_zeros::<u64>(n_hash_array).unwrap();

    // execute kernel
    let config = LaunchConfig::for_num_elems(n_threads as u32);
    let params = (
        &gpu_seq,
        n_bps,
        kmer_per_thread,
        n_hash_per_thread,
        ksize,
        u64::MAX / scaled,
        seed,
        canonical,
        &gpu_kmer_hash,
    );
    let f = gpu.get_func("cuda_kernel", "cuda_kmer_t1ha2").unwrap();
    unsafe { f.launch(config, params) }.unwrap();

    let host_kmer_hash = gpu.sync_reclaim(gpu_kmer_hash).unwrap();
    for h in host_kmer_hash {
        if h != 0 {
            sketch.hash_set.insert(h);
        }
    }
}

//  Sketch function to sketch all .fna files in folder path
#[cfg(target_arch = "x86_64")]
pub fn sketch_cuda(params: SketchParams) {
    let files = utils::get_fasta_files(&params.path);

    info!("Start GPU sketching...");

    // progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    // setup GPU device
    let gpu = CudaDevice::new(0).unwrap();

    // compile ptx
    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(
        ptx,
        "cuda_kernel",
        &["cuda_kmer_bit_pack_mmhash", "cuda_kmer_t1ha2"],
    )
    .unwrap();
    let gpu = &gpu;

    // start cuda sketching
    let index_vec: Vec<usize> = (0..files.len()).collect();
    let file_sketch: Vec<Sketch> = index_vec
        .par_iter()
        .map(|&i| {
            // NOTE: this is the important call to have
            // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
            gpu.bind_to_thread().unwrap();

            let file = files[i].clone();
            let mut sketch = Sketch::new(
                String::from(file.file_name().unwrap().to_str().unwrap()),
                &params,
            );

            if params.sketch_method == "mmhash" {
                extract_kmer_mmhash_bitpack_cuda(&file, &mut sketch, gpu);
            } else {
                extract_kmer_t1ha2_cuda(&file, &mut sketch, gpu);
            }

            // Encode extracted kmer hash into sketch HV
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    hd::encode_hash_hd_avx2(&mut sketch);
                }
            } else {
                hd::encode_hash_hd(&mut sketch);
            }

            // Pre-compute HV's norm
            dist::compute_hv_l2_norm(&mut sketch);

            // Sketch HV compression
            if params.if_compressed {
                unsafe {
                    hd::compress_hd_sketch(&mut sketch);
                }
            }

            pb.inc(1);
            pb.eta();
            sketch
        })
        .collect();

    pb.finish_and_clear();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        files.len(),
        pb.elapsed().as_secs_f32(),
        (files.len() as f32 / pb.elapsed().as_secs_f32())
    );

    // Dump sketch file
    utils::dump_sketch(&file_sketch, &params);
}
