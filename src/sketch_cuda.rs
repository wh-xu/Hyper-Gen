use crate::types::*;

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
use {
    crate::{dist, fastx_reader, hd, utils},
    cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig},
    cudarc::nvrtc::Ptx,
    glob::glob,
    log::info,
    rayon::prelude::*,
    std::cmp::max,
    std::collections::HashSet,
    std::path::Path,
    std::path::PathBuf,
    std::sync::Arc,
    std::time::Instant,
};

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
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

#[allow(unused_variables)]
#[cfg(not(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper")))]
pub fn sketch_cuda(params: SketchParams) {
    use log::error;

    error!("Cuda sketching is not supported. Please add `--features cuda-sketch-ada-lovelace/cuda-sketch-ampere/cuda-sketch-hopper` for installation to enable it.");
}

//  Sketch function to sketch all .fna files in folder path
#[cfg(all(target_arch = "x86_64", any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper")))]
pub fn sketch_cuda(params: SketchParams) {
    let files = utils::get_fasta_files(&params.path);
    let n_file = files.len();

    info!("Start GPU sketching...");
    let pb = utils::get_progress_bar(n_file);

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

    let mut all_filesketch: Vec<FileSketch> = (0..n_file)
        .into_iter()
        .map(|i| FileSketch {
            ksize: params.ksize,
            scaled: params.scaled,
            seed: params.seed,
            canonical: params.canonical,
            hv_d: params.hv_d,
            hv_quant_bits: 16 as u8,
            hv_norm_2: 0 as i32,
            file_str: files[i].display().to_string(),
            hv: Vec::<i16>::new(),
        })
        .collect();

    // start cuda sketching
    all_filesketch.par_iter_mut().for_each(|sketch| {
        // NOTE: this is the important call to have
        // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
        gpu.bind_to_thread().unwrap();

        // Extract kmer set from genome sequence
        let kmer_hash_set = extract_kmer_t1ha2_cuda(&sketch, gpu);

        // Encode extracted kmer hash into sketch HV
        let hv = if is_x86_feature_detected!("avx2") {
            unsafe { hd::encode_hash_hd_avx2(&kmer_hash_set, &sketch) }
        } else {
            hd::encode_hash_hd(&kmer_hash_set, &sketch)
        };

        // Pre-compute HV's norm
        sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);

        // Sketch HV compression
        if params.if_compressed {
            sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(sketch, &hv) };
        }

        pb.inc(1);
        pb.eta();
    });

    pb.finish_and_clear();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        files.len(),
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    // Dump sketch file
    utils::dump_sketch(&all_filesketch, &params.out_file);
}

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
fn extract_kmer_t1ha2_cuda(sketch: &FileSketch, gpu: &Arc<CudaDevice>) -> HashSet<u64> {
    let fna_file = PathBuf::from(sketch.file_str.clone());
    let fna_seqs = fastx_reader::read_merge_seq(&fna_file);

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

    let mut kmer_hash_set = HashSet::<u64>::new();
    for h in host_kmer_hash {
        if h != 0 {
            kmer_hash_set.insert(h);
        }
    }

    kmer_hash_set
}

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Vec<HashSet<u64>> {
    // get files
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));
    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

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

#[cfg(any(feature = "cuda-sketch-ada-lovelace", feature = "cuda-sketch-ampere", feature = "cuda-sketch-hopper"))]
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

    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

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
