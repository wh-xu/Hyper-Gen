use crate::hd;
use crate::types::*;
use crate::utils;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use std::time::Instant;

pub fn compute_hv_l2_norm(sketch: &mut Sketch) {
    sketch.hv_l2_norm_sq = sketch
        .hv
        .iter()
        .fold(0, |sum: i32, &num| sum + (num as i32 * num as i32));
}

pub fn compute_pairwise_ani(
    r: Vec<i16>,
    norm2_r: i32,
    q: Vec<i16>,
    norm2_q: i32,
    ksize: u8,
) -> f32 {
    let dot_r_q: i32 = r
        .iter()
        .zip(q.iter())
        .map(|(x, y)| (*x as i32) * (*y as i32))
        .sum();

    let jaccard: f32 = dot_r_q as f32 / (norm2_r + norm2_q - dot_r_q) as f32;
    let ani: f32 = 1.0 + (2.0 / (1.0 / jaccard + 1.0)).ln() / (ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.min(1.0).max(0.0) * 100.0
    }
}

pub fn compute_hv_ani(
    sketch_dist: &mut SketchDist,
    ref_filesketch: &FileSketch,
    query_filesketch: &FileSketch,
    ksize: u8,
    if_asymmetric: bool,
) {
    let num_ref_files = ref_filesketch.hv_vec.len();
    let num_query_files = query_filesketch.hv_vec.len();

    let num_dists = if if_asymmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb: ProgressBar = ProgressBar::new(num_dists as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut index_dist: Vec<(usize, usize)> = vec![(0, 0); num_dists];
    sketch_dist.files = vec![("".to_string(), "".to_string()); num_dists];

    let mut cnt = 0;
    for i in 0..num_ref_files {
        for j in {
            if if_asymmetric {
                i + 1
            } else {
                0
            }
        }..num_query_files
        {
            index_dist[cnt] = (i, j);
            sketch_dist.files[cnt] = (
                ref_filesketch.file_vec[i].clone(),
                query_filesketch.file_vec[j].clone(),
            );
            cnt += 1;
        }
    }

    sketch_dist.ani = index_dist
        .par_iter()
        .map(|(i, j)| {
            pb.inc(1);
            pb.eta();

            compute_pairwise_ani(
                ref_filesketch.hv_vec[*i].clone(),
                ref_filesketch.hv_norm_2[*i],
                query_filesketch.hv_vec[*j].clone(),
                query_filesketch.hv_norm_2[*j],
                ksize,
            )
        })
        .collect();

    pb.finish();
}

pub fn compute_compressed_hv_ani(
    sketch_dist: &mut SketchDist,
    ref_filesketch: &FileSketch,
    query_filesketch: &FileSketch,
    ksize: u8,
    if_asymmetric: bool,
) {
    let num_ref_files = ref_filesketch.hv_vec.len();
    let num_query_files = query_filesketch.hv_vec.len();

    let num_dists = if if_asymmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb: ProgressBar = ProgressBar::new(num_dists as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut index_dist: Vec<(usize, usize)> = vec![(0, 0); num_dists];
    sketch_dist.files = vec![("".to_string(), "".to_string()); num_dists];

    let mut cnt = 0;
    for i in 0..num_ref_files {
        for j in {
            if if_asymmetric {
                i + 1
            } else {
                0
            }
        }..num_query_files
        {
            index_dist[cnt] = (i, j);
            sketch_dist.files[cnt] = (
                ref_filesketch.file_vec[i].clone(),
                query_filesketch.file_vec[j].clone(),
            );
            cnt += 1;
        }
    }

    sketch_dist.ani = index_dist
        .par_iter()
        .map(|(i, j)| {
            pb.inc(1);
            pb.eta();

            // println!("{} vs. {}", ref_filesketch.0[*i], query_filesketch.0[*j]);

            let ref_hv_decompressed = unsafe {
                hd::decompress_hd_sketch(
                    &ref_filesketch.hv_vec[*i],
                    ref_filesketch.hv_d,
                    ref_filesketch.hv_quant_bits_vec[*i],
                )
            };
            // let ref_hv_decompressed = vec![0; ref_filesketch.hv_d];

            compute_pairwise_ani(
                ref_hv_decompressed,
                ref_filesketch.hv_norm_2[*i],
                query_filesketch.hv_vec[*j].clone(),
                query_filesketch.hv_norm_2[*j],
                ksize,
            )
        })
        .collect();

    pb.finish();
}

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();

    let if_asym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    // Load ref and query sketch files
    let mut ref_file_sketch = utils::load_sketch(&sketch_dist.path_ref_sketch.as_path());

    let mut query_file_sketch = if if_asym {
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(&sketch_dist.path_query_sketch.as_path())
    };

    assert_eq!(
        ref_file_sketch.ksize, query_file_sketch.ksize,
        "Ref and query sketches use different kmer sizes!"
    );

    assert_eq!(
        ref_file_sketch.hv_d, query_file_sketch.hv_d,
        "Ref and query sketches use different HV dimensions!"
    );

    println!("STEP 1 took {:.3}s", tstart.elapsed().as_secs_f32());
    let tstart = Instant::now();

    if if_asym {
        // Decompress sketch HVs
        hd::decompress_file_sketch(&mut ref_file_sketch);

        // println!("STEP 2.1 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        hd::decompress_file_sketch(&mut query_file_sketch);

        // println!("STEP 2.2 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        // Compute distance for decompressed sketch
        compute_hv_ani(
            sketch_dist,
            &ref_file_sketch,
            &query_file_sketch,
            ref_file_sketch.ksize,
            if_asym,
        );
    } else {
        hd::decompress_file_sketch(&mut query_file_sketch);

        // Compute distance for compressed sketch
        compute_compressed_hv_ani(
            sketch_dist,
            &ref_file_sketch,
            &query_file_sketch,
            ref_file_sketch.ksize,
            if_asym,
        );
    }

    // println!("STEP 3 took {:.3}s", tstart.elapsed().as_secs_f32());
    // let tstart = Instant::now();

    // Dump dist file
    utils::dump_ani_file(&sketch_dist);

    // println!("STEP 4 took {:.3}s", tstart.elapsed().as_secs_f32());
    // let tstart = Instant::now();

    println!(
        "{} refs and {} queries took {:.3}s",
        ref_file_sketch.file_vec.len(),
        query_file_sketch.file_vec.len(),
        tstart.elapsed().as_secs_f32()
    );
}
