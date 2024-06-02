use crate::hd;
use crate::types::*;
use crate::utils;
use std::time::Instant;

use log::info;
use rayon::prelude::*;

use std::arch::x86_64::*;

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    // Load ref and query sketch files
    let mut ref_file_sketch = utils::load_sketch(&sketch_dist.path_ref_sketch.as_path());

    // Create query sketch
    let mut query_file_sketch = if if_sym {
        // just clone if ref and query are the same
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(&sketch_dist.path_query_sketch.as_path())
    };

    let ksize_ref = ref_file_sketch[0].ksize;
    let ksize_query = query_file_sketch[0].ksize;

    assert_eq!(
        ksize_ref, ksize_query,
        "Ref and query sketches use different kmer sizes!"
    );

    let hv_d_ref = ref_file_sketch[0].hv_d;
    let hv_d_query = query_file_sketch[0].hv_d;
    assert_eq!(
        hv_d_ref, hv_d_query,
        "Ref and query sketches use different HV dimensions!"
    );

    // Decompress sketch HVs
    hd::decompress_file_sketch(&mut ref_file_sketch);
    hd::decompress_file_sketch(&mut query_file_sketch);

    // Compute ANIs for decompressed sketch
    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        &query_file_sketch,
        ksize_ref,
        if_sym,
    );

    // Dump dist file
    utils::dump_ani_file(&sketch_dist);

    info!(
        "Computed ANIs for {} ref files and {} query files took {:.3}s",
        ref_file_sketch.len(),
        query_file_sketch.len(),
        tstart.elapsed().as_secs_f32()
    );
}

// pub fn compute_compressed_hv_ani(
//     sketch_dist: &mut SketchDist,
//     ref_filesketch: &Vec<FileSketch>,
//     query_filesketch: &Vec<FileSketch>,
//     ksize: u8,
//     if_symmetric: bool,
// ) {
//     info!("Computing ANI..");

//     let num_ref_files = ref_filesketch.len();
//     let num_query_files = query_filesketch.len();

//     let num_dists = if if_symmetric {
//         num_ref_files * (num_query_files - 1) / 2
//     } else {
//         num_ref_files * num_query_files
//     };

//     let pb = utils::get_progress_bar(num_dists);

//     let mut index_dist: Vec<(usize, usize)> = vec![(0, 0); num_dists];
//     sketch_dist.files = vec![("".to_string(), "".to_string()); num_dists];

//     let mut cnt = 0;
//     for i in 0..num_ref_files {
//         for j in {
//             if if_symmetric {
//                 i + 1
//             } else {
//                 0
//             }
//         }..num_query_files
//         {
//             index_dist[cnt] = (i, j);
//             sketch_dist.files[cnt] = (ref_filesketch[i].file_str, query_filesketch[j].file_str);
//             cnt += 1;
//         }
//     }

//     // sketch_dist.ani = vec![0.0; num_dists];

//     // sketch_dist.ani.into_par_iter().for_each(|ani|{

//     // });

//     sketch_dist.ani = index_dist
//         .par_iter()
//         .map(|(i, j)| {
//             pb.inc(1);
//             pb.eta();

//             let ref_hv_decompressed = unsafe { hd::decompress_hd_sketch(ref_filesketch[*i]) };
//             // let ref_hv_decompressed = vec![0; ref_filesketch.hv_d];

//             compute_pairwise_ani(
//                 ref_hv_decompressed,
//                 ref_filesketch.hv_norm_2[*i],
//                 query_filesketch.hv_vec[*j].clone(),
//                 query_filesketch.hv_norm_2[*j],
//                 ksize,
//             )
//         })
//         .collect();

//     pb.finish_and_clear();
// }

pub fn compute_hv_l2_norm(hv: &Vec<i16>) -> i32 {
    let hv_l2_norm_sq = hv
        .iter()
        .fold(0, |sum: i32, &num| sum + (num as i32 * num as i32));
    hv_l2_norm_sq
}

pub fn compute_pairwise_ani(
    r: &Vec<i16>,
    norm2_r: i32,
    q: &Vec<i16>,
    norm2_q: i32,
    ksize: u8,
) -> f32 {
    // Scalar-based inner product
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

#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_ani_avx2(
    r: Vec<i16>,
    norm2_r: i32,
    q: Vec<i16>,
    norm2_q: i32,
    ksize: u8,
) -> f32 {
    // SIMD-based inner product
    let len = r.len();
    let mut dot_r_q: i32 = 0;
    for i in 0..len / 16 {
        let mm256_r = _mm256_set_epi16(
            r[i * 16 + 0],
            r[i * 16 + 1],
            r[i * 16 + 2],
            r[i * 16 + 3],
            r[i * 16 + 4],
            r[i * 16 + 5],
            r[i * 16 + 6],
            r[i * 16 + 7],
            r[i * 16 + 8],
            r[i * 16 + 9],
            r[i * 16 + 10],
            r[i * 16 + 11],
            r[i * 16 + 12],
            r[i * 16 + 13],
            r[i * 16 + 14],
            r[i * 16 + 15],
        );
        let mm256_q = _mm256_set_epi16(
            q[i * 16 + 0],
            q[i * 16 + 1],
            q[i * 16 + 2],
            q[i * 16 + 3],
            q[i * 16 + 4],
            q[i * 16 + 5],
            q[i * 16 + 6],
            q[i * 16 + 7],
            q[i * 16 + 8],
            q[i * 16 + 9],
            q[i * 16 + 10],
            q[i * 16 + 11],
            q[i * 16 + 12],
            q[i * 16 + 13],
            q[i * 16 + 14],
            q[i * 16 + 15],
        );

        let mm256_madd_32x8 = _mm256_madd_epi16(mm256_r, mm256_q);
        let mm256_madd_32x4 = _mm256_hadd_epi32(mm256_madd_32x8, _mm256_setzero_si256());
        let dot = _mm256_extract_epi32::<0>(mm256_madd_32x4)
            + _mm256_extract_epi32::<1>(mm256_madd_32x4)
            + _mm256_extract_epi32::<4>(mm256_madd_32x4)
            + _mm256_extract_epi32::<5>(mm256_madd_32x4);
        dot_r_q += dot;
    }

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
    ref_filesketch: &Vec<FileSketch>,
    query_filesketch: &Vec<FileSketch>,
    ksize: u8,
    if_symmetric: bool,
) {
    info!("Computing ANI..");

    let num_ref_files = ref_filesketch.len();
    let num_query_files = query_filesketch.len();

    let num_dists = if if_symmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb = utils::get_progress_bar(num_dists);

    let mut cnt = 0;
    let mut index_dist = vec![(0, 0); num_dists];
    for i in 0..num_ref_files {
        for j in {
            if if_symmetric {
                i + 1
            } else {
                0
            }
        }..num_query_files
        {
            index_dist[cnt] = (i, j);
            cnt += 1;
        }
    }

    sketch_dist.file_ani = vec![(("".to_string(), "".to_string()), 0.0); num_dists];
    sketch_dist
        .file_ani
        .par_iter_mut()
        .zip(index_dist)
        .for_each(|(file_ani_pair, ind)| {
            let ani = compute_pairwise_ani(
                &ref_filesketch[ind.0].hv,
                ref_filesketch[ind.0].hv_norm_2,
                &query_filesketch[ind.1].hv,
                query_filesketch[ind.1].hv_norm_2,
                ksize,
            );

            *file_ani_pair = (
                (
                    ref_filesketch[ind.0].file_str.clone(),
                    query_filesketch[ind.1].file_str.clone(),
                ),
                ani,
            );

            pb.inc(1);
            pb.eta();
        });

    pb.finish_and_clear();
}
