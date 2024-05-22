use std::{arch::x86_64::*, path::PathBuf};

use log::info;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use needletail::{parse_fastx_file, Sequence};

use crate::types::*;
use crate::{dist, hd, utils};

///  Sketch function to sketch all .fna files in folder path
#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) {
    let files = utils::get_fasta_files(&params.path);

    info!("Start sketching...");

    let pb = ProgressBar::new(files.len() as u64);

    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap()
            .progress_chars("##-"),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let file_sketch: Vec<Sketch> = index_vec
        .par_iter()
        .map(|&i| {
            let file = files[i].clone();
            let mut sketch = Sketch::new(
                String::from(file.file_name().unwrap().to_str().unwrap()),
                &params,
            );

            // Extract kmer hash from genome sequence
            if is_x86_feature_detected!("avx2") && sketch.sketch_method.contains("avx") {
                unsafe {
                    extract_kmer_hash_avx2(file, &mut sketch);
                }
            } else {
                extract_kmer_hash(file, &mut sketch);
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

fn extract_kmer_hash(file: PathBuf, sketch: &mut Sketch) {
    let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

    while let Some(record) = fastx_reader.next() {
        let seqrec: needletail::parser::SequenceRecord<'_> = record.expect("invalid record");

        // normalize to make sure all the bases are consistently capitalized
        let norm_seq = seqrec.normalize(false);

        if sketch.sketch_method.contains("64") {
            for (_, (kmer_u64, _), _) in norm_seq.bit_kmers(sketch.ksize, true) {
                sketch.insert_kmer_u64(kmer_u64);
            }
        } else {
            // we make a reverse complemented copy of the sequence
            let rc = norm_seq.reverse_complement();

            for (_, kmer, _) in norm_seq.canonical_kmers(sketch.ksize, &rc) {
                sketch.insert_kmer(kmer);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn extract_kmer_hash_avx2(file: PathBuf, sketch: &mut Sketch) {
    let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

    while let Some(record) = fastx_reader.next() {
        let seqrec: needletail::parser::SequenceRecord<'_> = record.expect("invalid record");

        // normalize to make sure all the bases are consistently capitalized
        let norm_seq = seqrec.normalize(false);

        let mut bitkmer_array: [i64; 4] = [0, 0, 0, 0];
        let mut bitkmer_m256: __m256i;
        let mut cnt: usize = 0;
        for (_, (bit_kmer_u64, _), _) in norm_seq.bit_kmers(sketch.ksize, true) {
            if cnt < 4 {
                bitkmer_array[cnt] = bit_kmer_u64 as i64;
                cnt += 1;
            } else {
                bitkmer_m256 = _mm256_set_epi64x(
                    bitkmer_array[0],
                    bitkmer_array[1],
                    bitkmer_array[2],
                    bitkmer_array[3],
                );
                sketch.insert_kmer_u64_avx2(bitkmer_m256);
                cnt = 0;
            }
        }

        if cnt > 0 {
            bitkmer_m256 = match cnt {
                1 => _mm256_set_epi64x(bitkmer_array[0], 0, 0, 0),
                2 => _mm256_set_epi64x(bitkmer_array[0], bitkmer_array[1], 0, 0),
                3 => _mm256_set_epi64x(bitkmer_array[0], bitkmer_array[1], bitkmer_array[2], 0),
                _ => _mm256_set_epi64x(
                    bitkmer_array[0],
                    bitkmer_array[1],
                    bitkmer_array[2],
                    bitkmer_array[3],
                ),
            };
            sketch.insert_kmer_u64_avx2(bitkmer_m256);
        }
    }
}
