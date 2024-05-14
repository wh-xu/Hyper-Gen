use glob::glob;
use log::info;

use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use needletail::{parse_fastx_file, Sequence};

use crate::types::*;
use crate::{dist, hd, utils};

///  Sketch function to sketch all .fna files in folder path
#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) {
    let files: Vec<_> = glob(params.path.join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

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
        .map(|i| {
            let file = files[*i].as_ref().unwrap().clone();
            let mut sketch = Sketch::new(
                String::from(file.file_name().unwrap().to_str().unwrap()),
                &params,
            );

            // Extract kmer hash from genome sequence
            extract_kmer_hash(file, &mut sketch);

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

    pb.finish();

    println!(
        "Sketching {} files took {:.3}s\t {:.1} files/s",
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

        if sketch.sketch_method.as_str() == "mmhash64_xor_c" {
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

    // Filter kmers
    // for i in sketch.hash_set.clone() {
    //     if sketch.cbf.estimate_count(&i) > 8 {
    //         sketch.hash_set.remove(&i);
    //     }
    // }
}
