use log::info;
use std::collections::HashSet;
use std::path::PathBuf;

use needletail::{parse_fastx_file, Sequence};
use rayon::prelude::*;

use crate::types::*;
use crate::{dist, hd, utils};

#[cfg(target_arch = "x86_64")]
pub fn sketch(params: SketchParams) {
    let files = utils::get_fasta_files(&params.path);
    let n_file = files.len();

    info!("Start sketching...");
    let pb = utils::get_progress_bar(n_file);

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

    // Parallel sketching
    all_filesketch.par_iter_mut().for_each(|sketch| {
        // Extract kmer set from genome sequence
        let kmer_hash_set = extract_kmer_hash(&sketch);

        // Encode extracted kmer hash into sketch HV
        let mut hv = if is_x86_feature_detected!("avx2") {
            unsafe { hd::encode_hash_hd_avx2(&kmer_hash_set, &sketch) }
        } else {
            hd::encode_hash_hd(&kmer_hash_set, &sketch)
        };

        // Pre-compute HV's norm
        sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);

        // Sketch HV compression
        sketch.hv_quant_bits = if params.if_compressed {
            unsafe { hd::compress_hd_sketch_new(&mut hv, params.hv_d) }
        } else {
            16
        };

        sketch.hv.clone_from(&hv);

        pb.inc(1);
        pb.eta();
    });

    pb.finish_and_clear();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        n_file,
        pb.elapsed().as_secs_f32(),
        pb.per_sec(),
        // (n_file as f32 / pb.elapsed().as_secs_f32())
    );

    // Dump sketch file
    utils::dump_sketch(&all_filesketch, &params.out_file);
}

fn extract_kmer_hash(sketch: &FileSketch) -> HashSet<u64> {
    let ksize = sketch.ksize;
    let threshold = u64::MAX / sketch.scaled;
    let seed = sketch.seed;

    let mut fastx_reader = parse_fastx_file(PathBuf::from(sketch.file_str.clone()))
        .expect("Opening .fna files failed");

    let mut hash_set = HashSet::<u64>::new();
    while let Some(record) = fastx_reader.next() {
        let seqrec: needletail::parser::SequenceRecord<'_> = record.expect("invalid record");

        // normalize to make sure all the bases are consistently capitalized
        let norm_seq = seqrec.normalize(false);

        // we make a reverse complemented copy of the sequence
        let rc = norm_seq.reverse_complement();

        for (_, kmer, _) in norm_seq.canonical_kmers(ksize, &rc) {
            let h = t1ha::t1ha2_atonce(kmer, seed);

            if h < threshold {
                hash_set.insert(h);
            }
        }
    }
    hash_set
}

// #[cfg(target_arch = "x86_64")]
// unsafe fn extract_kmer_hash_avx2(file: PathBuf, sketch: &mut Sketch) {
//     let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

//     while let Some(record) = fastx_reader.next() {
//         let seqrec: needletail::parser::SequenceRecord<'_> = record.expect("invalid record");

//         // normalize to make sure all the bases are consistently capitalized
//         let norm_seq = seqrec.normalize(false);

//         let mut bitkmer_array: [i64; 4] = [0, 0, 0, 0];
//         let mut bitkmer_m256: __m256i;
//         let mut cnt: usize = 0;
//         for (_, (bit_kmer_u64, _), _) in norm_seq.bit_kmers(sketch.ksize, true) {
//             if cnt < 4 {
//                 bitkmer_array[cnt] = bit_kmer_u64 as i64;
//                 cnt += 1;
//             } else {
//                 bitkmer_m256 = _mm256_set_epi64x(
//                     bitkmer_array[0],
//                     bitkmer_array[1],
//                     bitkmer_array[2],
//                     bitkmer_array[3],
//                 );
//                 sketch.insert_kmer_u64_avx2(bitkmer_m256);
//                 cnt = 0;
//             }
//         }

//         if cnt > 0 {
//             bitkmer_m256 = match cnt {
//                 1 => _mm256_set_epi64x(bitkmer_array[0], 0, 0, 0),
//                 2 => _mm256_set_epi64x(bitkmer_array[0], bitkmer_array[1], 0, 0),
//                 3 => _mm256_set_epi64x(bitkmer_array[0], bitkmer_array[1], bitkmer_array[2], 0),
//                 _ => _mm256_set_epi64x(
//                     bitkmer_array[0],
//                     bitkmer_array[1],
//                     bitkmer_array[2],
//                     bitkmer_array[3],
//                 ),
//             };
//             sketch.insert_kmer_u64_avx2(bitkmer_m256);
//         }
//     }
// }
