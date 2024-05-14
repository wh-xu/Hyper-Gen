use std::{arch::x86_64::*, hash::Hasher};
// use std::time::Instant;

use crate::types::{FileSketch, Sketch};
use rand::{RngCore, SeedableRng};
use rand_xoshiro;
use wyhash::WyRng;

extern crate bitpacking;
use bitpacking::{BitPacker, BitPacker8x};

use rayon::prelude::*;

#[target_feature(enable = "avx2")]
pub unsafe fn encode_hash_hd_avx2(sketch: &mut Sketch) {
    let _mm256_const_one_epi16 = _mm256_set1_epi16(1);
    let _mm256_const_zero = _mm256_setzero_si256();
    let shuffle_mask = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let mut rng_vec = vec![WyRng::default(); 4];
    // let mut rng_vec = vec![SeedableRng::seed_from_u64(0); 4];
    let mut rnd_vec: Vec<u64> = vec![0; 4];

    let num_seed = sketch.hash_set.len();
    sketch.hv = vec![-(num_seed as i16); sketch.hv_d];

    let num_tail = num_seed % 4;
    let num_seed_round_4 = num_seed + (if num_tail == 0 { 0 } else { 4 - num_tail });
    let num_batch_round_4 = num_seed_round_4 / 4;
    let num_chunk = sketch.hv_d / 64;

    let mut seed_vec = Vec::from_iter(sketch.hash_set.clone());
    // padding seed_vec with size rounded by 4
    seed_vec.resize(num_seed_round_4, 0);

    // loop through all batches with seeds<=4
    for b_i in 0..num_batch_round_4 {
        // fetch seeds and load into RNG
        for j in 0..4 {
            rng_vec[j] = WyRng::seed_from_u64(seed_vec[b_i * 4 + j]);
            // rng_vec[j] = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed_vec[b_i * 4 + j]);
        }

        // SIMD-based HV encoding
        for i in 0..num_chunk {
            // load rnd into 256b buffer
            for j in 0..4 {
                rnd_vec[j] = rng_vec[j].next_u64();
            }

            if b_i == num_batch_round_4 - 1 && num_tail > 0 {
                for j in num_tail..4 {
                    rnd_vec[j] = 0;
                }
            }

            // HD aggregation encoding
            let simd_rnd_4_shuffle = _mm256_shuffle_epi8(
                _mm256_set_epi64x(
                    rnd_vec[0] as i64,
                    rnd_vec[1] as i64,
                    rnd_vec[2] as i64,
                    rnd_vec[3] as i64,
                ),
                shuffle_mask,
            );

            for k in 0..16_usize {
                let shift_and_256 = _mm256_and_si256(
                    _mm256_srl_epi16(simd_rnd_4_shuffle, _mm_set1_epi64x(k as i64)),
                    // _mm256_slli_epi16(simd_rnd_4_shuffle, k),
                    _mm256_const_one_epi16,
                );

                let mut hadd_ = _mm256_hadd_epi16(shift_and_256, _mm256_const_zero);
                hadd_ = _mm256_permute4x64_epi64(hadd_, 0xD8);
                hadd_ = _mm256_shuffle_epi8(hadd_, shuffle_mask);
                hadd_ = _mm256_hadd_epi16(hadd_, _mm256_const_zero);
                hadd_ = _mm256_slli_epi16(hadd_, 1);

                sketch.hv[i * 64 + k * 4] += _mm256_extract_epi16::<0>(hadd_) as i16;
                sketch.hv[i * 64 + k * 4 + 1] += _mm256_extract_epi16::<1>(hadd_) as i16;
                sketch.hv[i * 64 + k * 4 + 2] += _mm256_extract_epi16::<2>(hadd_) as i16;
                sketch.hv[i * 64 + k * 4 + 3] += _mm256_extract_epi16::<3>(hadd_) as i16;
            }
        }
    }
}

pub fn encode_hash_hd(sketch: &mut Sketch) {
    sketch.hv = vec![-(sketch.hash_set.len() as i16); sketch.hv_d];

    for hash in &sketch.hash_set {
        let mut rng = WyRng::seed_from_u64(*hash);

        for i in 0..(sketch.hv_d / 64) {
            let rnd_btis = rng.next_u64();

            for j in 0..64 {
                sketch.hv[i * 64 + j] += (((rnd_btis >> j) & 1) << 1) as i16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_hd_sketch(sketch: &mut Sketch) {
    // find the lossless quantization bit width
    // let tstart = Instant::now();

    let min_hv = sketch.hv.iter().min().unwrap().clone();
    let max_hv = sketch.hv.iter().max().unwrap().clone();

    // println!("Step 0 took {:.3}s", tstart.elapsed().as_secs_f32());
    // let tstart = Instant::now();

    let mut quant_bit: i16 = 6;
    loop {
        let quant_min: i16 = -(1 << (quant_bit - 1));
        let quant_max: i16 = (1 << (quant_bit - 1)) - 1;

        if quant_min <= min_hv && quant_max >= max_hv {
            break;
        }

        if quant_bit == 16 {
            break;
        }
        quant_bit += 1;
    }
    sketch.hv_quant_bits = quant_bit as u8;

    // println!("Step 1 took {:.3}s", tstart.elapsed().as_secs_f32());
    // let tstart = Instant::now();

    // bit packing
    if is_x86_feature_detected!("avx2") {
        let offset: i16 = 1 << (quant_bit - 1);
        let hv_u32: Vec<u32> = sketch.hv.iter().map(|&i| (i + offset) as u32).collect();

        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        // println!("Step 2 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        let mut hv_compress_bits = vec![0u8; (quant_bit as usize) * (sketch.hv_d >> 3)];
        for i in 0..(sketch.hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.compress(
                &hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                &mut hv_compress_bits[(bits_per_block * i)..(bits_per_block * (i + 1))],
                quant_bit as u8,
            );
        }

        // println!("Step 3 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        sketch
            .hv
            .clone_from(&hv_compress_bits[..].align_to::<i16>().1.to_vec());

        // println!("Step 4 took {:.3}s", tstart.elapsed().as_secs_f32());
    } else {
        let len_bit_vec_u16 = (quant_bit as usize * sketch.hv_d + 16) / 16;
        let mut hv_compress_bits: Vec<i16> = vec![0; len_bit_vec_u16];
        for i in 0..(quant_bit as usize * sketch.hv_d) {
            hv_compress_bits[i / 16] |=
                ((sketch.hv[i / quant_bit as usize] >> (i % quant_bit as usize)) & 1) << (i % 16);
        }
        sketch.hv.clone_from(&hv_compress_bits);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn decompress_hd_sketch(hv: &Vec<i16>, hv_d: usize, quant_bit: u8) -> Vec<i16> {
    let mut hv_decompressed: Vec<i16> = vec![0; hv_d];

    if is_x86_feature_detected!("avx2") {
        // SIMD-based Bit Unpacking
        // let tstart = Instant::now();

        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let hv_u8 = hv.align_to::<u8>().1.to_vec();
        let mut _hv_decompressed: Vec<u32> = vec![0; hv_d];

        // println!("Step 1 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.decompress(
                &hv_u8[(bits_per_block * i)..(bits_per_block * (i + 1))],
                &mut _hv_decompressed[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                quant_bit,
            );
        }

        // println!("Step 2 took {:.3}s", tstart.elapsed().as_secs_f32());
        // let tstart = Instant::now();

        let offset: i16 = 1 << (quant_bit - 1);
        hv_decompressed.clone_from(
            &_hv_decompressed
                .into_iter()
                .map(|i| (i as i16 - offset))
                .collect(),
        );

        // println!("Step 3 took {:.3}s", tstart.elapsed().as_secs_f32());
    } else {
        // Scalar Bit Unpacking
        for i in 0..(quant_bit as usize * hv_d) {
            hv_decompressed[i / quant_bit as usize] |=
                (((hv[i / 16] >> (i % 16)) & 1) << (i % quant_bit as usize)) as i16;

            if (i + 1) % quant_bit as usize == 0 {
                hv_decompressed[i / quant_bit as usize] = {
                    if hv_decompressed[i / quant_bit as usize] > (1 << (quant_bit - 1)) {
                        hv_decompressed[i / quant_bit as usize] - (1 << quant_bit)
                    } else {
                        hv_decompressed[i / quant_bit as usize]
                    }
                };
            }
        }
    }

    hv_decompressed
}

pub fn decompress_file_sketch(file_sketch: &mut FileSketch) {
    let hv_dim = file_sketch.hv_d.clone();
    let quant_bits_vec = file_sketch.hv_quant_bits_vec.clone();
    // let hv_norm_2: Vec<i32> = file_sketch.hv_norm_2.clone();

    let index_hv: Vec<usize> = (0..file_sketch.hv_vec.len()).collect();
    let compressed_hv_vec: Vec<Vec<i16>> = file_sketch.hv_vec.clone();

    let decompressed_sketch_hv_vec: Vec<Vec<i16>> = index_hv
        .par_iter()
        .map(|i| unsafe {
            decompress_hd_sketch(&compressed_hv_vec[*i], hv_dim, quant_bits_vec[*i])
        })
        .collect();

    file_sketch.hv_vec.clone_from(&decompressed_sketch_hv_vec);
}
