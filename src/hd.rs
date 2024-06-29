use log::info;
use std::arch::x86_64::*;
use std::collections::HashSet;

use crate::types::FileSketch;
use rand::{RngCore, SeedableRng};
use wyhash::WyRng;

extern crate bitpacking;
use bitpacking::{BitPacker, BitPacker8x};

use rayon::prelude::*;

#[target_feature(enable = "avx2")]
pub unsafe fn encode_hash_hd_avx2(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i16> {
    let hv_d = sketch.hv_d;
    let _mm256_const_one_epi16 = _mm256_set1_epi16(1);
    let _mm256_const_zero = _mm256_setzero_si256();
    let shuffle_mask = _mm256_set_epi8(
        15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 15, 14, 7, 6, 13, 12, 5, 4, 11, 10,
        3, 2, 9, 8, 1, 0,
    );

    let mut rng_vec = vec![WyRng::default(); 4];
    // let mut rng_vec = vec![SeedableRng::seed_from_u64(0); 4];
    let mut rnd_vec: Vec<u64> = vec![0; 4];

    let num_seed = kmer_hash_set.len();
    let mut hv = vec![-(num_seed as i16); hv_d];

    let num_tail = num_seed % 4;
    let num_seed_round_4 = num_seed + (if num_tail == 0 { 0 } else { 4 - num_tail });
    let num_batch_round_4 = num_seed_round_4 / 4;
    let num_chunk = hv_d / 64;

    let mut seed_vec = Vec::from_iter(kmer_hash_set.clone());
    // padding seed_vec with size rounded by 4
    seed_vec.resize(num_seed_round_4, 0);

    // loop through all batches with seeds<=4
    for b_i in 0..num_batch_round_4 {
        // fetch seeds and load into RNG
        for j in 0..4 {
            rng_vec[j] = WyRng::seed_from_u64(seed_vec[b_i * 4 + j]);
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

                hv[i * 64 + k * 4] += _mm256_extract_epi16::<0>(hadd_) as i16;
                hv[i * 64 + k * 4 + 1] += _mm256_extract_epi16::<1>(hadd_) as i16;
                hv[i * 64 + k * 4 + 2] += _mm256_extract_epi16::<2>(hadd_) as i16;
                hv[i * 64 + k * 4 + 3] += _mm256_extract_epi16::<3>(hadd_) as i16;
            }
        }
    }
    hv
}

pub fn encode_hash_hd(kmer_hash_set: &HashSet<u64>, sketch: &FileSketch) -> Vec<i16> {
    let hv_d = sketch.hv_d;
    let seed_vec = Vec::from_iter(kmer_hash_set.clone());
    let mut hv = vec![-(kmer_hash_set.len() as i16); hv_d];

    for hash in seed_vec {
        let mut rng = WyRng::seed_from_u64(hash);

        for i in 0..(hv_d / 64) {
            let rnd_btis = rng.next_u64();

            for j in 0..64 {
                hv[i * 64 + j] += (((rnd_btis >> j) & 1) << 1) as i16;
            }
        }
    }

    hv
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_hd_sketch(sketch: &mut FileSketch, hv: &Vec<i16>) -> u8 {
    let hv_d = sketch.hv_d;

    // find the lossless quantization bit width
    let min_hv = hv.iter().min().unwrap().clone();
    let max_hv = hv.iter().max().unwrap().clone();

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

    // bit packing
    if is_x86_feature_detected!("avx2") {
        let offset: i16 = 1 << (quant_bit - 1);
        let hv_u32: Vec<u32> = hv.iter().map(|&i| (i + offset) as u32).collect();

        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let mut hv_compress_bits = vec![0u8; (quant_bit as usize) * (hv_d >> 3)];
        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.compress(
                &hv_u32[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                &mut hv_compress_bits[(bits_per_block * i)..(bits_per_block * (i + 1))],
                quant_bit as u8,
            );
        }

        sketch
            .hv
            .clone_from(&hv_compress_bits[..].align_to::<i16>().1.to_vec());
    } else {
        let len_bit_vec_u16 = (quant_bit as usize * hv_d + 16) / 16;
        let mut hv_compress_bits: Vec<i16> = vec![0; len_bit_vec_u16];
        for i in 0..(quant_bit as usize * hv_d) {
            hv_compress_bits[i / 16] |=
                ((hv[i / quant_bit as usize] >> (i % quant_bit as usize)) & 1) << (i % 16);
        }
        sketch.hv.clone_from(&hv_compress_bits);
    }

    quant_bit as u8
}

pub fn decompress_file_sketch(file_sketch: &mut Vec<FileSketch>) {
    let hv_dim = file_sketch[0].hv_d;

    info!("Decompressing sketch with HV dim={}", hv_dim);

    file_sketch.into_par_iter().for_each(|sketch| {
        let hv_decompressed = unsafe { decompress_hd_sketch(sketch) };
        sketch.hv.clone_from(&hv_decompressed);
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn decompress_hd_sketch(sketch: &mut FileSketch) -> Vec<i16> {
    let hv_d = sketch.hv_d;
    let quant_bit = sketch.hv_quant_bits;

    let mut hv_decompressed: Vec<i16> = vec![0; hv_d];

    if is_x86_feature_detected!("avx2") {
        // SIMD-based Bit Unpacking
        let bitpacker = BitPacker8x::new();
        let bits_per_block = quant_bit as usize * 32;

        let hv_u8 = sketch.hv.align_to::<u8>().1.to_vec();
        let mut _hv_decompressed: Vec<u32> = vec![0; hv_d];

        for i in 0..(hv_d / BitPacker8x::BLOCK_LEN) {
            bitpacker.decompress(
                &hv_u8[(bits_per_block * i)..(bits_per_block * (i + 1))],
                &mut _hv_decompressed[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                quant_bit,
            );
        }

        let offset: i16 = 1 << (quant_bit - 1);
        hv_decompressed.clone_from(
            &_hv_decompressed
                .into_iter()
                .map(|i| (i as i16 - offset))
                .collect(),
        );
    } else {
        // Scalar Bit Unpacking
        for i in 0..(quant_bit as usize * hv_d) {
            hv_decompressed[i / quant_bit as usize] |=
                (((sketch.hv[i / 16] >> (i % 16)) & 1) << (i % quant_bit as usize)) as i16;

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
