#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::default;
    use std::hash::Hasher;

    use std::time::Instant;

    use needletail::Sequence;
    use rand_xoshiro::Xoroshiro64Star;
    use rayon::iter::Enumerate;
    use wyhash::{wyrng, WyRng};
    use xxhash_rust::xxh3::xxh3_64_with_seed;

    use crate::hd;
    use crate::{
        hd::{compress_hd_sketch, decompress_hd_sketch},
        types::{FileSketch, Sketch},
    };

    #[test]
    fn test_ahash() {
        use ahash::AHasher;
        use rand::{RngCore, SeedableRng};
        use t1ha;

        let test_kmer = "AAAAAAAAAAGGGGGGGGGG";

        let start = Instant::now();
        let mut hasher = AHasher::default();
        for _ in 0..100000 {
            hasher.write(test_kmer.as_bytes());
            hasher.finish();
        }
        let duration = start.elapsed();
        println!("Time elapsed in ahash is: {:?}", duration);

        let start = Instant::now();
        let mut h = wyhash::wyhash(test_kmer.as_bytes(), 42);
        for _ in 0..100000 {
            h = wyhash::wyhash(&h.to_ne_bytes(), 42);
        }
        let duration = start.elapsed();
        println!("Time elapsed in wyhash2 is: {:?}", duration);

        let start = Instant::now();
        let mut h = t1ha::t1ha0(test_kmer.as_bytes(), 42);
        for _ in 0..100000 {
            h = t1ha::t1ha0(&h.to_ne_bytes(), 42);
        }
        let duration = start.elapsed();
        println!("Time elapsed in t1ha0 is: {:?}", duration);

        let start = Instant::now();
        let mut h = t1ha::t1ha1(test_kmer.as_bytes(), 42);
        for _ in 0..100000 {
            h = t1ha::t1ha1(&h.to_ne_bytes(), 42);
        }
        let duration = start.elapsed();
        println!("Time elapsed in t1ha1 is: {:?}", duration);

        let start = Instant::now();
        let mut h = t1ha::t1ha2_atonce(test_kmer.as_bytes(), 42);
        for _ in 0..100000 {
            h = t1ha::t1ha2_atonce(&h.to_ne_bytes(), 42);
        }
        let duration = start.elapsed();
        println!("Time elapsed in t1ha2 is: {:?}", duration);

        let start = Instant::now();
        for i in 0..10000 {
            let mut rng = Xoroshiro64Star::seed_from_u64(i);
            let mut hv = vec![0; 1024];
            for i in 0..(1024 / 64) {
                let rnd_btis = rng.next_u64();

                for j in 0..64 {
                    hv[i * 64 + j] += (((rnd_btis >> j) & 1) << 1) as i16;
                }
            }
        }
        let duration = start.elapsed();
        println!("Time elapsed in Xoroshiro64Star is: {:?}", duration);

        let start = Instant::now();
        for i in 0..10000 {
            let mut rng = WyRng::seed_from_u64(i);
            let mut hv = vec![0; 1024];
            for i in 0..(1024 / 64) {
                let rnd_btis = rng.next_u64();

                for j in 0..64 {
                    hv[i * 64 + j] += (((rnd_btis >> j) & 1) << 1) as i16;
                }
            }
        }
        let duration = start.elapsed();
        println!("Time elapsed in WyRng is: {:?}", duration);

        let start = Instant::now();
        for i in 0..10000 as u64 {
            let rng = t1ha::t1ha1(&i.to_ne_bytes(), 42);
            let mut hv = vec![0; 1024];
            for i in 0..(1024 / 64) {
                let rnd_btis = t1ha::t1ha1(&rng.to_ne_bytes(), 42);

                for j in 0..64 {
                    hv[i * 64 + j] += (((rnd_btis >> j) & 1) << 1) as i16;
                }
            }
        }
        let duration = start.elapsed();
        println!("Time elapsed in t1ha1 is: {:?}", duration);
    }

    #[test]
    fn test_kmer_bit_pack() {
        use needletail::{bitkmer, parse_fastx_file, Sequence};

        let file = "./test/test.fna";
        let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

        while let Some(record) = fastx_reader.next() {
            let seq = record.expect("invalid record");

            println!("{:?}", String::from_utf8(seq.seq().to_vec()));
            for i in seq.bit_kmers(5, true) {
                println!("{:?}\t{:?}", i, 0);
            }
        }
    }

    #[test]
    fn test_fast_xxhash() {
        use needletail::parse_fastx_file;
        use xxhash_rust::xxh3::xxh3_64;

        let file = "./test/test.fna";
        let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

        while let Some(record) = fastx_reader.next() {
            let seq = record.expect("invalid record");
            let norm_seq = seq.normalize(false);

            let rc = norm_seq.reverse_complement();

            println!("{:?}", String::from_utf8(norm_seq.to_vec()));
            for (_, kmer, _) in norm_seq.canonical_kmers(5, &rc) {
                let h = xxh3_64(kmer);
                println!("{:?}\t{:?}", String::from_utf8(kmer.to_vec()), h);
            }
        }
    }

    #[test]
    fn test_vec_norm() {
        let a = vec![1, 2, 3, -4];
        let l2_norm_sq = a.iter().fold(0, |sum: u64, &num| sum + (num * num) as u64);

        println!("L2 norm square = {}", l2_norm_sq);
    }

    fn dot_product(x: &Vec<i16>, y: &Vec<i16>) -> i32 {
        assert_eq!(x.len(), y.len());

        let mut ip: i32 = 0;
        for i in 0..x.len() {
            ip += x[i] as i32 * y[i] as i32;
        }
        ip
    }

    #[test]
    fn test_simd_hd_enc() {
        use crate::hd;

        let hv_dim = 8192;
        let mut sketch_A = Sketch::default();
        sketch_A.hv_d = hv_dim;
        for i in 0..123 {
            sketch_A.hash_set.insert(i);
        }

        let mut sketch_B = Sketch::default();
        sketch_B.hv_d = hv_dim;
        for i in 10..456 {
            sketch_B.hash_set.insert(i);
        }

        unsafe {
            hd::encode_hash_hd(&mut sketch_A);
            hd::encode_hash_hd(&mut sketch_B);
            let norm_A = dot_product(&sketch_A.hv, &sketch_A.hv);
            let norm_B = dot_product(&sketch_B.hv, &sketch_B.hv);
            println!(
                "HV norm using scalar encoding: A={:?}, B={:?}",
                norm_A, norm_B
            );
            let ip_scalar = dot_product(&sketch_A.hv, &sketch_B.hv);
            println!("HV IP using scalar encoding: {:?}", ip_scalar);

            hd::encode_hash_hd_avx2(&mut sketch_A);
            hd::encode_hash_hd_avx2(&mut sketch_B);
            let norm_A = dot_product(&sketch_A.hv, &sketch_A.hv);
            let norm_B = dot_product(&sketch_B.hv, &sketch_B.hv);
            println!(
                "\nHV norm using SIMD encoding: A={:?}, B={:?}",
                norm_A, norm_B
            );
            let ip_simd = dot_product(&sketch_A.hv, &sketch_B.hv);
            println!("HV IP using SIMD encoding: {:?}", ip_simd);

            assert_eq!(ip_scalar, ip_simd);
        }
    }

    #[test]
    fn test_bit_vec() {
        use rand::Rng;
        extern crate bitpacking;
        use bitpacking::{BitPacker, BitPacker8x};

        let mut rng = rand::thread_rng();

        let hv_dim: usize = 4096 * 1024;
        let hv: Vec<i16> = (0..hv_dim).map(|_| rng.gen_range(-600..600)).collect();

        // 1.1 Scalar Bit Packing
        let mut sketch = Sketch::default();
        sketch.hv_d = hv_dim;
        sketch.hv = hv.clone();
        let tstart = Instant::now();
        unsafe {
            hd::compress_hd_sketch(&mut sketch);
        }
        println!(
            "Scalar HV compression took {:.3}s",
            tstart.elapsed().as_secs_f32()
        );
        let hv_compressed = sketch.hv.clone();

        // 1.2 Scalar Bit Unpacking
        let tstart = Instant::now();
        let hv_decompressed: Vec<i16> =
            unsafe { hd::decompress_hd_sketch(&hv_compressed, hv_dim, sketch.hv_quant_bits) };
        println!(
            "Naive HV decompression took {:.3}s",
            tstart.elapsed().as_secs_f32()
        );
        assert_eq!(&hv_decompressed, &hv);

        let quant_bit: u8 = sketch.hv_quant_bits;
        println!("num_bits = {}", quant_bit);

        // 2.1 SIMD Bit Packing
        let hv: Vec<u32> = vec![77; hv_dim];
        let bitpacker = BitPacker8x::new();
        let tstart = Instant::now();
        let mut hv_compressed = vec![0u8; (quant_bit as usize) * (hv_dim >> 3)];
        let bits_per_block = quant_bit as usize * 32;
        for i in 0..(hv_dim / 256) {
            bitpacker.compress(
                &hv[i * 256..(i + 1) * 256],
                &mut hv_compressed[(bits_per_block * i)..(bits_per_block * (i + 1))],
                quant_bit,
            );
        }
        println!(
            "SIMD HV compression took {:.3}s",
            tstart.elapsed().as_secs_f32()
        );

        // 2.2 SIMD Bit Unpacking
        let mut hv_decompressed: Vec<u32> = vec![0; hv_dim];
        let tstart = Instant::now();
        for i in 0..(hv_dim / BitPacker8x::BLOCK_LEN) {
            bitpacker.decompress(
                &hv_compressed[(bits_per_block * i)..(bits_per_block * (i + 1))],
                &mut hv_decompressed[i * BitPacker8x::BLOCK_LEN..(i + 1) * BitPacker8x::BLOCK_LEN],
                quant_bit,
            );
        }
        println!(
            "SIMD HV decompression took {:.3}s",
            tstart.elapsed().as_secs_f32()
        );
        assert_eq!(&hv, &hv_decompressed);
    }

    #[test]
    fn test_xxhash_rng() {
        let mut rng = xxhash_rust::xxh3::Xxh3::with_seed(1);

        for i in 0..4 {
            let h = rng.digest();
            rng.write_u64(h);
            println!("{}", h);
        }
    }

    #[test]
    fn test_minmer() {
        use needletail::sequence::minimizer;
        use rust_seq2kminmers::KminmersIterator;

        let seq = b"AACTGCACTGCACTGCACTGCACACTGCACTGCACTGCACTGCACACTGCACTGCACTGACTGCACTGCACTGCACTGCACTGCCTGC";

        let iter = KminmersIterator::new(seq, 4, 7, 0.1, false).unwrap();
        for kminmer in iter {
            println!("kminmer: {:?}", kminmer);
            println!("{:?}", kminmer.mers()[0]);
        }

        // let mz = minimizer(seq, 5);
        // println!("{:?}", mz);
    }

    #[test]
    fn test_minimizer_strobemer() {
        use needletail::{bitkmer, parse_fastx_file, Sequence};
        use rust_seq2kminmers::KminmersIterator;
        use std::collections::HashSet;
        use std::time;

        fn Jaccard_to_ANI(J: f32, k: usize) -> f32 {
            let ani: f32 = 1.0 + (2.0 / (1.0 / J + 1.0)).ln() / (k as f32);
            ani
        }

        // use minimizer for sampling
        let file_query = "./test/Escherichia_coli_0_1288_GCA_000303255.LargeContigs.fna";
        // let file_ref = "./test/Escherichia_coli_2871950_GCA_000355455.LargeContigs.fna";
        let file_ref = "./test/Escherichia_coli_HVH_217__4_1022806__GCA_000459375.LargeContigs.fna";
        let mut sketch_vecs: Vec<HashSet<u64>> = vec![HashSet::<u64>::default(); 2];

        let w = 20;
        let k = 21;
        let density = 0.01;

        for (i, file) in [file_query, file_ref].iter().enumerate() {
            let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

            while let Some(record) = fastx_reader.next() {
                let seq = record.expect("invalid record");
                let seq = seq.normalize(false);

                let iter = KminmersIterator::new(seq.sequence(), k, w, density, false).unwrap();
                for kminmer in iter {
                    sketch_vecs[i].insert(kminmer.mers()[0]);
                    // println!("kminmer: {:?}", kminmer);
                    // println!("{:?}", kminmer.mers()[0]);
                }
                // println!("{:?}", String::from_utf8(seq.seq().to_vec()));
                // for i in seq.bit_kmers(5, true) {
                //     println!("{:?}\t{:?}", i, 0);
                // }
                // return;
            }
            println!("{:?}", sketch_vecs[i].len());
        }

        let i = sketch_vecs[0].intersection(&sketch_vecs[1]).count();
        let u = sketch_vecs[0].union(&sketch_vecs[1]).count();
        let J = (i as f32) / u as f32;
        let ANI = Jaccard_to_ANI(J, k);

        println!("Minmer: J = {}\t ANI = {}", J, ANI);

        //
        let scaled = 2000;
        let mut sketch_vecs: Vec<HashSet<u64>> = vec![HashSet::<u64>::default(); 2];

        let now = time::Instant::now();
        for (i, file) in [file_query, file_ref].iter().enumerate() {
            let mut fastx_reader = parse_fastx_file(&file).expect("Opening .fna files failed");

            while let Some(record) = fastx_reader.next() {
                let seq = record.expect("invalid record");
                let norm_seq = seq.normalize(false);

                // let rc = norm_seq.reverse_complement();
                // for (_, kmer, _) in norm_seq.canonical_kmers(k as u8, &rc) {
                //     let h = xxh3_64_with_seed(kmer, 1);
                //     if h < u64::MAX / scaled {
                //         sketch_vecs[i].insert(h);
                //     }
                // }

                for kmer in norm_seq.kmers(k as u8) {
                    let h = xxh3_64_with_seed(kmer, 1);
                    if h < u64::MAX / scaled {
                        sketch_vecs[i].insert(h);
                    }
                }

                // for (_, (kmer_u64, _), _) in norm_seq.bit_kmers(k as u8, true) {
                // let h = xxh3_64_with_seed(&kmer_u64.to_be_bytes(), 1);
                // if h < u64::MAX / scaled {
                //     sketch_vecs[i].insert(h);
                // }
                // }
            }
            println!("{:?}", sketch_vecs[i].len());
        }

        let elapsed = now.elapsed();

        let i = sketch_vecs[0].intersection(&sketch_vecs[1]).count();
        let u = sketch_vecs[0].union(&sketch_vecs[1]).count();
        let J = (i as f32) / u as f32;
        let ANI = Jaccard_to_ANI(J, k);

        println!(
            "kmer: J = {}\t ANI = {}, elapsed time = {:?}",
            J, ANI, elapsed
        );
        // use strobemer for sampling
    }
}
