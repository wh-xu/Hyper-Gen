use std::{arch::x86_64::*, path::PathBuf};

use std::collections::HashSet;

use t1ha;

use serde::{Deserialize, Serialize};

// type HashSetFast<K> = HashSet<K, BuildHasherDefault<xxhash_rust::xxh3::Xxh3>>;

#[inline]
pub fn mm_hash(bytes: &[u8]) -> usize {
    let mut key = usize::from_ne_bytes(bytes.try_into().unwrap());
    key = !key.wrapping_add(key << 21); // key = (key << 21) - key - 1;
    key = key ^ key >> 24;
    key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
    key = key ^ key >> 14;
    key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
    key = key ^ key >> 28;
    key = key.wrapping_add(key << 31);
    key
}

#[inline]
pub fn mm_hash64(kmer: u64) -> u64 {
    let mut key = kmer;
    key = !key + (key << 21);
    key = key ^ key >> 24;
    key = (key + (key << 3)) + (key << 8);
    key = key ^ key >> 14;
    key = (key + (key << 2)) + (key << 4);
    key = key ^ key >> 28;
    key = key + (key << 31);
    key
}

// Use avx2 mmhash in https://github.com/bluenote-1577/skani/blob/main/src/avx2_seeding.rs
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn mm_hash64_avx2(kmer: __m256i) -> __m256i {
    let mut key = kmer;
    let s1 = _mm256_slli_epi64(key, 21);
    key = _mm256_add_epi64(key, s1);
    key = _mm256_xor_si256(key, _mm256_cmpeq_epi64(key, key));

    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 24));
    let s2 = _mm256_slli_epi64(key, 3);
    let s3 = _mm256_slli_epi64(key, 8);

    key = _mm256_add_epi64(key, s2);
    key = _mm256_add_epi64(key, s3);
    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 14));
    let s4 = _mm256_slli_epi64(key, 2);
    let s5 = _mm256_slli_epi64(key, 4);
    key = _mm256_add_epi64(key, s4);
    key = _mm256_add_epi64(key, s5);
    key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 28));

    let s6 = _mm256_slli_epi64(key, 31);
    key = _mm256_add_epi64(key, s6);

    key
}

pub struct CliParams {
    pub mode: String,
    pub path: PathBuf,
    pub path_ref_sketch: PathBuf,
    pub path_query_sketch: PathBuf,
    pub out_file: PathBuf,

    pub ksize: u8,
    pub seed: u64,
    pub sketch_method: String,
    pub canonical: bool,
    pub device: String,
    // pub min_kmer_cnt: u32,
    pub scaled: u64,
    pub hv_d: usize,
    pub ani_threshold: f32,
    pub if_compressed: bool,

    pub threads: u8,
}

pub struct SketchParams {
    pub path: PathBuf,
    pub out_file: PathBuf,
    pub sketch_method: String,
    pub canonical: bool,
    pub device: String,
    pub ksize: u8,
    pub seed: u64,
    // pub min_kmer_cnt: u32,
    pub scaled: u64,
    pub hv_d: usize,
    pub if_compressed: bool,
}

impl Default for SketchParams {
    fn default() -> Self {
        SketchParams {
            path: (PathBuf::new()),
            out_file: (PathBuf::new()),
            sketch_method: String::from("t1ha2"),
            canonical: (true),
            device: String::from("cpu"),
            ksize: (21),
            seed: (123),
            // min_kmer_cnt: (1),
            scaled: (1500),
            hv_d: (4096),
            if_compressed: (true),
        }
    }
}

impl SketchParams {
    pub fn new(params: &CliParams) -> SketchParams {
        let mut new_sketch = SketchParams::default();
        new_sketch.path = params.path.clone();
        new_sketch.out_file = params.out_file.clone();
        new_sketch.sketch_method = params.sketch_method.clone();
        new_sketch.canonical = params.canonical.clone();
        new_sketch.device = params.device.clone();
        new_sketch.ksize = params.ksize;
        new_sketch.seed = params.seed;
        // new_sketch.min_kmer_cnt = params.min_kmer_cnt;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.if_compressed = params.if_compressed;
        new_sketch
    }
}

pub struct Sketch {
    pub file_name: String,
    pub sketch_method: String,
    pub canonical: bool,
    pub ksize: u8,
    pub seed: u64,
    // pub min_kmer_cnt: u32,
    pub scaled: u64,
    pub threshold: u64,
    pub hash_set: HashSet<u64>,
    // pub hash_set: HashSetFast<u64>,
    pub hv_quant_bits: u8,
    pub hv_d: usize,
    pub hv: Vec<i16>,
    pub hv_l2_norm_sq: i32,
}

impl Default for Sketch {
    fn default() -> Self {
        Sketch {
            file_name: String::from(""),
            sketch_method: String::from("xxh3"),
            canonical: (true),
            ksize: (21),
            seed: (123),
            // min_kmer_cnt: (1),
            scaled: (2000),
            threshold: (u64::MAX / 2000),
            hash_set: HashSet::default(),
            hv_quant_bits: (0),
            hv_d: (4096),
            hv: (vec![]),
            hv_l2_norm_sq: (0),
        }
    }
}

impl Sketch {
    pub fn new(file: String, params: &SketchParams) -> Sketch {
        let mut new_sketch = Sketch::default();
        new_sketch.file_name = file;
        new_sketch.sketch_method = params.sketch_method.clone();
        new_sketch.canonical = params.canonical;
        new_sketch.ksize = params.ksize;
        new_sketch.seed = params.seed;
        // new_sketch.min_kmer_cnt = params.min_kmer_cnt;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.threshold = u64::MAX / params.scaled;
        new_sketch
    }

    pub fn insert_kmer(&mut self, kmer: &[u8]) {
        let h = match self.sketch_method.as_str() {
            "t1ha2" => t1ha::t1ha2_atonce(kmer, self.seed),
            "mmhash" => mm_hash(kmer) as u64,
            _ => t1ha::t1ha2_atonce(kmer, self.seed),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
        }
    }

    pub fn insert_kmer_u64(&mut self, kmer: u64) {
        let h = match self.sketch_method.as_str() {
            "t1ha2_64" => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
            "mmhash64" => mm_hash64(kmer),
            _ => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
        }
    }

    pub unsafe fn insert_kmer_u64_avx2(&mut self, kmer: __m256i) {
        let hash_256 = mm_hash64_avx2(kmer);

        let h1 = _mm256_extract_epi64(hash_256, 0) as u64;
        let h2 = _mm256_extract_epi64(hash_256, 1) as u64;
        let h3 = _mm256_extract_epi64(hash_256, 2) as u64;
        let h4 = _mm256_extract_epi64(hash_256, 3) as u64;

        for h in [h1, h2, h3, h4] {
            if h > 0 && h < self.threshold {
                self.hash_set.insert(h);
            }
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct FileSketch {
    pub ksize: u8,
    pub scaled: u64,
    pub hv_d: usize,
    pub hv_quant_bits_vec: Vec<u8>,
    pub hv_norm_2: Vec<i32>,
    pub file_vec: Vec<String>,
    pub hv_vec: Vec<Vec<i16>>,
}

pub struct SketchDist {
    pub path_ref_sketch: PathBuf,
    pub path_query_sketch: PathBuf,
    pub out_file: PathBuf,
    pub ksize: u8,
    pub hv_d: usize,
    pub files: Vec<(String, String)>,
    pub ani: Vec<f32>,
    pub ani_threshold: f32,
}

impl Default for SketchDist {
    fn default() -> Self {
        SketchDist {
            path_ref_sketch: (PathBuf::new()),
            path_query_sketch: (PathBuf::new()),
            out_file: (PathBuf::new()),
            ksize: (21),
            hv_d: (1024),
            files: (vec![]),
            ani: (vec![]),
            ani_threshold: (85.0),
        }
    }
}

impl SketchDist {
    pub fn new(params: &CliParams) -> SketchDist {
        let mut new_dist = SketchDist::default();
        new_dist.path_ref_sketch = params.path_ref_sketch.clone();
        new_dist.path_query_sketch = params.path_query_sketch.clone();
        new_dist.out_file = params.out_file.clone();
        new_dist.ksize = params.ksize;
        new_dist.hv_d = params.hv_d;
        new_dist.ani_threshold = params.ani_threshold;
        new_dist
    }
}
