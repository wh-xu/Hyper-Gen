use std::path::PathBuf;

use std::collections::HashSet;
use std::hash::BuildHasherDefault;

use gxhash;
use t1ha;
use xxhash_rust;

type HashSetFast<V> = HashSet<V, BuildHasherDefault<xxhash_rust::xxh3::Xxh3>>;

use bloom::{self, CountingBloomFilter, ASMS};

use serde::{Deserialize, Serialize};

// #[inline]
// pub fn mm_hash64(kmer: u64) -> u64 {
//     let mut key = kmer;
//     key = !key.wrapping_add(key << 21); // key = (key << 21) - key - 1;
//     key = key ^ key >> 24;
//     key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
//     key = key ^ key >> 14;
//     key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
//     key = key ^ key >> 28;
//     key = key.wrapping_add(key << 31);
//     key
// }

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

#[inline]
pub fn mm_hash(bytes: &[u8]) -> u64 {
    let mut key = usize::from_ne_bytes(bytes.try_into().unwrap());
    key = !key.wrapping_add(key << 21); // key = (key << 21) - key - 1;
    key = key ^ key >> 24;
    key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
    key = key ^ key >> 14;
    key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
    key = key ^ key >> 28;
    key = key.wrapping_add(key << 31);
    key as u64
}

pub struct CliParams {
    pub mode: String,
    pub path: PathBuf,
    pub path_ref_sketch: PathBuf,
    pub path_query_sketch: PathBuf,
    pub out_file: PathBuf,

    pub ksize: u8,
    pub sketch_method: String,
    pub min_kmer_cnt: u32,
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
    pub ksize: u8,
    pub min_kmer_cnt: u32,
    pub scaled: u64,
    pub hv_d: usize,
    pub if_compressed: bool,
}

impl Default for SketchParams {
    fn default() -> Self {
        SketchParams {
            path: (PathBuf::new()),
            out_file: (PathBuf::new()),
            sketch_method: String::from("xxh3"),
            ksize: (21),
            min_kmer_cnt: (1),
            scaled: (800),
            hv_d: (1024),
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
        new_sketch.ksize = params.ksize;
        new_sketch.min_kmer_cnt = params.min_kmer_cnt;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.if_compressed = params.if_compressed;
        new_sketch
    }
}

pub struct Sketch {
    pub file_name: String,
    pub sketch_method: String,
    pub ksize: u8,
    pub min_kmer_cnt: u32,
    pub scaled: u64,
    pub threshold: u64,
    // pub hash_set: HashSet<u64>,
    pub hash_set: HashSetFast<u64>,
    pub cbf: CountingBloomFilter,
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
            ksize: (21),
            min_kmer_cnt: (1),
            scaled: (2000),
            threshold: (u64::MAX / 2000),
            // hash_set: HashSet::default(),
            hash_set: HashSetFast::default(),
            // hash_set: gxhash::GxHashSet::default(),
            cbf: CountingBloomFilter::with_rate(6, 0.01, 100_000),
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
        new_sketch.ksize = params.ksize;
        new_sketch.min_kmer_cnt = params.min_kmer_cnt;
        new_sketch.scaled = params.scaled;
        new_sketch.hv_d = params.hv_d;
        new_sketch.threshold = u64::MAX / params.scaled;
        new_sketch
    }

    pub fn insert_kmer(&mut self, kmer: &[u8]) {
        let h = match self.sketch_method.as_str() {
            "t1ha2" => t1ha::t1ha2_atonce(kmer, 123),
            "xxh3" => xxhash_rust::xxh3::xxh3_64_with_seed(kmer, 42),
            "gxhash" => gxhash::gxhash64(kmer, 42),
            "mmhash_xor" => mm_hash(kmer),
            _ => t1ha::t1ha2_atonce(kmer, 123),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
            // self.cbf.insert(&h);
        }
    }

    pub fn insert_kmer_u64(&mut self, kmer: u64) {
        let h = match self.sketch_method.as_str() {
            "t1ha2" => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
            "xxh3" => xxhash_rust::xxh3::xxh3_64_with_seed(&kmer.to_be_bytes(), 42),
            "gxhash" => gxhash::gxhash64(&kmer.to_be_bytes(), 42),
            "mmhash64_xor_c" => mm_hash64(kmer),
            _ => t1ha::t1ha2_atonce(&kmer.to_be_bytes(), 123),
        };

        if h < self.threshold {
            self.hash_set.insert(h);
            // self.cbf.insert(&h);
        }
    }
}

// #[derive(Serialize, Deserialize, Debug, Clone)]
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
