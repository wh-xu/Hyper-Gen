#include <stdint.h>
#include "cuda_kernel.h"

extern "C" __device__ uint64_t mmhash_u64(uint64_t key) {
  key = ~key + (key << 21);
  key = key ^ key >> 24;
  key = (key + (key << 3)) + (key << 8);
  key = key ^ key >> 14;
  key = (key + (key << 2)) + (key << 4);
  key = key ^ key >> 28;
  key = key + (key << 31);
  return key;
}

extern "C" __global__ void cuda_kmer_bit_pack_mmhash(
    char *seq, const size_t n_bps, const size_t n_kmer_per_thread,
    const size_t n_hash_per_thread, const size_t ksize,
    const uint64_t threshold, const bool canonical,
    const uint8_t *seq_nt4_table_ext, uint64_t *kmer_scaled_hash) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // copy table to shared memory
  __shared__ uint8_t seq_nt4_table[256];
  if (threadIdx.x < 256) {
    seq_nt4_table[threadIdx.x] = seq_nt4_table_ext[threadIdx.x];
  }
  __syncthreads(); // wait for each thread to copy its elemenet

  // Each thread only processes n_per_thread kmers
  // BPs that each thread accesses
  size_t access_per_thread = n_kmer_per_thread + ksize - 1;

  // Each kmer starts from (tid) to () in the seq
  uint64_t cur_kmer_fwd = 0, cur_kmer_rev = 0;
  uint64_t mask = (1ULL << (ksize * 2)) - 1;
  uint64_t kmer_hash;
  size_t shift = (ksize - 1) * 2;

  size_t start_idx = tid * n_kmer_per_thread;
  size_t end_idx = start_idx + min(access_per_thread, n_bps - start_idx);

  size_t i, l, cnt = 0;
  for (i = start_idx, l = 0; i < end_idx; i++) {
    if (i < n_bps) {
      uint8_t c = seq_nt4_table[seq[i]];

      // valid base
      if (c < 4) {
        cur_kmer_fwd = ((cur_kmer_fwd << 2) | c) & mask; // forward strand
        cur_kmer_rev = cur_kmer_rev >> 2 | (uint64_t)(3 - c)
                                               << shift; // reverse strand

        if (++l >= ksize) {
          // compute kmer hash
          if (canonical) {
            kmer_hash = mmhash_u64(min(cur_kmer_fwd, cur_kmer_rev));
            // kmer_hash = min(mmhash_u64(cur_kmer_fwd), mmhash_u64(cur_kmer_rev));
          } else {
            kmer_hash = mmhash_u64(cur_kmer_fwd);
          }

          if (kmer_hash < threshold && cnt < n_hash_per_thread)
            kmer_scaled_hash[tid * n_hash_per_thread + (cnt++)] = kmer_hash;
        }
      } else {
        l = 0, cur_kmer_fwd = cur_kmer_rev = 0; // if there is an "N", restart
      }
    }
  }
}