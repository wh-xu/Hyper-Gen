#include "cuda_kernel.h"
#include <stdint.h>

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
    uint8_t *seq, const size_t n_bps, const size_t n_kmer_per_thread,
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

  // Each thread only processes n_kmer_thread kmers
  // BPs that each thread accesses
  size_t access_per_thread = n_kmer_per_thread + ksize - 1;

  // Each kmer starts from (tid) to () in the seq
  uint64_t cur_kmer_fwd = 0, cur_kmer_rev = 0;
  uint64_t mask = (1ULL << (ksize * 2)) - 1;
  uint64_t kmer_hash;
  size_t shift = (ksize - 1) * 2;

  size_t start_idx = tid * n_kmer_per_thread;
  size_t end_idx = start_idx + min(access_per_thread, n_bps - start_idx);

  size_t i, l = 0, cnt = 0;
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

static const uint64_t prime_0 = UINT64_C(0xEC99BF0D8372CAAB);
static const uint64_t prime_1 = UINT64_C(0x82434FE90EDCEF39);
static const uint64_t prime_2 = UINT64_C(0xD4F06DB99D67BE4B);
static const uint64_t prime_3 = UINT64_C(0xBD9CACC22C6E9571);
static const uint64_t prime_4 = UINT64_C(0x9C06FAF4D023E3AB);
static const uint64_t prime_5 = UINT64_C(0xC060724A8424F345);
static const uint64_t prime_6 = UINT64_C(0xCB5AF53AE3AAAC31);

typedef struct {
  uint64_t a;
  uint64_t b;
} t1ha_state256_t;

extern "C" __device__ int64_t strcmp_l(const uint8_t *s1, const uint8_t *s2,
                                       const size_t length) {
  for (size_t i = 0; i < length && *s1 == *s2; i++, ++s1, ++s2) {
  }
  return *s1 < *s2 ? -1 : 1;
}

extern "C" __device__ static inline uint64_t rot64(uint64_t v, unsigned s) {
  return (v >> s) | (v << (64 - s));
}

#ifndef add64carry_first
extern "C" __device__ static inline unsigned
add64carry_first(uint64_t base, uint64_t addend, uint64_t *sum) {
  *sum = base + addend;
  return *sum < addend;
}
#endif /* add64carry_fist */

#ifndef add64carry_last
extern "C" __device__ static inline void
add64carry_last(unsigned carry, uint64_t base, uint64_t addend, uint64_t *sum) {
  *sum = base + addend + carry;
}
#endif /* add64carry_last */

#ifndef mul_64x64_128
extern "C" __device__ static inline uint64_t mul_32x32_64(uint32_t a,
                                                          uint32_t b) {
  return a * (uint64_t)b;
}

extern "C" __device__ static inline uint64_t
mul_64x64_128(uint64_t a, uint64_t b, uint64_t *h) {
  /* performs 64x64 to 128 bit multiplication */
  const uint64_t ll = mul_32x32_64((uint32_t)a, (uint32_t)b);
  const uint64_t lh = mul_32x32_64(a >> 32, (uint32_t)b);
  const uint64_t hl = mul_32x32_64((uint32_t)a, b >> 32);
  const uint64_t hh = mul_32x32_64(a >> 32, b >> 32);

  /* Few simplification are possible here for 32-bit architectures,
   * but thus we would lost compatibility with the original 64-bit
   * version.  Think is very bad idea, because then 32-bit t1ha will
   * still (relatively) very slowly and well yet not compatible. */
  uint64_t l;
  add64carry_last(add64carry_first(ll, lh << 32, &l), hh, lh >> 32, h);
  add64carry_last(add64carry_first(l, hl << 32, &l), *h, hl >> 32, h);

  return l;
}
#endif /* mul_64x64_128() */

extern "C" __device__ inline void mixup64(uint64_t *a, uint64_t *b, uint64_t v,
                                          uint64_t prime) {
  uint64_t h;
  *a ^= mul_64x64_128(*b + v, prime, &h);
  *b += h;
}

extern "C" __device__ inline uint64_t mux64(uint64_t v, uint64_t prime) {
  uint64_t l, h;
  l = mul_64x64_128(v, prime, &h);
  return l ^ h;
}

extern "C" __device__ inline uint64_t final64(uint64_t a, uint64_t b) {
  uint64_t x = (a + rot64(b, 41)) * prime_0;
  uint64_t y = (rot64(a, 23) + b) * prime_6;
  return mux64(x ^ y, prime_5);
}

extern "C" __device__ inline uint64_t tail64_le_unaligned(const void *v,
                                                          size_t tail) {
  const uint8_t *p = (const uint8_t *)v;

  uint64_t r = 0;
  switch (tail & 7) {
  /* For most CPUs this code is better than a
   * copying for alignment and/or byte reordering. */
  case 0:
    r = p[7] << 8;
  /* fall through */
  case 7:
    r += p[6];
    r <<= 8;
  /* fall through */
  case 6:
    r += p[5];
    r <<= 8;
  /* fall through */
  case 5:
    r += p[4];
    r <<= 8;
  /* fall through */
  case 4:
    r += p[3];
    r <<= 8;
  /* fall through */
  case 3:
    r += p[2];
    r <<= 8;
  /* fall through */
  case 2:
    r += p[1];
    r <<= 8;
  /* fall through */
  case 1:
    return r + p[0];
    // #endif
  }
}

extern "C" __device__ uint64_t t1ha2_atonce(uint8_t *data, size_t length,
                                            uint64_t seed) {

  // init_ab
  t1ha_state256_t s;
  s.a = seed;
  s.b = length;

  //
  const uint64_t *v = (const uint64_t *)data;
  switch (length) {
  default:
    mixup64(&s.a, &s.b, tail64_le_unaligned(v++, 8),
            prime_4); /* fall through */
  // mixup64(&s.a, &s.b, *(v++), prime_4); /* fall through */
  case 24:
  case 23:
  case 22:
  case 21:
  case 20:
  case 19:
  case 18:
  case 17:
    mixup64(&s.b, &s.a, tail64_le_unaligned(v++, 8),
            prime_3); /* fall through */
    // mixup64(&s.b, &s.a, *(v++), prime_3); /* fall through */
  case 16:
  case 15:
  case 14:
  case 13:
  case 12:
  case 11:
  case 10:
  case 9:
    mixup64(&s.a, &s.b, tail64_le_unaligned(v++, 8),
            prime_2); /* fall through */
    // mixup64(&s.a, &s.b, *(v++), prime_2); /* fall through */
  case 8:
  case 7:
  case 6:
  case 5:
  case 4:
  case 3:
  case 2:
  case 1:
    mixup64(&s.b, &s.a, tail64_le_unaligned(v, length),
            prime_1); /* fall through */
  case 0:
    return final64(s.a, s.b);
  }
}

#define KMER_BUF_SIZE 512 + 32

extern "C" __global__ void
cuda_kmer_t1ha2(uint8_t *seq, const size_t n_bps,
                const size_t n_kmer_per_thread, const size_t n_hash_per_thread,
                const size_t ksize, const uint64_t threshold,
                const uint64_t seed, const bool canonical,
                uint64_t *kmer_scaled_hash) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread only processes n_kmer_thread kmers
  // BPs that each thread accesses
  size_t access_per_thread = n_kmer_per_thread + ksize - 1;

  // Each kmer starts from (tid) to () in the seq
  size_t start_idx = tid * n_kmer_per_thread;
  size_t end_idx = start_idx + min(access_per_thread, n_bps - start_idx);

  // copy kmers to buffer
  uint8_t buffer_bps_fwd[KMER_BUF_SIZE], buffer_bps_rev[KMER_BUF_SIZE];

  size_t cnt_ptr, cnt_hash, cnt_valid_kmer;
  uint64_t i, kmer_hash;
  for (i = start_idx, cnt_ptr = 0, cnt_hash = 0, cnt_valid_kmer = 0;
       i < end_idx; i++, cnt_ptr++) {
    if (i < n_bps) {
      uint8_t c_fwd = seq[i], c_rev;

      if (c_fwd == 'A' || c_fwd == 'a') {
        c_fwd = 'A';
        c_rev = 'T';
        cnt_valid_kmer++;
      } else if (c_fwd == 'T' || c_fwd == 't') {
        c_fwd = 'T';
        c_rev = 'A';
        cnt_valid_kmer++;
      } else if (c_fwd == 'C' || c_fwd == 'c') {
        c_fwd = 'C';
        c_rev = 'G';
        cnt_valid_kmer++;
      } else if (c_fwd == 'G' || c_fwd == 'g') {
        c_fwd = 'G';
        c_rev = 'C';
        cnt_valid_kmer++;
      } else {
        c_rev = c_fwd;
        cnt_valid_kmer = 0;
      }

      buffer_bps_fwd[cnt_ptr] = c_fwd;
      buffer_bps_rev[KMER_BUF_SIZE - cnt_ptr - 1] = c_rev;

      // compute kmer hash
      if (cnt_valid_kmer >= ksize) {
        uint8_t *ptr_fwd = buffer_bps_fwd + cnt_ptr - ksize + 1;
        uint8_t *ptr_rev = buffer_bps_rev + KMER_BUF_SIZE - 1 - cnt_ptr;

        if (canonical) {
          if (strcmp_l(ptr_fwd, ptr_rev, ksize) > 0) {
            kmer_hash = t1ha2_atonce(ptr_rev, ksize, seed);
          } else {
            kmer_hash = t1ha2_atonce(ptr_fwd, ksize, seed);
          }
        } else {
          kmer_hash = t1ha2_atonce(ptr_fwd, ksize, seed);
        }

        if (cnt_hash < n_hash_per_thread && kmer_hash < threshold)
          kmer_scaled_hash[tid * n_hash_per_thread + (cnt_hash++)] = kmer_hash;
      }
    }
  }
}
