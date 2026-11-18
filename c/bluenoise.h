/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * bluenoise.h -- blue-noise ordered dithering.
 *
 * Self-contained, dependency-free implementation of:
 *   - void-and-cluster threshold matrix generation (Ulichney 1993),
 *     with toroidal (wrap-around) Gaussian energy,
 *   - ordered 1-bit quantization driven by such a matrix,
 *   - a small cache so a (size, seed) matrix is generated only once.
 *
 * Matrix size must be a power of two (checked explicitly). The initial
 * sparse binary pattern (~10% of cells) is derived from a configurable
 * 64-bit seed through a deterministic splitmix64 PRNG, so equal
 * (size, seed) pairs always yield identical matrices and no OS RNG is
 * required.
 *
 * Usage:
 *   const bn_matrix_t *m = bn_get_matrix(32, 12345);          // cached
 *   bn_quantize_grayscale(m, in, w, h, 0, 0, out);
 *
 * This header is standalone: it does not include any other project header
 * and only uses the C standard library plus libm.
 */
#ifndef BITGRAIN_BLUENOISE_H
#define BITGRAIN_BLUENOISE_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define BN_OK              0
#define BN_ERR_INVALID_ARG 1
#define BN_ERR_NO_MEMORY   2

/* Supported matrix side-length range. Side must also be a power of two. */
#define BN_MIN_SIZE 2u
#define BN_MAX_SIZE 256u

/* A generated blue-noise threshold matrix. side*side cells, row-major.
 *   rank[i]       : unique permutation of 0..side^2-1 (full VAC ordering)
 *   thresholds[i] : rank normalized to 0..255 for 8-bit comparison
 * side == 2 is a degenerate case (4 cells) and still rank-unique. */
typedef struct {
    uint32_t  side;
    uint64_t  seed;
    uint32_t *rank;
    uint8_t  *thresholds;
} bn_matrix_t;

/* ------------------------------------------------------------------ */
/* Small deterministic PRNG (splitmix64).                              */
/* ------------------------------------------------------------------ */
static inline uint64_t bn_splitmix64(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline int bn_is_power_of_two(uint32_t x)
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

/* ------------------------------------------------------------------ */
/* Void-and-cluster generation.                                        */
/* ------------------------------------------------------------------ */

/* Toroidal squared distance between two cells of a side x side grid. */
static inline uint32_t bn_wrap_dist2(uint32_t x1, uint32_t y1,
                                     uint32_t x2, uint32_t y2, uint32_t side)
{
    uint32_t h = side / 2u;
    uint32_t dx = (x1 >= x2) ? (x1 - x2) : (x2 - x1);
    uint32_t dy = (y1 >= y2) ? (y1 - y2) : (y2 - y1);
    if (dx > h) dx = side - dx;
    if (dy > h) dy = side - dy;
    return dx * dx + dy * dy;
}

/* Gaussian spread (in pixels) used by the void-and-cluster energy field.
 * A fixed, narrow kernel (Ulichney's convention) rather than one scaled to
 * the grid width: a grid-proportional sigma makes distant pixels dominate the
 * energy and washes out the local neighbor repulsion that defines blue
 * noise. 1.5 keeps the strong short-range exclusion that separates dots. */
#define BN_GAUSSIAN_SIGMA 1.5

/* Gaussian kernel table for all wrapped distances of a given side.
 * Caller frees with free(). Returns NULL on allocation failure. */
static inline double *bn_build_kernel(uint32_t side)
{
    uint32_t half = side / 2u;
    uint32_t max_d2 = 2u * half * half;
    double *kernel = (double *)malloc(((size_t)max_d2 + 1u) * sizeof(double));
    if (!kernel) return NULL;

    double two_s2 = 2.0 * BN_GAUSSIAN_SIGMA * BN_GAUSSIAN_SIGMA;
    for (uint32_t d2 = 0; d2 <= max_d2; ++d2)
        kernel[d2] = exp(-(double)d2 / two_s2);
    return kernel;
}

/* Generate a matrix. Caller owns *out and must release it with
 * bn_matrix_free(). Validates side first. */
static inline int bn_generate_matrix(uint32_t side, uint64_t seed,
                                     bn_matrix_t **out)
{
    if (!out) return BN_ERR_INVALID_ARG;
    *out = NULL;
    if (side < BN_MIN_SIZE || side > BN_MAX_SIZE ||
        !bn_is_power_of_two(side))
        return BN_ERR_INVALID_ARG;

    const uint32_t n = side * side;
    /* Sparse initial occupancy: ~10%, at least one cell, never full. */
    const uint32_t initial = n >= 10u ? n / 10u : 1u;

    uint8_t     *pattern = (uint8_t *)calloc(n, sizeof(uint8_t));
    uint32_t    *rank    = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t    *removed = (uint32_t *)malloc(initial * sizeof(uint32_t));
    double      *energy  = (double *)malloc(n * sizeof(double));
    double      *kernel  = bn_build_kernel(side);
    bn_matrix_t *m       = (bn_matrix_t *)malloc(sizeof(bn_matrix_t));
    if (!pattern || !rank || !removed || !energy || !kernel || !m) {
        free(pattern); free(rank); free(removed);
        free(energy); free(kernel); free(m);
        return BN_ERR_NO_MEMORY;
    }

    /* Initial sparse binary pattern via partial Fisher-Yates on the
     * cell indices, driven by the seeded PRNG. */
    uint32_t *indices = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!indices) {
        free(pattern); free(rank); free(removed);
        free(energy); free(kernel); free(m);
        return BN_ERR_NO_MEMORY;
    }
    for (uint32_t i = 0; i < n; ++i) indices[i] = i;
    uint64_t rng = seed;
    for (uint32_t i = 0; i < initial; ++i) {
        uint32_t j = i + (uint32_t)(bn_splitmix64(&rng) % (uint64_t)(n - i));
        uint32_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
        pattern[indices[i]] = 1u;
    }
    free(indices);

    /* Initial energy: Gaussian sum of the set pixels, torus-wrapped. */
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            double e = 0.0;
            for (uint32_t j = 0; j < n; ++j) {
                if (!pattern[j]) continue;
                uint32_t jx = j % side, jy = j / side;
                e += kernel[bn_wrap_dist2(x, y, jx, jy, side)];
            }
            energy[y * side + x] = e;
        }
    }

    /* Phase 1: iteratively remove the tightest cluster (largest energy
     * among set pixels) until the sparse pattern is exhausted. */
    for (uint32_t k = 0; k < initial; ++k) {
        uint32_t best = UINT32_MAX;
        double best_e = -1.0;
        for (uint32_t i = 0; i < n; ++i) {
            /* Tightest cluster = max energy; ties settle on the lowest index
             * so near-equal energies never collapse into scan-order runs. */
            if (pattern[i] && (energy[i] > best_e ||
                               (energy[i] == best_e && i < best))) {
                best_e = energy[i];
                best = i;
            }
        }
        pattern[best] = 0u;
        removed[k] = best;
        uint32_t bx = best % side, by = best / side;
        for (uint32_t i = 0; i < n; ++i) {
            if (!pattern[i]) continue;
            uint32_t ix = i % side, iy = i / side;
            energy[i] -= kernel[bn_wrap_dist2(bx, by, ix, iy, side)];
        }
    }

    /* Phase 1 leaves the pattern empty. Its energy array is stale for the
     * cells that were never set, so restart from a zero-energy field: every
     * point placed below updates the field incrementally. */
    memset(energy, 0, n * sizeof(double));

    /* Phase 2a: re-add the removed pixels at their ORIGINAL locations in
     * reverse removal order (loosest original pixel first). They receive the
     * lowest ranks; the random, void-and-cluster-relaxed placement here is
     * what keeps the final matrix from collapsing into a regular lattice. */
    for (uint32_t k = 0; k < initial; ++k) {
        uint32_t loc = removed[initial - 1u - k];
        pattern[loc] = 1u;
        rank[loc] = k;
        uint32_t bx = loc % side, by = loc / side;
        /* Include the placed cell itself (kernel[0] = 1) so the field stays
         * identical to a full Gaussian-sum recomputation at every step. */
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t ix = i % side, iy = i / side;
            energy[i] += kernel[bn_wrap_dist2(bx, by, ix, iy, side)];
        }
    }

    /* Phase 2b: fill the sparsest remaining voids with rising ranks until
     * every cell is occupied. */
    for (uint32_t r = initial; r < n; ++r) {
        uint32_t best = UINT32_MAX;
        double best_e = 0.0;
        int found = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (pattern[i]) continue;
            /* Largest void = min energy; ascending scan already settles
             * exact ties on the lowest index deterministically. */
            if (!found || energy[i] < best_e) {
                best_e = energy[i];
                best = i;
                found = 1;
            }
        }
        pattern[best] = 1u;
        rank[best] = r;
        uint32_t bx = best % side, by = best / side;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t ix = i % side, iy = i / side;
            energy[i] += kernel[bn_wrap_dist2(bx, by, ix, iy, side)];
        }
    }

    uint8_t *thresholds = (uint8_t *)malloc(n * sizeof(uint8_t));
    if (!thresholds) {
        free(pattern); free(rank); free(removed);
        free(energy); free(kernel); free(m);
        return BN_ERR_NO_MEMORY;
    }
    for (uint32_t i = 0; i < n; ++i)
        thresholds[i] = (uint8_t)(rank[i] * 255u / (n - 1u));

    free(pattern);
    free(removed);
    free(energy);
    free(kernel);

    m->side = side;
    m->seed = seed;
    m->rank = rank;
    m->thresholds = thresholds;
    *out = m;
    return BN_OK;
}

static inline void bn_matrix_free(bn_matrix_t *m)
{
    if (!m) return;
    free(m->rank);
    free(m->thresholds);
    free(m);
}

/* ------------------------------------------------------------------ */
/* Matrix cache: identical (side, seed) reuses the generated matrix.   */
/* ------------------------------------------------------------------ */

#define BN_CACHE_CAP 8

typedef struct {
    bn_matrix_t *entries[BN_CACHE_CAP];
    uint32_t count;
} bn_cache_t;

static inline bn_cache_t *bn_cache_default(void)
{
    static bn_cache_t cache;
    return &cache;
}

/* Return a cached matrix for (side, seed), generating it on first use.
 * The returned pointer is owned by the cache; do not free it.
 * Returns NULL when side is invalid or allocation fails. */
static inline const bn_matrix_t *bn_get_matrix_cached(bn_cache_t *cache,
                                                      uint32_t side,
                                                      uint64_t seed)
{
    if (!cache) return NULL;
    if (side < BN_MIN_SIZE || side > BN_MAX_SIZE ||
        !bn_is_power_of_two(side))
        return NULL;

    for (uint32_t i = 0; i < cache->count; ++i) {
        if (cache->entries[i]->side == side &&
            cache->entries[i]->seed == seed)
            return cache->entries[i];
    }

    bn_matrix_t *m = NULL;
    if (bn_generate_matrix(side, seed, &m) != BN_OK)
        return NULL;

    if (cache->count == BN_CACHE_CAP) {
        /* Evict the oldest (slot 0) and shift the rest down. */
        bn_matrix_free(cache->entries[0]);
        memmove(&cache->entries[0], &cache->entries[1],
                (BN_CACHE_CAP - 1u) * sizeof(bn_matrix_t *));
        cache->count = BN_CACHE_CAP - 1u;
    }
    cache->entries[cache->count++] = m;
    return m;
}

/* Convenience wrapper around the process-wide default cache. */
static inline const bn_matrix_t *bn_get_matrix(uint32_t side, uint64_t seed)
{
    return bn_get_matrix_cached(bn_cache_default(), side, seed);
}

/* Release every matrix held by a cache. */
static inline void bn_cache_clear(bn_cache_t *cache)
{
    if (!cache) return;
    for (uint32_t i = 0; i < cache->count; ++i)
        bn_matrix_free(cache->entries[i]);
    cache->count = 0;
}

/* ------------------------------------------------------------------ */
/* Ordered 1-bit quantization.                                         */
/* ------------------------------------------------------------------ */

/* Dither an 8-bit grayscale plane against a blue-noise matrix.
 * Strides are in bytes; pass 0 to use width / width respectively.
 * Output pixels are 0 or 255. The matrix tiles toroidally. */
static inline int bn_quantize_grayscale(const bn_matrix_t *m,
                                        const uint8_t *in,
                                        uint32_t width, uint32_t height,
                                        size_t in_stride,
                                        size_t out_stride,
                                        uint8_t *out)
{
    if (!m || !m->thresholds || !in || !out || width == 0u || height == 0u)
        return BN_ERR_INVALID_ARG;
    if (in_stride == 0u) in_stride = width;
    if (out_stride == 0u) out_stride = width;

    const uint32_t side = m->side;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t *src = in + (size_t)y * in_stride;
        uint8_t *dst = out + (size_t)y * out_stride;
        uint32_t my = y % side;
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t t = m->thresholds[my * side + (x % side)];
            dst[x] = (src[x] > t) ? 255u : 0u;
        }
    }
    return BN_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* BITGRAIN_BLUENOISE_H */
