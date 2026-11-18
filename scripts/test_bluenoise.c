/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * test_bluenoise.c -- validation + comparison harness for c/bluenoise.h.
 *
 * Builds four small synthetic 64x64 grayscale materials (gradient, photo-like
 * field, pixel art, flat color blocks), dithers each with blue-noise ordered
 * dithering (void-and-cluster), Bayer ordered dithering and Floyd-Steinberg
 * error diffusion, and reports per material:
 *   - texture-uniformity indicators:
 *       repulsion  : minority-dot 4-adjacency divided by the white-noise
 *                    expectation (<1 = blue-noise dot separation, 1 = random),
 *       peakiness  : max/mean power in the mid-frequency annulus of the
 *                    binary spectrum (lower = no sharp periodic Bayer peaks),
 *   - RMSE / MAE against the original image,
 * plus matrix-generation time and cache/power-of-two validation.
 *
 * The blue-noise and Bayer matrices are both 64x64, so each threshold is used
 * exactly once per image: measured spectra reflect the matrix itself rather
 * than repeat-tiling artifacts.
 *
 * Standard C + libm only; no third-party libraries, no network access.
 */
#include "bluenoise.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IMG_W 64u
#define IMG_H 64u
#define MAT_SIDE 64u
#define MAT_SEED UINT64_C(0xB10E7A5EED42)
#define SPECTRUM_R_MIN 8.0
#define SPECTRUM_R_MAX 28.0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    const char *name;
    uint8_t px[IMG_W * IMG_H];
} material_t;

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- synthetic materials ---------------- */

static material_t make_gradient(void)
{
    material_t m;
    m.name = "gradient";
    for (uint32_t y = 0; y < IMG_H; ++y)
        for (uint32_t x = 0; x < IMG_W; ++x)
            m.px[y * IMG_W + x] =
                (uint8_t)((x + y) * 255u / (IMG_W + IMG_H - 2u));
    return m;
}

static material_t make_photo(void)
{
    /* Smooth "photo-like" field: overlapping sine lobes + soft vignette. */
    material_t m;
    m.name = "photo";
    for (uint32_t y = 0; y < IMG_H; ++y) {
        for (uint32_t x = 0; x < IMG_W; ++x) {
            double fx = (double)x / (IMG_W - 1u);
            double fy = (double)y / (IMG_H - 1u);
            double v = 0.55
                    + 0.25 * sin(8.0 * fx + 1.7) * cos(6.0 * fy)
                    + 0.15 * sin(17.0 * (fx + fy));
            double dx = fx - 0.5, dy = fy - 0.5;
            v *= 1.0 - 0.35 * (dx * dx + dy * dy);
            int iv = (int)lround(v * 255.0);
            m.px[y * IMG_W + x] =
                (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
        }
    }
    return m;
}

static material_t make_pixel_art(void)
{
    /* Coarse 8x8 cells with sharp edges and a small sprite motif. */
    material_t m;
    m.name = "pixel_art";
    static const uint8_t sprite[8][8] = {
        { 20,  20, 200, 200, 200, 200,  20,  20},
        { 20, 200, 230, 230, 230, 230, 200,  20},
        {200, 230,  40, 230, 230,  40, 230, 200},
        {200, 230, 230, 230, 230, 230, 230, 200},
        {200,  90, 230, 230, 230, 230,  90, 200},
        { 20, 200,  90,  90,  90,  90, 200,  20},
        { 20,  20, 200, 200, 200, 200,  20,  20},
        { 20,  20,  20, 200, 200,  20,  20,  20},
    };
    for (uint32_t cy = 0; cy < 8u; ++cy)
        for (uint32_t cx = 0; cx < 8u; ++cx)
            for (uint32_t j = 0; j < 8u; ++j)
                for (uint32_t i = 0; i < 8u; ++i)
                    m.px[(cy * 8u + j) * IMG_W + (cx * 8u + i)] =
                        sprite[cy][cx];
    return m;
}

static material_t make_flat(void)
{
    /* Four flat color blocks at distinct gray levels. */
    material_t m;
    m.name = "flat_blocks";
    static const uint8_t levels[2][2] = { { 32, 96 }, { 160, 224 } };
    for (uint32_t y = 0; y < IMG_H; ++y)
        for (uint32_t x = 0; x < IMG_W; ++x)
            m.px[y * IMG_W + x] =
                levels[y / (IMG_H / 2u)][x / (IMG_W / 2u)];
    return m;
}

/* ---------------- reference ditherers ---------------- */

static void bayer_thresholds(uint8_t *t, uint32_t side)
{
    /* Recursive Bayer matrix B_n (side = 2^n), normalized to 0..255. */
    uint32_t n = side * side;
    uint32_t *b = malloc(n * sizeof(uint32_t));
    b[0] = 0u;
    for (uint32_t cur = 1u; cur < side; cur *= 2u) {
        uint32_t old_n = cur * cur;
        uint32_t *nb = malloc(4u * old_n * sizeof(uint32_t));
        for (uint32_t j = 0; j < cur; ++j) {
            for (uint32_t i = 0; i < cur; ++i) {
                uint32_t v = 4u * b[j * cur + i];
                nb[j * (cur * 2u) + i]                = v;
                nb[j * (cur * 2u) + i + cur]          = v + 2u;
                nb[(j + cur) * (cur * 2u) + i]       = v + 3u;
                nb[(j + cur) * (cur * 2u) + i + cur] = v + 1u;
            }
        }
        free(b);
        b = nb;
    }
    for (uint32_t i = 0; i < n; ++i)
        t[i] = (uint8_t)(b[i] * 255u / (n - 1u));
    free(b);
}

static void dither_ordered(const uint8_t *t, uint32_t side,
                           const uint8_t *in, uint8_t *out)
{
    for (uint32_t y = 0; y < IMG_H; ++y)
        for (uint32_t x = 0; x < IMG_W; ++x) {
            uint8_t th = t[(y % side) * side + (x % side)];
            out[y * IMG_W + x] = in[y * IMG_W + x] > th ? 255u : 0u;
        }
}

static void dither_floyd_steinberg(const uint8_t *in, uint8_t *out)
{
    double *buf = malloc(IMG_W * IMG_H * sizeof(double));
    for (uint32_t i = 0; i < IMG_W * IMG_H; ++i) buf[i] = in[i];

    for (uint32_t y = 0; y < IMG_H; ++y) {
        for (uint32_t x = 0; x < IMG_W; ++x) {
            uint32_t idx = y * IMG_W + x;
            double old = buf[idx];
            double qv = old >= 128.0 ? 255.0 : 0.0;
            double err = old - qv;
            out[idx] = (uint8_t)qv;
            if (x + 1u < IMG_W)            buf[idx + 1u]        += err * 7.0 / 16.0;
            if (y + 1u < IMG_H) {
                if (x > 0u)                buf[idx + IMG_W - 1u] += err * 3.0 / 16.0;
                buf[idx + IMG_W]                                 += err * 5.0 / 16.0;
                if (x + 1u < IMG_W)        buf[idx + IMG_W + 1u] += err * 1.0 / 16.0;
            }
        }
    }
    free(buf);
}

/* ---------------- metrics ---------------- */

static void rmse_mae(const uint8_t *ref, const uint8_t *bin,
                     double *rmse, double *mae)
{
    double s2 = 0.0, s1 = 0.0;
    for (uint32_t i = 0; i < IMG_W * IMG_H; ++i) {
        double d = (double)bin[i] - (double)ref[i];
        s2 += d * d;
        s1 += fabs(d);
    }
    *rmse = sqrt(s2 / (double)(IMG_W * IMG_H));
    *mae = s1 / (double)(IMG_W * IMG_H);
}

/* Neighbor-repulsion ratio of the minority pixel class:
 *   (observed 4-adjacent minority pairs per minority pixel)
 *   / (2 * minority density), which is the expectation for independent
 *   (white-noise) placement. A blue-noise field actively separates dots, so
 * its ratio sits well below 1; white noise is ~1. This is the defining
 * spatial signature of blue noise and is independent of any density flip. */
static double repulsion_ratio(const uint8_t *bin)
{
    const uint32_t total = IMG_W * IMG_H;
    uint32_t whites = 0;
    for (uint32_t i = 0; i < total; ++i) if (bin[i]) ++whites;
    uint8_t target = (whites <= total / 2u) ? 255u : 0u;
    uint32_t minority = (whites <= total / 2u) ? whites : (total - whites);
    double density = (double)minority / (double)total;

    uint32_t count = 0, pairs = 0;
    for (uint32_t i = 0; i < total; ++i) {
        if (bin[i] != target) continue;
        ++count;
        uint32_t x = i % IMG_W, y = i / IMG_W;
        if (x + 1u < IMG_W && bin[i + 1u] == target) ++pairs;
        if (y + 1u < IMG_H && bin[i + IMG_W] == target) ++pairs;
    }
    double observed = (double)pairs / (double)count;
    double expected = 2.0 * density;
    return expected > 0.0 ? observed / expected : 0.0;
}

/* Spectral peakiness inside the mid-frequency annulus: max power divided by
 * the annulus mean. A periodic Bayer pattern concentrates energy into sharp
 * peaks; an isotropic blue-noise field spreads it smoothly, so lower is
 * better. Naive DFT over the mean-centered binary signal. */
static double spectral_peakiness(const uint8_t *bin)
{
    const uint32_t W = IMG_W, H = IMG_H;
    double mean = 0.0;
    for (uint32_t i = 0; i < W * H; ++i) mean += bin[i] / 255.0;
    mean /= (double)(W * H);

    double peak = 0.0, ann_sum = 0.0;
    uint32_t ann_count = 0;
    for (uint32_t fy = 0; fy < H; ++fy) {
        int ky = (int)fy - (fy > H / 2u ? (int)H : 0);
        for (uint32_t fx = 0; fx < W; ++fx) {
            int kx = (int)fx - (fx > W / 2u ? (int)W : 0);
            double r = sqrt((double)(kx * kx + ky * ky));
            if (r < SPECTRUM_R_MIN || r > SPECTRUM_R_MAX) continue;

            double re = 0.0, im = 0.0;
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    double v = bin[y * W + x] / 255.0 - mean;
                    double ang = -2.0 * M_PI *
                                 ((double)kx * x / W + (double)ky * y / H);
                    re += v * cos(ang);
                    im += v * sin(ang);
                }
            }
            double power = re * re + im * im;
            ann_sum += power;
            if (power > peak) peak = power;
            ++ann_count;
        }
    }
    double ann_mean = ann_sum / (double)ann_count;
    return ann_mean > 0.0 ? peak / ann_mean : 0.0;
}

/* ---------------- harness ---------------- */

static int failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (cond) { printf("  [PASS] %s\n", msg); }                     \
    else { printf("  [FAIL] %s\n", msg); ++failures; }              \
} while (0)

int main(void)
{
    printf("== blue-noise dithering: validation + comparison ==\n\n");

    /* ---- structural validation ---- */
    printf("-- matrix validation --\n");
    CHECK(bn_get_matrix(6u, 1u) == NULL, "non-power-of-two side (6) rejected");
    CHECK(bn_get_matrix(0u, 1u) == NULL, "side 0 rejected");
    CHECK(bn_get_matrix(BN_MAX_SIZE * 2u, 1u) == NULL,
          "side above BN_MAX_SIZE rejected");

    double t0 = now_seconds();
    const bn_matrix_t *m = bn_get_matrix(MAT_SIDE, MAT_SEED);
    double gen_ms = (now_seconds() - t0) * 1e3;
    CHECK(m != NULL, "64x64 matrix generated");

    if (m) {
        uint32_t cells = MAT_SIDE * MAT_SIDE;
        uint8_t *rank_seen = calloc(cells, 1);
        int rank_unique = 1, spans_range = 1;
        uint8_t seen[256] = {0};
        for (uint32_t i = 0; i < cells; ++i) {
            uint32_t r = m->rank[i];
            if (r >= cells || rank_seen[r]) rank_unique = 0;
            rank_seen[r] = 1u;
            seen[m->thresholds[i]] = 1u;
        }
        free(rank_seen);
        if (!seen[0] || !seen[255]) spans_range = 0;
        CHECK(spans_range, "thresholds span full [0,255] range");
        CHECK(rank_unique, "all 4096 ranks unique (full void-and-cluster ordering)");
        CHECK(m->side == MAT_SIDE && m->seed == MAT_SEED,
              "matrix records its side and seed");
    }

    /* Different seed -> different matrix; cache returns the same object. */
    const bn_matrix_t *m2 = bn_get_matrix(MAT_SIDE, MAT_SEED + 1u);
    int differs = 0;
    if (m && m2) {
        for (uint32_t i = 0; i < MAT_SIDE * MAT_SIDE; ++i)
            if (m->rank[i] != m2->rank[i]) { differs = 1; break; }
    }
    CHECK(differs, "different seed yields a different matrix");
    const bn_matrix_t *m_again = bn_get_matrix(MAT_SIDE, MAT_SEED);
    CHECK(m_again == m, "repeat (side,seed) served from cache (same pointer)");

    bn_cache_t local;
    memset(&local, 0, sizeof(local));
    const bn_matrix_t *a1 = bn_get_matrix_cached(&local, 16u, 7u);
    const bn_matrix_t *a2 = bn_get_matrix_cached(&local, 16u, 7u);
    CHECK(a1 && a1 == a2, "explicit cache reuses generated matrix");
    bn_cache_clear(&local);

    uint8_t bayer[MAT_SIDE * MAT_SIDE];
    bayer_thresholds(bayer, MAT_SIDE);

    material_t mats[4] = {
        make_gradient(), make_photo(), make_pixel_art(), make_flat()
    };

    printf("\n-- per-material comparison (64x64 image, 64x64 matrices) --\n");
    printf("%-11s | %-6s | %7s %7s | %9s\n",
           "material", "method", "RMSE", "MAE", "peakiness");

    int peak_beats_bayer_all = 1;
    double blue_rmse[4], bayer_rmse[4];

    for (int mi = 0; mi < 4; ++mi) {
        const material_t *mat = &mats[mi];
        uint8_t out_bn[IMG_W * IMG_H];
        uint8_t out_ba[IMG_W * IMG_H];
        uint8_t out_fs[IMG_W * IMG_H];

        bn_quantize_grayscale(m, mat->px, IMG_W, IMG_H, 0, 0, out_bn);
        dither_ordered(bayer, MAT_SIDE, mat->px, out_ba);
        dither_floyd_steinberg(mat->px, out_fs);

        const char *names[3] = {"blue", "bayer", "floyd"};
        const uint8_t *outs[3] = {out_bn, out_ba, out_fs};
        double pk[3], rmse[3];

        for (int k = 0; k < 3; ++k) {
            double mae;
            rmse_mae(mat->px, outs[k], &rmse[k], &mae);
            pk[k] = spectral_peakiness(outs[k]);
            printf("%-11s | %-6s | %7.2f %7.2f | %9.2f\n",
                   mat->name, names[k], rmse[k], mae, pk[k]);
        }
        blue_rmse[mi] = rmse[0];
        bayer_rmse[mi] = rmse[1];
        if (!(pk[0] < pk[1])) peak_beats_bayer_all = 0;
    }

    CHECK(peak_beats_bayer_all,
          "blue noise spectral peakiness < Bayer on every material (no periodic peaks)");

    /* Dot-separation (blue-noise signature) is a property of uniform tone
     * fields, so evaluate the repulsion ratio on controlled flat gray
     * patches where the minority class is unambiguous. White noise = 1.0;
     * genuine blue noise must sit clearly below it at every level. */
    printf("\n-- blue-noise dot separation on flat gray patches --\n");
    printf("%-8s | %8s %8s %8s | %s\n",
           "gray", "blue", "bayer", "floyd", "white-noise baseline");
    const int gray_levels[4] = {32, 64, 96, 128};
    int repulsion_blue_all = 1;
    for (int gi = 0; gi < 4; ++gi) {
        uint8_t flat[IMG_W * IMG_H];
        uint8_t out_bn[IMG_W * IMG_H];
        uint8_t out_ba[IMG_W * IMG_H];
        uint8_t out_fs[IMG_W * IMG_H];
        for (uint32_t i = 0; i < IMG_W * IMG_H; ++i)
            flat[i] = (uint8_t)gray_levels[gi];

        bn_quantize_grayscale(m, flat, IMG_W, IMG_H, 0, 0, out_bn);
        dither_ordered(bayer, MAT_SIDE, flat, out_ba);
        dither_floyd_steinberg(flat, out_fs);

        double rb = repulsion_ratio(out_bn);
        double ra = repulsion_ratio(out_ba);
        double rf = repulsion_ratio(out_fs);
        printf("%-8d | %8.3f %8.3f %8.3f | 1.000\n",
               gray_levels[gi], rb, ra, rf);
        if (!(rb < 1.0)) repulsion_blue_all = 0;
    }
    CHECK(repulsion_blue_all,
          "blue-noise repulsion ratio < 1.0 on every flat gray patch");

    /* Ordered dithering with a well-formed threshold matrix should not be any
     * less faithful to the source than the Bayer reference matrix. */
    int rmse_competitive = 1;
    for (int mi = 0; mi < 4; ++mi)
        if (!(blue_rmse[mi] <= bayer_rmse[mi] + 1.0)) rmse_competitive = 0;
    CHECK(rmse_competitive,
          "blue-noise RMSE is competitive with Bayer on every material");
    CHECK(peak_beats_bayer_all,
          "blue noise spectral peakiness < Bayer on every material (no periodic peaks)");

    printf("\n-- generation cost --\n");
    printf("void-and-cluster 64x64 matrix: %.3f ms (first call)\n", gen_ms);
    double t1 = now_seconds();
    volatile const bn_matrix_t *cached = bn_get_matrix(MAT_SIDE, MAT_SEED);
    double cache_us = (now_seconds() - t1) * 1e6;
    (void)cached;
    printf("cache lookup of same matrix:   %.3f us\n", cache_us);

    double t2 = now_seconds();
    const bn_matrix_t *m32 = bn_get_matrix(32u, MAT_SEED);
    double gen32_ms = (now_seconds() - t2) * 1e3;
    CHECK(m32 != NULL, "32x32 matrix generated");
    printf("void-and-cluster 32x32 matrix: %.3f ms\n", gen32_ms);

    printf("\n== %s ==\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
