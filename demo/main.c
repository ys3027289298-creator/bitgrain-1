/* demo/main.c — three-stage preprocessing pipeline demo.
 *
 * Generates a small synthetic photo-like image, corrupts it with salt-and-pepper
 * + Gaussian noise, runs median -> bilateral -> unsharp with each stage
 * independently switchable, and reports:
 *
 *   1. PSNR before/after denoising (against the clean reference)
 *   2. edge-gradient retention (processed / clean Sobel energy)
 *   3. wall time of the three stages chained
 *   4. encoded .bg size change caused by preprocessing
 *
 * It also prints a per-stage comparison (median only, bilateral only,
 * sharpen only) next to the full chain, and exercises illegal parameters to
 * confirm they are rejected.
 *
 * Nothing here is part of the library. The library contribution is confined to
 * c/metrics.c.
 */
#include "preprocess_api.h"
#include "../includes/encoder.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * The library file (c/metrics.c) declares this struct itself, because
 * metrics.h may not be modified. Pin the layout on both sides: if either the
 * field order or the types drift, this demo refuses to compile instead of
 * corrupting parameters at run time.
 */
_Static_assert(sizeof(bg_pipeline_params) == 64, "params size drift");
_Static_assert(offsetof(bg_pipeline_params, median_enabled) == 0, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, median_radius) == 4, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, bilateral_enabled) == 8, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, bilateral_radius) == 12, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, bilateral_sigma_s) == 16, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, bilateral_sigma_r) == 24, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, sharpen_enabled) == 32, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, sharpen_amount) == 40, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, sharpen_radius) == 48, "field order drift");
_Static_assert(offsetof(bg_pipeline_params, sharpen_threshold) == 56, "field order drift");
_Static_assert(sizeof(bg_pipeline_timings) == 32, "timings size drift");

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IMG_W 128u
#define IMG_H 128u
#define CHANNELS 3u
#define BYTES ((size_t)IMG_W * IMG_H * CHANNELS)
#define QUALITY 85
#define TIMING_RUNS 9

/* ---------------------------------------------------------------- */
/* Deterministic synthetic image + noise                             */
/* ---------------------------------------------------------------- */

static uint32_t rng_state = 0x12345678u;
static uint32_t rng_next(void)
{
    /* xorshift32: deterministic so the report is reproducible. */
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static double rng_unit(void) { return (double)(rng_next() >> 8) / 16777216.0; }
static double rng_gauss(void)
{
    /* Box-Muller, two uniforms per call. */
    double u1 = rng_unit(), u2 = rng_unit();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static uint8_t clamp_u8(double v)
{
    long r = lrint(v);
    if (r < 0) return 0;
    if (r > 255) return 255;
    return (uint8_t)r;
}

/* Smooth gradients + hard-edged shapes, so both flat and edge regions exist. */
static void make_clean_image(uint8_t *img)
{
    for (uint32_t y = 0; y < IMG_H; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            double fx = (double)x / (double)(IMG_W - 1);
            double fy = (double)y / (double)(IMG_H - 1);
            double r = 40.0 + 170.0 * fx;
            double g = 40.0 + 150.0 * fy;
            double b = 60.0 + 90.0 * (1.0 - fx) * fy;

            /* A bright disc and a dark square create strong edges. */
            double dx = (double)x - 40.0, dy = (double)y - 38.0;
            if (dx * dx + dy * dy <= 18.0 * 18.0) { r = 235.0; g = 220.0; b = 150.0; }
            if (x >= 80 && x <= 112 && y >= 70 && y <= 112) { r = 25.0; g = 35.0; b = 55.0; }
            /* Thin structure: a light bar two pixels wide. */
            if (y >= 20 && y <= 21) { r = 250.0; g = 250.0; b = 250.0; }

            size_t i = ((size_t)y * IMG_W + x) * CHANNELS;
            img[i + 0] = clamp_u8(r);
            img[i + 1] = clamp_u8(g);
            img[i + 2] = clamp_u8(b);
        }
    }
}

/* Salt-and-pepper (0.5% each) plus additive Gaussian (sigma 16). */
static void make_noisy_image(const uint8_t *clean, uint8_t *noisy)
{
    memcpy(noisy, clean, BYTES);
    for (size_t i = 0; i < (size_t)IMG_W * IMG_H; i++) {
        double strike = rng_unit();
        if (strike < 0.005) {
            for (uint32_t c = 0; c < CHANNELS; c++) noisy[i * CHANNELS + c] = 0;
            continue;
        }
        if (strike < 0.010) {
            for (uint32_t c = 0; c < CHANNELS; c++) noisy[i * CHANNELS + c] = 255;
            continue;
        }
        for (uint32_t c = 0; c < CHANNELS; c++) {
            size_t p = i * CHANNELS + c;
            noisy[p] = clamp_u8((double)noisy[p] + 16.0 * rng_gauss());
        }
    }
}

/* ---------------------------------------------------------------- */
/* Helpers                                                           */
/* ---------------------------------------------------------------- */

static double psnr(const uint8_t *a, const uint8_t *b)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < BYTES; i++) {
        int d = (int)a[i] - (int)b[i];
        sum += (uint64_t)(d * d);
    }
    double mse = (double)sum / (double)BYTES;
    if (mse <= 0.0) return 99.0;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

/* Encode RGB to a .bg stream; returns encoded length or -1. */
static long encode_size(const uint8_t *rgb, uint8_t *scratch, uint32_t cap)
{
    int32_t out_len = 0;
    int rc = bitgrain_encode_rgb(rgb, IMG_W, IMG_H, scratch, cap, &out_len, QUALITY);
    if (rc != 0) return -1;
    return (long)out_len;
}

static void print_params(const char *label, const bg_pipeline_params *p)
{
    printf("  %-22s median=%s(r=%d)  bilateral=%s(r=%d, ss=%.1f, sr=%.1f)  "
           "sharpen=%s(amt=%.2f, r=%d, thr=%.1f)\n",
           label,
           p->median_enabled ? "on" : "off", p->median_radius,
           p->bilateral_enabled ? "on" : "off", p->bilateral_radius,
           p->bilateral_sigma_s, p->bilateral_sigma_r,
           p->sharpen_enabled ? "on" : "off", p->sharpen_amount,
           p->sharpen_radius, p->sharpen_threshold);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of(double *values, int n)
{
    qsort(values, (size_t)n, sizeof(double), cmp_double);
    return values[n / 2];
}

/* One row of the comparison table. */
typedef struct {
    const char *name;
    double psnr_in;     /* noisy vs clean (same for every row) */
    double psnr_out;    /* processed vs clean */
    double edge_ret;    /* processed edges / clean edges */
    double time_ms;     /* summed stage time */
    long   size_bytes;  /* .bg size after preprocessing */
} row_t;

static int run_variant(const char *name, const bg_pipeline_params *params,
                       const uint8_t *clean, const uint8_t *noisy,
                       uint8_t *out, uint8_t *scratch, uint32_t cap,
                       row_t *row)
{
    char err[256];
    bg_pipeline_timings t;
    memset(&t, 0, sizeof(t));

    if (bg_pipeline_run(noisy, out, IMG_W, IMG_H, CHANNELS, params, &t,
                        err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR: pipeline '%s' failed: %s\n", name, err);
        return -1;
    }

    long sz = encode_size(out, scratch, cap);
    if (sz < 0) {
        fprintf(stderr, "ERROR: encode failed for '%s'\n", name);
        return -1;
    }

    row->name = name;
    row->psnr_in = psnr(clean, noisy);
    row->psnr_out = psnr(clean, out);
    row->edge_ret = bg_edge_gradient_retention(clean, out, IMG_W, IMG_H, CHANNELS);
    row->time_ms = t.total_ms;
    row->size_bytes = sz;
    return 0;
}

static void print_row(const row_t *r, long baseline_size)
{
    double delta = baseline_size > 0
        ? 100.0 * ((double)r->size_bytes - (double)baseline_size) / (double)baseline_size
        : 0.0;
    printf("  %-18s %8.2f %8.2f %10.3f %9.2f %9ld %+8.2f%%\n",
           r->name, r->psnr_out, r->psnr_in, r->edge_ret, r->time_ms,
           r->size_bytes, delta);
}

/* ---------------------------------------------------------------- */
/* Illegal-parameter checks                                          */
/* ---------------------------------------------------------------- */

static int expect_invalid(const char *label, bg_pipeline_params *p,
                          const uint8_t *in, uint8_t *out)
{
    char err[256];
    bg_pipeline_timings t;
    int rc = bg_pipeline_run(in, out, IMG_W, IMG_H, CHANNELS, p, &t,
                             err, sizeof(err));
    if (rc == 0) {
        printf("  [FAIL] %-40s accepted (expected rejection)\n", label);
        return 1;
    }
    printf("  [ ok ] %-40s rejected: %s\n", label, err);
    return 0;
}

/* ---------------------------------------------------------------- */

int main(void)
{
    uint8_t *clean = (uint8_t *)malloc(BYTES);
    uint8_t *noisy = (uint8_t *)malloc(BYTES);
    uint8_t *out   = (uint8_t *)malloc(BYTES);
    uint32_t cap   = (uint32_t)(BYTES * 2 + 1024 * 1024);
    uint8_t *scratch = (uint8_t *)malloc(cap);

    if (!clean || !noisy || !out || !scratch) {
        fprintf(stderr, "ERROR: out of memory\n");
        return 1;
    }

    make_clean_image(clean);
    rng_state = 0x12345678u;
    make_noisy_image(clean, noisy);

    printf("=================================================================\n");
    printf(" bitgrain three-stage preprocessing demo\n");
    printf(" image: %ux%u RGB, quality %d | window/demo only\n",
           IMG_W, IMG_H, QUALITY);
    printf(" noise: salt-pepper 1%% (0.5%% each) + Gaussian sigma=16\n");
    printf("=================================================================\n\n");

    /* ---- 0. Baseline .bg sizes: clean and noisy, no preprocessing ---- */
    long size_clean = encode_size(clean, scratch, cap);
    long size_noisy = encode_size(noisy, scratch, cap);
    if (size_clean < 0 || size_noisy < 0) {
        fprintf(stderr, "ERROR: baseline encode failed\n");
        return 1;
    }

    printf("Baselines (no preprocessing)\n");
    printf("  clean  .bg: %ld bytes\n", size_clean);
    printf("  noisy  .bg: %ld bytes  (%+.2f%% vs clean)\n\n",
           size_noisy, 100.0 * ((double)size_noisy - (double)size_clean) /
                       (double)size_clean);

    /* ---- 1. Defaults + print parameters ---- */
    bg_pipeline_params def;
    bg_pipeline_defaults(&def);
    printf("Chain parameters\n");
    print_params("full chain", &def);
    printf("\n");

    /* ---- 2. Per-stage vs full chain ---- */
    bg_pipeline_params median_only = def;
    median_only.bilateral_enabled = 0;
    median_only.sharpen_enabled = 0;

    bg_pipeline_params bilateral_only = def;
    bilateral_only.median_enabled = 0;
    bilateral_only.sharpen_enabled = 0;

    bg_pipeline_params sharpen_only = def;
    sharpen_only.median_enabled = 0;
    sharpen_only.bilateral_enabled = 0;

    bg_pipeline_params no_pre = def;
    no_pre.median_enabled = 0;
    no_pre.bilateral_enabled = 0;
    no_pre.sharpen_enabled = 0;

    row_t rows[5];
    const struct { const char *name; bg_pipeline_params *p; } variants[] = {
        { "no preprocessing", &no_pre },
        { "median only",      &median_only },
        { "bilateral only",   &bilateral_only },
        { "sharpen only",     &sharpen_only },
        { "full 3-stage",     &def },
    };

    for (size_t i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
        if (run_variant(variants[i].name, variants[i].p, clean, noisy,
                        out, scratch, cap, &rows[i]) != 0)
            return 1;
    }

    printf("Comparison against the clean reference (128x128 RGB)\n");
    printf("  %-18s %8s %8s %10s %9s %9s %9s\n",
           "variant", "PSNR-out", "PSNR-in", "edge-ret", "time(ms)",
           ".bg size", "size d");
    printf("  %-18s %8s %8s %10s %9s %9s %9s\n",
           "", "(dB)", "(dB)", "", "", "(bytes)", "vs noisy");
    printf("  ------------------ -------- -------- ---------- --------- "
           "--------- --------\n");
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        print_row(&rows[i], size_noisy);
    printf("\n");

    /* ---- 3. Per-stage timings of the full chain ---- */
    /* Small image: repeat and report the median so scheduler noise does not
     * dominate a single-shot number. */
    printf("Full-chain stage timings (median of %d runs, monotonic clock)\n",
           TIMING_RUNS);
    {
        char err[256];
        bg_pipeline_timings t;
        double med[TIMING_RUNS], bil[TIMING_RUNS], sha[TIMING_RUNS], tot[TIMING_RUNS];
        for (int i = 0; i < TIMING_RUNS; i++) {
            if (bg_pipeline_run(noisy, out, IMG_W, IMG_H, CHANNELS, &def, &t,
                                err, sizeof(err)) != 0) {
                fprintf(stderr, "ERROR: timing run failed: %s\n", err);
                return 1;
            }
            med[i] = t.median_ms;
            bil[i] = t.bilateral_ms;
            sha[i] = t.sharpen_ms;
            tot[i] = t.total_ms;
        }
        printf("  median   : %8.3f ms\n", median_of(med, TIMING_RUNS));
        printf("  bilateral: %8.3f ms\n", median_of(bil, TIMING_RUNS));
        printf("  sharpen  : %8.3f ms\n", median_of(sha, TIMING_RUNS));
        printf("  total    : %8.3f ms  (sum of stages + plane split/merge)\n\n",
               median_of(tot, TIMING_RUNS));
    }

    /* ---- 4. Illegal parameters ---- */
    printf("Illegal-parameter rejection\n");
    int failures = 0;
    bg_pipeline_params bad;

    bad = def; bad.median_radius = 0;
    failures += expect_invalid("median radius = 0", &bad, noisy, out);
    bad = def; bad.median_radius = 4;
    failures += expect_invalid("median radius = 4", &bad, noisy, out);
    bad = def; bad.median_enabled = 2;
    failures += expect_invalid("median enabled = 2", &bad, noisy, out);
    bad = def; bad.bilateral_radius = 6;
    failures += expect_invalid("bilateral radius = 6", &bad, noisy, out);
    bad = def; bad.bilateral_sigma_s = 0.0;
    failures += expect_invalid("bilateral sigma_s = 0", &bad, noisy, out);
    bad = def; bad.bilateral_sigma_r = -1.0;
    failures += expect_invalid("bilateral sigma_r = -1", &bad, noisy, out);
    bad = def; bad.sharpen_amount = 3.5;
    failures += expect_invalid("sharpen amount = 3.5", &bad, noisy, out);
    bad = def; bad.sharpen_radius = 0;
    failures += expect_invalid("sharpen radius = 0", &bad, noisy, out);
    bad = def; bad.sharpen_threshold = 300.0;
    failures += expect_invalid("sharpen threshold = 300", &bad, noisy, out);
    bad = def; bad.bilateral_enabled = -1;
    failures += expect_invalid("bilateral enabled = -1", &bad, noisy, out);
    printf("\n");

    int ok = (failures == 0);

    /* ---- 5. Independent switches do work (sanity) ---- */
    {
        uint8_t *tmp = (uint8_t *)malloc(BYTES);
        if (!tmp) return 1;
        char err[256];
        bg_pipeline_timings t;
        bg_pipeline_params identity_no_pre = no_pre;
        if (bg_pipeline_run(noisy, tmp, IMG_W, IMG_H, CHANNELS,
                            &identity_no_pre, &t, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR: all-off run failed: %s\n", err);
            free(tmp);
            return 1;
        }
        int identical = (memcmp(tmp, noisy, BYTES) == 0);
        printf("All stages off leaves the image untouched: %s\n",
               identical ? "yes" : "NO");
        ok = ok && identical;
        free(tmp);
    }

    printf("\n%s\n", ok ? "DEMO OK" : "DEMO FAILED");

    free(clean);
    free(noisy);
    free(out);
    free(scratch);
    return ok ? 0 : 1;
}
