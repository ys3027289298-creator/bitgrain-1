/* SPDX-License-Identifier: GPL-3.0-or-later */
#define _POSIX_C_SOURCE 200809L
/*
 * WebP reader verification harness.
 *
 * Uses only repository code (c/webp_io.c, c/image_loader.c style entry points)
 * plus the already-declared libwebp dependency:
 *   - decodes each sample through bitgrain's WebP loader
 *   - decodes the same bytes through a direct libwebp reference call
 *   - reports max abs error, mean abs error, PSNR and decode time
 *   - probes alpha preservation and exercises failure samples
 *
 * Build/run: tests/webp/run.sh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <webp/decode.h>

#include "webp_io.h"
#include "image_loader.h"

static uint8_t *read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_len = (size_t)n;
    return buf;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static double psnr_from_mse(double mse)
{
    if (mse <= 0.0) return 999.0;
    return 10.0 * (log10(255.0 * 255.0 / mse));
}

static void compare(const char *label, const uint8_t *a, const uint8_t *b,
                    size_t count, int channels)
{
    unsigned long max_abs = 0;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        long d = (long)a[i] - (long)b[i];
        unsigned long ad = (unsigned long)(d < 0 ? -d : d);
        if (ad > max_abs) max_abs = ad;
        sum_abs += (double)ad;
        sum_sq += (double)(d * d);
    }
    double mae = sum_abs / (double)count;
    double mse = sum_sq / (double)count;
    printf("%-18s ch=%d bytes=%-6zu maxAbs=%-3lu MAE=%-7.4f MSE=%-9.5f PSNR=%7.3f dB\n",
           label, channels, count, max_abs, mae, mse, psnr_from_mse(mse));
}

static int check_sample(const char *path, int want_alpha)
{
    size_t raw_len = 0;
    uint8_t *raw = read_file(path, &raw_len);
    if (!raw) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;

    uint32_t w = 0, h = 0;
    int has_alpha = -1;
    if (bitgrain_webp_probe_mem(raw, raw_len, &w, &h, &has_alpha) != 0) {
        fprintf(stderr, "[FAIL] %s probe: %s\n", name, bitgrain_webp_last_error());
        free(raw);
        return 1;
    }
    if ((want_alpha && has_alpha != 1) || (!want_alpha && has_alpha != 0)) {
        fprintf(stderr, "[FAIL] %s alpha flag=%d expected %d\n",
                name, has_alpha, want_alpha);
        free(raw);
        return 1;
    }

    int channels = want_alpha ? 4 : 3;
    int rw = 0, rh = 0;
    uint8_t *ref = want_alpha
        ? WebPDecodeRGBA(raw, raw_len, &rw, &rh)
        : WebPDecodeRGB(raw, raw_len, &rw, &rh);
    if (!ref) { fprintf(stderr, "[FAIL] %s reference decode\n", name); free(raw); return 1; }

    /* Warm up + timed loader calls (loader = repo WebP reader). */
    uint32_t lw = 0, lh = 0;
    uint8_t *warm = want_alpha
        ? bitgrain_load_webp_rgba_mem(raw, raw_len, &lw, &lh)
        : bitgrain_load_webp_rgb_mem(raw, raw_len, &lw, &lh);
    if (!warm) {
        fprintf(stderr, "[FAIL] %s loader: %s\n", name, bitgrain_webp_last_error());
        WebPFree(ref); free(raw); return 1;
    }
    free(warm);

    enum { RUNS = 1000 };
    double t0 = now_ms();
    uint8_t *got = NULL;
    for (int i = 0; i < RUNS; i++) {
        lw = lh = 0;
        got = want_alpha
            ? bitgrain_load_webp_rgba_mem(raw, raw_len, &lw, &lh)
            : bitgrain_load_webp_rgb_mem(raw, raw_len, &lw, &lh);
        if (!got) break;
        if (i + 1 < RUNS) free(got);
    }
    double t1 = now_ms();
    if (!got) {
        fprintf(stderr, "[FAIL] %s timed loader: %s\n", name, bitgrain_webp_last_error());
        WebPFree(ref); free(raw); return 1;
    }

    if (lw != (uint32_t)rw || lh != (uint32_t)rh) {
        fprintf(stderr, "[FAIL] %s dims %ux%u vs ref %dx%d\n",
                name, lw, lh, rw, rh);
        free(got); WebPFree(ref); free(raw); return 1;
    }

    size_t count = (size_t)rw * (size_t)rh * (size_t)channels;
    compare(name, got, ref, count, channels);
    printf("%-18s decode avg: %.4f ms over %d runs (%ux%u)\n",
           "", (t1 - t0) / (double)RUNS, RUNS, rw, rh);

    /* Alpha plane must be byte-identical to the reference when present. */
    if (want_alpha) {
        unsigned long alpha_max = 0;
        for (int y = 0; y < rh; y++)
            for (int x = 0; x < rw; x++) {
                long d = (long)got[(y * rw + x) * 4 + 3] -
                         (long)ref[(y * rw + x) * 4 + 3];
                unsigned long ad = (unsigned long)(d < 0 ? -d : d);
                if (ad > alpha_max) alpha_max = ad;
            }
        printf("%-18s alpha maxAbs=%lu (must be 0)\n", "", alpha_max);
        if (alpha_max != 0) {
            free(got); WebPFree(ref); free(raw); return 1;
        }
    }

    free(got);
    WebPFree(ref);
    free(raw);
    return 0;
}

static int expect_failure(const char *path)
{
    size_t raw_len = 0;
    uint8_t *raw = read_file(path, &raw_len);
    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;
    if (!raw) { fprintf(stderr, "[FAIL] %s unreadable\n", name); return 1; }

    uint32_t w = 0, h = 0;
    int alpha = 0;
    int probe_rc = bitgrain_webp_probe_mem(raw, raw_len, &w, &h, &alpha);
    uint8_t *dec = bitgrain_load_webp_rgba_mem(raw, raw_len, &w, &h);
    /* A truncated container may keep an intact RIFF/VP8 header (probe OK) and
     * still fail the actual decode; success criterion is a rejected decode. */
    int ok = (dec == NULL);
    printf("%-18s expected failure: probe_rc=%d pixels=%p reason=\"%s\" %s\n",
           name, probe_rc, (void *)dec, bitgrain_webp_last_error(),
           ok ? "OK" : "FAIL");
    free(dec);
    free(raw);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <samples-dir>\n", argv[0]);
        return 2;
    }
    char base[1024];
    int failures = 0;
    const char *good[][2] = {
        {"lossy_rgb.webp",    "0"},
        {"lossy_rgba.webp",   "1"},
        {"lossless_rgb.webp", "0"},
        {"lossless_rgba.webp","1"},
        {"lossy64_rgb.webp",    "0"},
        {"lossy64_rgba.webp",   "1"},
        {"lossless64_rgb.webp", "0"},
        {"lossless64_rgba.webp","1"},
    };
    for (size_t i = 0; i < sizeof(good) / sizeof(good[0]); i++) {
        snprintf(base, sizeof(base), "%s/%s", argv[1], good[i][0]);
        failures += check_sample(base, atoi(good[i][1]));
    }
    const char *bad[] = {"truncated.webp", "garbage.webp"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        snprintf(base, sizeof(base), "%s/%s", argv[1], bad[i]);
        failures += expect_failure(base);
    }
    if (failures) {
        printf("WEBP VERIFY: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("WEBP VERIFY: all checks passed\n");
    return 0;
}
