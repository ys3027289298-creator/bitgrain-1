/*
 * Gaussian/Laplacian pyramid band trade-off demo.
 *
 * Self-contained: synthetic images only, no third-party libraries, no
 * network. Links the library's image_writer.c (which hosts the pyramid)
 * plus libm. Run via run.sh from the repo root.
 *
 * Reports the four requested data points:
 *   1. reconstruction error vs original (RMSE + PSNR)
 *   2. visual loss after dropping a layer (mean gradient change)
 *   3. per-layer energy share
 *   4. wall time on a 512x512 image
 * Plus the requested drop-layer-1 vs drop-layer-2 comparison and the
 * invalid-argument / too-small-image error paths.
 */
#include "image_writer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- synthetic image generators (deterministic, no deps) ---- */

static uint8_t *make_gray(uint32_t w, uint32_t h)
{
    uint8_t *img = (uint8_t *)malloc((size_t)w * h);
    if (!img) return NULL;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            /* Smooth ramp + fine checker + radial band: mixed frequencies. */
            double ramp = 0.55 * (double)x / (double)w * 255.0;
            double checker = ((x ^ y) & 1) ? 28.0 : 0.0;
            double dx = (double)x - (double)w / 2.0;
            double dy = (double)y - (double)h / 2.0;
            double radial = 30.0 * sin(sqrt(dx * dx + dy * dy) * M_PI / 6.0);
            double v = ramp + checker + radial + 20.0;
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            img[(size_t)y * w + x] = (uint8_t)v;
        }
    return img;
}

static uint8_t *make_rgb(uint32_t w, uint32_t h)
{
    uint8_t *img = (uint8_t *)malloc((size_t)w * h * 3);
    if (!img) return NULL;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            size_t i = ((size_t)y * w + x) * 3;
            img[i + 0] = (uint8_t)((x * 255u) / w);
            img[i + 1] = (uint8_t)((y * 255u) / h);
            img[i + 2] = (uint8_t)(((x ^ y) & 1) ? 200 : 40);
        }
    return img;
}

/* ---- metrics over interleaved 8-bit buffers ---- */

static double rmse(const uint8_t *a, const uint8_t *b, size_t n)
{
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return sqrt(acc / (double)n);
}

static double psnr_from_rmse(double r)
{
    if (r <= 0.0) return 99.0;
    return 10.0 * log10((255.0 * 255.0) / (r * r));
}

static void report_rebuild(const char *label, const bgp_pyramid_t *pyr,
                           const uint8_t *orig, uint32_t w, uint32_t h,
                           uint32_t ch, const uint32_t *drop, uint32_t ndrop)
{
    size_t n = (size_t)w * h * ch;
    uint8_t *recon = (uint8_t *)malloc(n);
    if (!recon) { printf("  OOM\n"); return; }

    int rc = bgp_pyramid_rebuild(pyr, drop, ndrop, recon, (uint32_t)n);
    if (rc != 0) {
        printf("  [%s] rebuild failed: %s\n", label, bgp_last_error());
        free(recon);
        return;
    }

    double r = rmse(orig, recon, n);
    double g_orig = bgp_mean_gradient(orig, w, h, ch);
    double g_rec = bgp_mean_gradient(recon, w, h, ch);
    double g_delta = fabs(g_orig - g_rec);
    double g_rel = g_orig > 0.0 ? g_delta / g_orig * 100.0 : 0.0;

    printf("  %-28s RMSE=%7.4f  PSNR=%6.2f dB | gradMean %.4f -> %.4f "
           "(|d|=%.4f, %.2f%%)\n",
           label, r, psnr_from_rmse(r), g_orig, g_rec, g_delta, g_rel);
    free(recon);
}

static void analyze(const char *title, const uint8_t *img, uint32_t w,
                    uint32_t h, uint32_t ch, uint32_t levels)
{
    printf("\n================ %s (%ux%u, %u ch, %u levels) ================\n",
           title, w, h, ch, levels);

    bgp_pyramid_t *pyr = bgp_pyramid_build(img, w, h, ch, levels);
    if (!pyr) {
        printf("build failed: %s\n", bgp_last_error());
        return;
    }

    printf("Gaussian layer sizes (G0 = source):\n  ");
    for (uint32_t l = 0; l <= levels; l++) {
        uint32_t lw = 0, lh = 0;
        bgp_gaussian_dimensions(pyr, l, &lw, &lh);
        printf("G%u=%ux%u  ", l, lw, lh);
    }
    printf("\n");

    printf("Laplacian layer energy share:\n");
    for (uint32_t l = 1; l <= levels; l++) {
        double share = 0.0;
        bgp_energy_share(pyr, l, &share);
        uint32_t lw = 0, lh = 0;
        bgp_layer_dimensions(pyr, l, &lw, &lh);
        printf("  L%u (%ux%u): %6.2f%%\n", l, lw, lh, share * 100.0);
    }

    printf("Reconstruction / drop-layer experiments:\n");
    report_rebuild("no layers dropped", pyr, img, w, h, ch, NULL, 0);
    for (uint32_t l = 1; l <= levels; l++) {
        uint32_t one = l;
        char label[40];
        snprintf(label, sizeof(label), "drop L%u only", l);
        report_rebuild(label, pyr, img, w, h, ch, &one, 1);
    }
    uint32_t drop12[2] = {1, 2};
    if (levels >= 2)
        report_rebuild("drop L1+L2", pyr, img, w, h, ch, drop12, 2);

    bgp_pyramid_free(pyr);
}

/* Timing on 512x512: full build + a rebuild, best-of-N to reduce noise. */
static void time_512(uint32_t levels, int runs)
{
    const uint32_t W = 512, H = 512, CH = 1;
    uint8_t *img = make_gray(W, H);
    uint8_t *recon = (uint8_t *)malloc((size_t)W * H);
    if (!img || !recon) { printf("OOM in timing\n"); free(img); free(recon); return; }

    printf("\n================ Timing 512x512 grayscale, %u levels "
           "(best of %d runs) ================\n", levels, runs);
    double best_build = 1e30, best_rebuild = 1e30, best_total = 1e30;
    for (int r = 0; r < runs; r++) {
        double t0 = bgp_now_ms();
        bgp_pyramid_t *pyr = bgp_pyramid_build(img, W, H, CH, levels);
        double t1 = bgp_now_ms();
        uint32_t drop = 1;
        bgp_pyramid_rebuild(pyr, &drop, 1, recon, W * H);
        double t2 = bgp_now_ms();
        if (!pyr) { printf("build failed: %s\n", bgp_last_error()); break; }
        double build_ms = t1 - t0, rebuild_ms = t2 - t1;
        if (build_ms < best_build) best_build = build_ms;
        if (rebuild_ms < best_rebuild) best_rebuild = rebuild_ms;
        if (build_ms + rebuild_ms < best_total) best_total = build_ms + rebuild_ms;
        bgp_pyramid_free(pyr);
    }
    printf("  pyramid build (Gaussian+Laplacian): %8.3f ms\n", best_build);
    printf("  one rebuild (drop L1):              %8.3f ms\n", best_rebuild);
    printf("  total build+rebuild:                %8.3f ms\n", best_total);
    free(img);
    free(recon);
}

static void error_cases(void)
{
    printf("\n================ Error handling ================\n");
    uint8_t *img = make_gray(64, 64);

    /* Illegal levels: zero and deeper than the image supports. */
    uint32_t maxlv = bgp_max_levels(64, 64);
    printf("bgp_max_levels(64x64) = %u\n", maxlv);
    bgp_pyramid_t *p0 = bgp_pyramid_build(img, 64, 64, 1, 0);
    printf("levels=0  -> %s (handle=%p)\n", bgp_last_error(), (void *)p0);
    bgp_pyramid_t *pdeep = bgp_pyramid_build(img, 64, 64, 1, maxlv + 1);
    printf("levels=%u  -> %s (handle=%p)\n", maxlv + 1,
           bgp_last_error(), (void *)pdeep);

    /* Image too small to build even one octave. */
    uint8_t tiny[16] = {0};
    bgp_pyramid_t *ptiny = bgp_pyramid_build(tiny, 4, 4, 1, 1);
    printf("4x4 image, levels=1 -> %s (handle=%p)\n",
           bgp_last_error(), (void *)ptiny);

    /* Bad channel count. */
    bgp_pyramid_t *pch = bgp_pyramid_build(img, 64, 64, 2, 2);
    printf("channels=2 -> %s (handle=%p)\n", bgp_last_error(), (void *)pch);

    /* Out-of-range dropped layer on an otherwise valid pyramid. */
    bgp_pyramid_t *ok = bgp_pyramid_build(img, 64, 64, 1, 3);
    if (ok) {
        uint8_t out[64 * 64];
        uint32_t bad = 99;
        int rc = bgp_pyramid_rebuild(ok, &bad, 1, out, sizeof(out));
        printf("rebuild drop L99 -> rc=%d (%s)\n", rc, bgp_last_error());
        bgp_pyramid_free(ok);
    }
    free(img);
}

int main(void)
{
    printf("Bitgrain Gaussian/Laplacian pyramid — band trade-off demo");

    /* Small grayscale sample, full four-metric analysis. */
    uint8_t *gray = make_gray(64, 64);
    analyze("synthetic grayscale", gray, 64, 64, 1, 3);
    free(gray);

    /* Small RGB sample to exercise the multi-channel path. */
    uint8_t *rgb = make_rgb(32, 32);
    analyze("synthetic RGB", rgb, 32, 32, 3, 2);
    free(rgb);

    /* Requested 512x512 timing data point. */
    time_512(4, 20);

    /* Validation/error paths. */
    error_cases();

    printf("\nDone.\n");
    return 0;
}
