#include "image_loader.h"
#include "webp_io.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* ------------------------------------------------------------------ */
/* PGM (P5) is a single-channel format. The forced 3/4-component       */
/* stb_image loads below used to inflate it to RGB/RGBA, which sent    */
/* grayscale input down the YCbCr encode path and made .pgm output     */
/* impossible (the PGM writer only accepts one channel). Detect P5 up  */
/* front so the existing grayscale fallback in the CLI keeps its 1ch.  */
/* Every other format keeps its previous load path byte-for-byte.      */
/* ------------------------------------------------------------------ */

static int path_ext_is_pgm(const char *path)
{
    if (!path) return 0;
    const char *dot = strrchr(path, '.');
    return dot && (strcasecmp(dot, ".pgm") == 0 || strcasecmp(dot, ".pnm") == 0);
}

static int mem_is_pgm_p5(const unsigned char *mem, size_t len)
{
    if (!mem || len < 3) return 0;
    return mem[0] == 'P' && mem[1] == '5' &&
           (mem[2] == ' ' || mem[2] == '\t' || mem[2] == '\n' || mem[2] == '\r');
}

static int file_is_pgm_p5(const char *path)
{
    if (!path_ext_is_pgm(path)) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char magic[3];
    size_t got = fread(magic, 1, sizeof(magic), f);
    fclose(f);
    return got == sizeof(magic) && mem_is_pgm_p5(magic, got);
}

/* ------------------------------------------------------------------ */
/* Smoothing pre-processing: box and Gaussian kernels, configurable    */
/* radius, optional edge preservation (weaker smoothing at edges).     */
/* No external dependencies beyond libm, which the project already     */
/* links. Routines work on channel-interleaved 8-bit images.           */
/* ------------------------------------------------------------------ */

#define BITGRAIN_SMOOTH_KERNEL_BOX 0
#define BITGRAIN_SMOOTH_KERNEL_GAUSSIAN 1

static const char *bg_smooth_error = "ok";

const char *bitgrain_smooth_last_error(void)
{
    return bg_smooth_error;
}

static int smooth_validate(const uint8_t *src, uint8_t *dst,
                           uint32_t width, uint32_t height, uint32_t channels,
                           int radius)
{
    if (!src || !dst || width == 0 || height == 0 ||
        (channels != 1 && channels != 3 && channels != 4) ||
        (uint64_t)width * height > (uint64_t)(SIZE_MAX / channels)) {
        bg_smooth_error = "invalid smoothing arguments";
        return -1;
    }
    if (radius < 0 || radius > 64) {
        bg_smooth_error = "invalid smoothing radius (allowed range 0..64)";
        return -1;
    }
    bg_smooth_error = "ok";
    return 0;
}

static double gauss_weight(int dx, int dy, double sigma)
{
    double d2 = (double)(dx * dx + dy * dy);
    return exp(-d2 / (2.0 * sigma * sigma));
}

static double edge_weight_at(const uint8_t *src, uint32_t width, uint32_t height,
                             uint32_t channels, uint32_t x, uint32_t y, uint32_t c)
{
    double gx = 0.0, gy = 0.0;
    if (x > 0 && x + 1 < width) {
        gx = (double)src[((size_t)y * width + (x + 1)) * channels + c] -
             (double)src[((size_t)y * width + (x - 1)) * channels + c];
    }
    if (y > 0 && y + 1 < height) {
        gy = (double)src[(((size_t)y + 1) * width + x) * channels + c] -
             (double)src[(((size_t)y - 1) * width + x) * channels + c];
    }
    double grad = sqrt(gx * gx + gy * gy);
    /* 1.0 on flat regions, down to 0.25 across the strongest gradients. */
    double w = 1.0 / (1.0 + grad / 64.0);
    if (w < 0.25) w = 0.25;
    return w;
}

static int smooth_apply(int kernel, const uint8_t *src, uint8_t *dst,
                        uint32_t width, uint32_t height, uint32_t channels,
                        int radius, int preserve_edges)
{
    if (smooth_validate(src, dst, width, height, channels, radius) != 0)
        return -1;
    if (radius == 0) {
        memcpy(dst, src, (size_t)width * height * channels);
        return 0;
    }

    double sigma = (radius > 1) ? (double)radius / 2.0 : 0.75;

    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                double acc = 0.0, wsum = 0.0;
                double center_edge = preserve_edges
                    ? edge_weight_at(src, width, height, channels, x, y, c) : 1.0;
                for (int dy = -radius; dy <= radius; dy++) {
                    int yy = (int)y + dy;
                    if (yy < 0) yy = 0;
                    if (yy >= (int)height) yy = (int)height - 1;
                    for (int dx = -radius; dx <= radius; dx++) {
                        int xx = (int)x + dx;
                        if (xx < 0) xx = 0;
                        if (xx >= (int)width) xx = (int)width - 1;
                        double w;
                        if (kernel == BITGRAIN_SMOOTH_KERNEL_GAUSSIAN)
                            w = gauss_weight(dx, dy, sigma);
                        else
                            w = 1.0;
                        if (preserve_edges) {
                            w *= center_edge *
                                 edge_weight_at(src, width, height, channels,
                                                (uint32_t)xx, (uint32_t)yy, c);
                        }
                        double v = (double)src[((size_t)yy * width + xx) * channels + c];
                        acc += w * v;
                        wsum += w;
                    }
                }
                long rounded = (long)lround(acc / wsum);
                if (rounded < 0) rounded = 0;
                if (rounded > 255) rounded = 255;
                dst[((size_t)y * width + x) * channels + c] = (uint8_t)rounded;
            }
        }
    }

    return 0;
}

int bitgrain_smooth_box(const uint8_t *src, uint8_t *dst,
                        uint32_t width, uint32_t height, uint32_t channels,
                        int radius, int preserve_edges)
{
    return smooth_apply(BITGRAIN_SMOOTH_KERNEL_BOX, src, dst,
                        width, height, channels, radius, preserve_edges);
}

int bitgrain_smooth_gaussian(const uint8_t *src, uint8_t *dst,
                             uint32_t width, uint32_t height, uint32_t channels,
                             int radius, int preserve_edges)
{
    return smooth_apply(BITGRAIN_SMOOTH_KERNEL_GAUSSIAN, src, dst,
                        width, height, channels, radius, preserve_edges);
}

/* ------------------------------------------------------------------ */
/* Stream reader                                                        */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_read_stream(FILE *f, size_t *out_size)
{
    size_t cap = 65536;
    size_t len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;

    size_t n;
    while ((n = fread(buf + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap) {
            cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
    }
    *out_size = len;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Grayscale                                                            */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_grayscale(const char *path,
                                  uint32_t *out_width,
                                  uint32_t *out_height)
{
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 1);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_grayscale_mem(const uint8_t *mem, size_t mem_size,
                                      uint32_t *out_width, uint32_t *out_height)
{
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 1);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

/* ------------------------------------------------------------------ */
/* RGB                                                                  */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_rgb(const char *path,
                           uint32_t *out_width,
                           uint32_t *out_height)
{
    const char *dot = strrchr(path, '.');
    if (dot && strcasecmp(dot, ".webp") == 0) {
        uint8_t *data = bitgrain_load_webp_rgb(path, out_width, out_height);
        if (data) return data;
    }
    if (file_is_pgm_p5(path)) return NULL; /* P5 is grayscale; let the 1ch loader handle it */
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 3);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_rgb_mem(const uint8_t *mem, size_t mem_size,
                                uint32_t *out_width, uint32_t *out_height)
{
    if (mem_is_pgm_p5(mem, mem_size)) return NULL; /* P5 is grayscale; let the 1ch loader handle it */
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 3);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

/* ------------------------------------------------------------------ */
/* RGBA                                                                 */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_rgba(const char *path,
                            uint32_t *out_width,
                            uint32_t *out_height)
{
    const char *dot = strrchr(path, '.');
    if (dot && strcasecmp(dot, ".webp") == 0) {
        uint8_t *data = bitgrain_load_webp_rgba(path, out_width, out_height);
        if (data) return data;
    }
    if (file_is_pgm_p5(path)) return NULL; /* P5 is grayscale; let the 1ch loader handle it */
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 4);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_rgba_mem(const uint8_t *mem, size_t mem_size,
                                 uint32_t *out_width, uint32_t *out_height)
{
    if (mem_is_pgm_p5(mem, mem_size)) return NULL; /* P5 is grayscale; let the 1ch loader handle it */
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 4);
    if (!data || w <= 0 || h <= 0) { stbi_image_free(data); return NULL; }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

void bitgrain_image_free(void *pixels)
{
    stbi_image_free(pixels);
}

/* ================================================================== */
/* One-shot smoothing evaluation harness.                              */
/* Build (no Makefile changes, all deps ship in the repo):             */
/*   gcc -std=c11 -Wall -Wextra -O2 -DBITGRAIN_SMOOTH_DEMO             */
/*       -Iincludes -Ic c/image_loader.c c/dct.c c/quant.c             */
/*       c/metrics.c c/webp_io.c rust/target/release/deps/libbitgrain.a*/
/*       -lpthread -ldl -lm $(pkg-config --cflags --libs libwebp)      */
/*       -o /tmp/bg_smooth_eval                                        */
/* The harness synthesises its own small noisy 32x32 sample, reports    */
/* PSNR vs. the un-smoothed input, edge-gradient retention, encoded     */
/* size delta, and two-run consistency for box/Gaussian at radius 1/3  */
/* with and without edge preservation, then probes invalid radii.       */
/* ================================================================== */
#ifdef BITGRAIN_SMOOTH_DEMO

#include "encoder.h"
#include "metrics.h"

#define EVAL_W 32u
#define EVAL_H 32u
#define EVAL_CH 1u
#define EVAL_QUALITY 85u

int bitgrain_smooth_box(const uint8_t *src, uint8_t *dst,
                        uint32_t width, uint32_t height, uint32_t channels,
                        int radius, int preserve_edges);
int bitgrain_smooth_gaussian(const uint8_t *src, uint8_t *dst,
                             uint32_t width, uint32_t height, uint32_t channels,
                             int radius, int preserve_edges);
const char *bitgrain_smooth_last_error(void);

static uint32_t lcg_state = 0x12345678u;
static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

/* Small synthetic sample: two flat regions split by a sharp vertical
   edge, a horizontal edge, plus deterministic +/-12 grain. */
static void make_sample(uint8_t *img)
{
    for (uint32_t y = 0; y < EVAL_H; y++) {
        for (uint32_t x = 0; x < EVAL_W; x++) {
            int base = (x >= EVAL_W / 2) ? 200 : 70;
            if (y >= EVAL_H / 2) base = base / 2 + 20;
            int grain = (int)(lcg_next() % 25) - 12;
            int v = base + grain;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            img[y * EVAL_W + x] = (uint8_t)v;
        }
    }
}

static double gradient_energy(const uint8_t *img)
{
    double sum = 0.0;
    for (uint32_t y = 1; y + 1 < EVAL_H; y++) {
        for (uint32_t x = 1; x + 1 < EVAL_W; x++) {
            double gx = (double)img[y * EVAL_W + (x + 1)] -
                        (double)img[y * EVAL_W + (x - 1)];
            double gy = (double)img[(y + 1) * EVAL_W + x] -
                        (double)img[(y - 1) * EVAL_W + x];
            sum += sqrt(gx * gx + gy * gy);
        }
    }
    return sum;
}

static int32_t encode_len(const uint8_t *pixels)
{
    size_t cap = (size_t)EVAL_W * EVAL_H * 2 + 1024 * 1024;
    uint8_t *out = (uint8_t *)malloc(cap);
    if (!out) return -1;
    int32_t len = 0;
    int rc = bitgrain_encode_grayscale(pixels, EVAL_W, EVAL_H, out,
                                       (uint32_t)cap, &len, EVAL_QUALITY);
    free(out);
    return rc == 0 ? len : -1;
}

typedef int (*smooth_fn)(const uint8_t *, uint8_t *, uint32_t, uint32_t, uint32_t, int, int);

static void run_case(const char *name, smooth_fn fn, const uint8_t *orig,
                     int radius, int preserve, double orig_energy, int32_t base_len)
{
    size_t n = (size_t)EVAL_W * EVAL_H;
    uint8_t *a = (uint8_t *)malloc(n);
    uint8_t *b = (uint8_t *)malloc(n);
    if (!a || !b) { fprintf(stderr, "oom\n"); exit(2); }

    if (fn(orig, a, EVAL_W, EVAL_H, EVAL_CH, radius, preserve) != 0) {
        printf("%-22s ERROR: %s\n", name, bitgrain_smooth_last_error());
        free(a); free(b);
        return;
    }
    if (fn(orig, b, EVAL_W, EVAL_H, EVAL_CH, radius, preserve) != 0) {
        printf("%-22s ERROR: %s\n", name, bitgrain_smooth_last_error());
        free(a); free(b);
        return;
    }

    double psnr = bitgrain_psnr(orig, a, EVAL_W, EVAL_H, EVAL_CH);
    double retain = 100.0 * gradient_energy(a) / orig_energy;
    int32_t len = encode_len(a);
    double size_pct = 100.0 * (double)len / (double)base_len;
    int consistent = (memcmp(a, b, n) == 0);

    printf("%-22s PSNR %6.2f dB | grad %6.2f%% | encoded %5d B (%+6.2f%%) | repeat %s\n",
           name, psnr, retain, len, size_pct - 100.0,
           consistent ? "identical" : "DIFFERS");
    free(a);
    free(b);
}

int main(void)
{
    if (bitgrain_set_threads(1) != 0) {
        fprintf(stderr, "bitgrain_set_threads failed\n");
        return 2;
    }

    size_t n = (size_t)EVAL_W * EVAL_H;
    uint8_t *orig = (uint8_t *)malloc(n);
    if (!orig) return 2;
    make_sample(orig);

    double orig_energy = gradient_energy(orig);
    int32_t base_len = encode_len(orig);
    printf("sample: %ux%u gray, quality=%u, baseline encode=%d B, grad energy=%.1f\n",
           EVAL_W, EVAL_H, EVAL_QUALITY, base_len, orig_energy);
    printf("--------------------- radius 1 ---------------------\n");
    run_case("box r=1",            bitgrain_smooth_box,      orig, 1, 0, orig_energy, base_len);
    run_case("box r=1 edge-keep",  bitgrain_smooth_box,      orig, 1, 1, orig_energy, base_len);
    run_case("gauss r=1",          bitgrain_smooth_gaussian, orig, 1, 0, orig_energy, base_len);
    run_case("gauss r=1 edge-keep",bitgrain_smooth_gaussian, orig, 1, 1, orig_energy, base_len);
    printf("--------------------- radius 3 ---------------------\n");
    run_case("box r=3",            bitgrain_smooth_box,      orig, 3, 0, orig_energy, base_len);
    run_case("box r=3 edge-keep",  bitgrain_smooth_box,      orig, 3, 1, orig_energy, base_len);
    run_case("gauss r=3",          bitgrain_smooth_gaussian, orig, 3, 0, orig_energy, base_len);
    run_case("gauss r=3 edge-keep",bitgrain_smooth_gaussian, orig, 3, 1, orig_energy, base_len);

    printf("--------------------- invalid radius ---------------\n");
    uint8_t tmp[EVAL_W * EVAL_H];
    int bad_values[3] = {-1, -7, 65};
    for (size_t i = 0; i < sizeof(bad_values) / sizeof(bad_values[0]); i++) {
        int r = bad_values[i];
        int rc_box = bitgrain_smooth_box(orig, tmp, EVAL_W, EVAL_H, EVAL_CH, r, 0);
        int rc_gau = bitgrain_smooth_gaussian(orig, tmp, EVAL_W, EVAL_H, EVAL_CH, r, 0);
        printf("radius=%3d box rc=%d gauss rc=%d err=\"%s\"\n",
               r, rc_box, rc_gau, bitgrain_smooth_last_error());
    }

    printf("--------------------- boundary checks --------------\n");
    int rc0 = bitgrain_smooth_box(orig, tmp, EVAL_W, EVAL_H, EVAL_CH, 0, 0);
    printf("radius=0 box rc=%d, output %s input\n",
           rc0, memcmp(tmp, orig, n) == 0 ? "equals" : "DIFFERS FROM");
    int rc_null = bitgrain_smooth_gaussian(NULL, tmp, EVAL_W, EVAL_H, EVAL_CH, 1, 0);
    printf("null src rc=%d err=\"%s\"\n", rc_null, bitgrain_smooth_last_error());
    int rc_badch = bitgrain_smooth_box(orig, tmp, EVAL_W, EVAL_H, 2, 1, 0);
    printf("channels=2 rc=%d err=\"%s\"\n", rc_badch, bitgrain_smooth_last_error());

    size_t rgb_n = (size_t)EVAL_W * EVAL_H * 3;
    uint8_t *rgb_orig = (uint8_t *)malloc(rgb_n);
    uint8_t *rgb_smooth = (uint8_t *)malloc(rgb_n);
    if (rgb_orig && rgb_smooth) {
        for (size_t i = 0; i < rgb_n; i++) rgb_orig[i] = orig[i / 3];
        int rc_rgb = bitgrain_smooth_gaussian(rgb_orig, rgb_smooth,
                                              EVAL_W, EVAL_H, 3, 1, 1);
        printf("rgb 3ch gauss r=1 edge-keep rc=%d err=\"%s\"\n",
               rc_rgb, bitgrain_smooth_last_error());
    }
    free(rgb_orig);
    free(rgb_smooth);

    free(orig);
    return 0;
}
#endif /* BITGRAIN_SMOOTH_DEMO */
