/* clock_gettime under strict -std=c11. Must precede all headers. */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "image_writer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <stdlib.h>
#include <string.h>

int bitgrain_write_jpg(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp,
                       int quality)
{
    if (!path || !pixels) return -1;
    if (comp < 1 || comp > 4) comp = 1;
    if (comp == 2) comp = 1;  /* grey+alpha -> grayscale */
    if (quality < 1) quality = 80;
    if (quality > 100) quality = 100;

    /* JPEG has no alpha; RGBA must be converted to RGB */
    if (comp == 4) {
        size_t n = (size_t)width * (size_t)height;
        uint8_t *rgb = (uint8_t *)malloc(n * 3);
        if (!rgb) return -1;
        for (size_t i = 0; i < n; i++) {
            rgb[i * 3 + 0] = pixels[i * 4 + 0];
            rgb[i * 3 + 1] = pixels[i * 4 + 1];
            rgb[i * 3 + 2] = pixels[i * 4 + 2];
        }
        int ok = stbi_write_jpg(path, (int)width, (int)height, 3, rgb, quality) ? 0 : -1;
        free(rgb);
        return ok;
    }
    return stbi_write_jpg(path, (int)width, (int)height, comp, pixels, quality) ? 0 : -1;
}

int bitgrain_write_png(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp)
{
    if (!path || !pixels) return -1;
    if (comp != 1 && comp != 3 && comp != 4) comp = 1;
    return stbi_write_png(path, (int)width, (int)height, comp, pixels, 0) ? 0 : -1;
}

int bitgrain_write_bmp(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp)
{
    if (!path || !pixels) return -1;
    if (comp != 1 && comp != 3 && comp != 4) comp = 1;
    return stbi_write_bmp(path, (int)width, (int)height, comp, pixels) ? 0 : -1;
}

int bitgrain_write_tga(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp)
{
    if (!path || !pixels) return -1;
    if (comp != 1 && comp != 3 && comp != 4) comp = 1;
    return stbi_write_tga(path, (int)width, (int)height, comp, pixels) ? 0 : -1;
}

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <errno.h>

/* BT.601 luma for RGB/RGBA pixels that a PGM (P5, single channel) cannot store
 * directly. The .bg pipeline intentionally encodes PGM inputs as RGB (see
 * load_image() in encode_cli.c), so decoded frames arrive as 3/4 channels;
 * collapsing them here keeps PGM output available without touching the codec. */
static uint8_t bitgrain_luma_pixel(const uint8_t *p, uint32_t channels)
{
    if (channels == 4)
        return (uint8_t)((299 * p[0] + 587 * p[1] + 114 * p[2]) / 1000);
    if (channels == 3)
        return (uint8_t)((299 * p[0] + 587 * p[1] + 114 * p[2]) / 1000);
    return p[0];
}

int bitgrain_write_pgm(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height)
{
    if (!path || !pixels || width == 0 || height == 0) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%u %u\n255\n", width, height);
    size_t n = (size_t)width * (size_t)height;
    int ok = (fwrite(pixels, 1, n, f) == n);
    fclose(f);
    return ok ? 0 : -1;
}

/*
 * Multi-channel PGM sink used by the CLI .pgm branch. PGM P5 stores one
 * channel, so RGB/RGBA frames are collapsed with BT.601 luma. Kept separate
 * from bitgrain_write_pgm() to preserve its existing 1-channel signature.
 */
int bitgrain_write_pgm_planes(const char *path,
                              const uint8_t *pixels,
                              uint32_t width,
                              uint32_t height,
                              uint32_t channels)
{
    if (!path || !pixels || width == 0 || height == 0) return -1;
    if (channels != 1 && channels != 3 && channels != 4) return -1;
    if (channels == 1)
        return bitgrain_write_pgm(path, pixels, width, height);

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%u %u\n255\n", width, height);

    uint8_t *line = (uint8_t *)malloc(width);
    if (!line) { fclose(f); return -1; }
    int ok = 1;
    for (uint32_t y = 0; y < height && ok; y++) {
        const uint8_t *row = pixels + (size_t)y * width * channels;
        for (uint32_t x = 0; x < width; x++)
            line[x] = bitgrain_luma_pixel(row + (size_t)x * channels, channels);
        ok = (fwrite(line, 1, width, f) == width);
    }
    free(line);
    fclose(f);
    return ok ? 0 : -1;
}

/* ================================================================== */
/* Gaussian / Laplacian pyramid (pre-encode band trade-off analysis)   */
/*                                                                      */
/* Self-contained, libc + libm only. Operates per channel plane so the  */
/* same code serves 1/3/4-channel 8-bit images. Gaussian layers come    */
/* from repeated low-pass + 2x down-sampling; each Laplacian layer is   */
/* the difference between a Gaussian layer and the up-sampled next one. */
/* Dropping a Laplacian layer (zeroing it) then rebuilding quantifies   */
/* the visual cost of not encoding that frequency band.                 */
/* ================================================================== */

/* Smallest Gaussian layer edge allowed; below this another octave would
 * degenerate (the 5-tap kernel has no room). */
#define BGP_MIN_EDGE 4u

/* Error codes. */
enum {
    BGP_OK = 0,
    BGP_ERR_ARG = -1,      /* NULL pointer / unsupported channels */
    BGP_ERR_LEVELS = -2,   /* levels out of range for the image size */
    BGP_ERR_LAYER = -3,    /* dropped layer index out of range */
    BGP_ERR_NOMEM = -4     /* allocation failure */
};

static const char *bgp_last_message = "ok";

const char *bgp_last_error(void)
{
    return bgp_last_message;
}

/* Burt–Adelson 5-tap kernel: a = 0.4 -> {1,4,6,4,1}/16. Symmetric,
 * separable; normalized so DC gain is exactly 1. */
static const double BGP_KERNEL[5] = {
    1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0
};

typedef struct {
    uint32_t width;
    uint32_t height;
    double *data; /* width*height doubles, row-major */
} bgp_plane_t;

struct bgp_pyramid {
    uint32_t levels;   /* Laplacian levels L (1..L), Gaussian layers L+1 */
    uint32_t channels; /* 1, 3 or 4 */
    bgp_plane_t *gauss;    /* levels+1 planes per channel, gauss[0] = source */
    bgp_plane_t *lap;      /* levels planes per channel; lap[L-1] holds residual */
};

static double *bgp_alloc_doubles(size_t n)
{
    if (n == 0) return NULL;
    return (double *)calloc(n, sizeof(double));
}

/* Separable 5-tap Gaussian low-pass with edge clamp, in place via scratch. */
static void bgp_blur(const double *src, double *dst, uint32_t w, uint32_t h)
{
    double *tmp = bgp_alloc_doubles((size_t)w * h);
    if (!tmp) { memcpy(dst, src, (size_t)w * h * sizeof(double)); return; }

    /* Horizontal pass: src -> tmp */
    for (uint32_t y = 0; y < h; y++) {
        const double *row = src + (size_t)y * w;
        double *out = tmp + (size_t)y * w;
        for (uint32_t x = 0; x < w; x++) {
            double acc = 0.0;
            for (int k = -2; k <= 2; k++) {
                int64_t cx = (int64_t)x + k;
                if (cx < 0) cx = 0;
                if (cx >= (int64_t)w) cx = (int64_t)w - 1;
                acc += row[cx] * BGP_KERNEL[k + 2];
            }
            out[x] = acc;
        }
    }
    /* Vertical pass: tmp -> dst */
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            double acc = 0.0;
            for (int k = -2; k <= 2; k++) {
                int64_t cy = (int64_t)y + k;
                if (cy < 0) cy = 0;
                if (cy >= (int64_t)h) cy = (int64_t)h - 1;
                acc += tmp[cy * w + x] * BGP_KERNEL[k + 2];
            }
            dst[(size_t)y * w + x] = acc;
        }
    }
    free(tmp);
}

/* Down-sample 2x after low-pass: dst dims are ceil(src/2). */
static void bgp_reduce(const bgp_plane_t *src, bgp_plane_t *dst)
{
    double *blurred = bgp_alloc_doubles((size_t)src->width * src->height);
    if (!blurred) return;
    bgp_blur(src->data, blurred, src->width, src->height);
    for (uint32_t y = 0; y < dst->height; y++) {
        uint32_t sy = y * 2;
        if (sy >= src->height) sy = src->height - 1;
        for (uint32_t x = 0; x < dst->width; x++) {
            uint32_t sx = x * 2;
            if (sx >= src->width) sx = src->width - 1;
            dst->data[(size_t)y * dst->width + x] =
                blurred[(size_t)sy * src->width + sx];
        }
    }
    free(blurred);
}

/* Up-sample 2x with the same kernel and crop to the target (prev) size. */
static void bgp_expand(const bgp_plane_t *src, bgp_plane_t *dst)
{
    uint32_t tw = dst->width, th = dst->height;
    double *up = bgp_alloc_doubles((size_t)(src->width * 2) * (src->height * 2));
    if (!up) return;
    uint32_t uw = src->width * 2, uh = src->height * 2;

    /* Inject source samples at even coordinates, then blur (4x kernel weight
     * compensates the zero holes) — standard Burt expand. */
    for (uint32_t y = 0; y < src->height; y++)
        for (uint32_t x = 0; x < src->width; x++)
            up[(size_t)(y * 2) * uw + (x * 2)] =
                src->data[(size_t)y * src->width + x];

    double *smoothed = bgp_alloc_doubles((size_t)uw * uh);
    if (!smoothed) { free(up); return; }
    bgp_blur(up, smoothed, uw, uh);
    for (uint32_t y = 0; y < th; y++)
        for (uint32_t x = 0; x < tw; x++)
            dst->data[(size_t)y * tw + x] = 4.0 * smoothed[(size_t)y * uw + x];
    free(smoothed);
    free(up);
}

static void bgp_free_plane(bgp_plane_t *p)
{
    if (p) { free(p->data); p->data = NULL; }
}

void bgp_pyramid_free(bgp_pyramid_t *pyr)
{
    if (!pyr) return;
    if (pyr->gauss) {
        size_t planes = (size_t)(pyr->levels + 1) * pyr->channels;
        for (size_t i = 0; i < planes; i++) bgp_free_plane(&pyr->gauss[i]);
        free(pyr->gauss);
    }
    if (pyr->lap) {
        size_t planes = (size_t)pyr->levels * pyr->channels;
        for (size_t i = 0; i < planes; i++) bgp_free_plane(&pyr->lap[i]);
        free(pyr->lap);
    }
    free(pyr);
}

/* Maximum octaves whose smallest Gaussian layer stays >= BGP_MIN_EDGE. */
uint32_t bgp_max_levels(uint32_t width, uint32_t height)
{
    uint32_t levels = 0;
    uint32_t w = width, h = height;
    while ((w + 1) / 2 >= BGP_MIN_EDGE && (h + 1) / 2 >= BGP_MIN_EDGE) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

bgp_pyramid_t *bgp_pyramid_build(const uint8_t *image, uint32_t width,
                                 uint32_t height, uint32_t channels,
                                 uint32_t levels)
{
    if (!image || width == 0 || height == 0) {
        bgp_last_message = "image pointer is NULL or dimensions are zero";
        return NULL;
    }
    if (channels != 1 && channels != 3 && channels != 4) {
        bgp_last_message = "channels must be 1, 3 or 4";
        return NULL;
    }
    uint32_t max_levels = bgp_max_levels(width, height);
    if (levels == 0 || levels > max_levels) {
        bgp_last_message = "levels out of range for this image size";
        errno = (levels == 0) ? EINVAL : ERANGE;
        return NULL;
    }

    bgp_pyramid_t *pyr = (bgp_pyramid_t *)calloc(1, sizeof(*pyr));
    if (!pyr) { bgp_last_message = "out of memory"; return NULL; }
    pyr->levels = levels;
    pyr->channels = channels;

    size_t gauss_planes = (size_t)(levels + 1) * channels;
    size_t lap_planes = (size_t)levels * channels;
    pyr->gauss = (bgp_plane_t *)calloc(gauss_planes, sizeof(bgp_plane_t));
    pyr->lap = (bgp_plane_t *)calloc(lap_planes, sizeof(bgp_plane_t));
    if (!pyr->gauss || !pyr->lap) {
        bgp_pyramid_free(pyr);
        bgp_last_message = "out of memory";
        return NULL;
    }

    /* Gaussian layer 0 = source pixels, one plane per channel. */
    for (uint32_t c = 0; c < channels; c++) {
        bgp_plane_t *g0 = &pyr->gauss[c];
        g0->width = width; g0->height = height;
        g0->data = bgp_alloc_doubles((size_t)width * height);
        if (!g0->data) { bgp_pyramid_free(pyr); bgp_last_message = "out of memory"; return NULL; }
        for (size_t i = 0; i < (size_t)width * height; i++)
            g0->data[i] = (double)image[i * channels + c];
    }

    for (uint32_t lvl = 0; lvl < levels; lvl++) {
        for (uint32_t c = 0; c < channels; c++) {
            bgp_plane_t *prev = &pyr->gauss[(size_t)lvl * channels + c];
            bgp_plane_t *next = &pyr->gauss[(size_t)(lvl + 1) * channels + c];
            bgp_plane_t *lap = &pyr->lap[(size_t)lvl * channels + c];

            next->width = (prev->width + 1) / 2;
            next->height = (prev->height + 1) / 2;
            next->data = bgp_alloc_doubles((size_t)next->width * next->height);
            lap->width = prev->width;
            lap->height = prev->height;
            lap->data = bgp_alloc_doubles((size_t)prev->width * prev->height);
            if (!next->data || !lap->data) {
                bgp_pyramid_free(pyr);
                bgp_last_message = "out of memory";
                return NULL;
            }

            bgp_reduce(prev, next);

            /* Laplacian = current Gaussian - expanded next Gaussian. */
            bgp_plane_t expanded = { prev->width, prev->height,
                                     bgp_alloc_doubles((size_t)prev->width * prev->height) };
            if (!expanded.data) { bgp_pyramid_free(pyr); bgp_last_message = "out of memory"; return NULL; }
            bgp_expand(next, &expanded);
            for (size_t i = 0; i < (size_t)prev->width * prev->height; i++)
                lap->data[i] = prev->data[i] - expanded.data[i];
            free(expanded.data);
        }
    }

    bgp_last_message = "ok";
    return pyr;
}

uint32_t bgp_pyramid_levels(const bgp_pyramid_t *pyr)
{
    return pyr ? pyr->levels : 0;
}

uint32_t bgp_pyramid_channels(const bgp_pyramid_t *pyr)
{
    return pyr ? pyr->channels : 0;
}

void bgp_layer_dimensions(const bgp_pyramid_t *pyr, uint32_t layer,
                          uint32_t *out_width, uint32_t *out_height)
{
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!pyr || layer < 1 || layer > pyr->levels) return;
    /* Laplacian layer l has the dimensions of Gaussian layer l-1. */
    const bgp_plane_t *p = &pyr->lap[(size_t)(layer - 1) * pyr->channels];
    if (out_width) *out_width = p->width;
    if (out_height) *out_height = p->height;
}

/* Gaussian layer g in [0..levels]; G0 is the source-resolution layer. */
void bgp_gaussian_dimensions(const bgp_pyramid_t *pyr, uint32_t g,
                             uint32_t *out_width, uint32_t *out_height)
{
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!pyr || g > pyr->levels) return;
    const bgp_plane_t *p = &pyr->gauss[(size_t)g * pyr->channels];
    if (out_width) *out_width = p->width;
    if (out_height) *out_height = p->height;
}

/* Sum of squared values across all channels of one Laplacian layer. */
static double bgp_layer_energy(const bgp_pyramid_t *pyr, uint32_t layer)
{
    double e = 0.0;
    for (uint32_t c = 0; c < pyr->channels; c++) {
        const bgp_plane_t *p = &pyr->lap[(size_t)(layer - 1) * pyr->channels + c];
        for (size_t i = 0; i < (size_t)p->width * p->height; i++)
            e += p->data[i] * p->data[i];
    }
    return e;
}

int bgp_energy_share(const bgp_pyramid_t *pyr, uint32_t layer, double *out_share)
{
    if (out_share) *out_share = 0.0;
    if (!pyr || !out_share) { bgp_last_message = "NULL pyramid or output"; return BGP_ERR_ARG; }
    if (layer < 1 || layer > pyr->levels) {
        bgp_last_message = "layer index out of range"; return BGP_ERR_LAYER;
    }
    double total = 0.0;
    for (uint32_t l = 1; l <= pyr->levels; l++) total += bgp_layer_energy(pyr, l);
    if (total <= 0.0) { *out_share = 0.0; return BGP_OK; }
    *out_share = bgp_layer_energy(pyr, layer) / total;
    return BGP_OK;
}

/*
 * Rebuild the image. Each entry of drop_layers in [1..levels] zeroes that
 * Laplacian layer (all channels). The coarsest Gaussian residual is always
 * kept. Output is 8-bit clamped pixels in the original interleaved layout.
 */
int bgp_pyramid_rebuild(const bgp_pyramid_t *pyr, const uint32_t *drop_layers,
                        uint32_t drop_count, uint8_t *out_pixels,
                        uint32_t out_capacity)
{
    if (!pyr || !out_pixels) { bgp_last_message = "NULL pyramid or output"; return BGP_ERR_ARG; }
    for (uint32_t i = 0; i < drop_count; i++) {
        if (drop_layers[i] < 1 || drop_layers[i] > pyr->levels) {
            bgp_last_message = "dropped layer index out of range";
            return BGP_ERR_LAYER;
        }
    }
    uint32_t w = pyr->gauss[0].width, h = pyr->gauss[0].height;
    if (out_capacity < (uint32_t)((size_t)w * h * pyr->channels)) {
        bgp_last_message = "output capacity too small"; return BGP_ERR_ARG;
    }

    /* Per-channel reconstruction, starting from the coarsest Gaussian. */
    double **recon = (double **)calloc(pyr->channels, sizeof(double *));
    if (!recon) { bgp_last_message = "out of memory"; return BGP_ERR_NOMEM; }
    int rc = BGP_OK;

    for (uint32_t c = 0; c < pyr->channels; c++) {
        const bgp_plane_t *coarse = &pyr->gauss[(size_t)pyr->levels * pyr->channels + c];
        double *cur = bgp_alloc_doubles((size_t)coarse->width * coarse->height);
        if (!cur) { rc = BGP_ERR_NOMEM; bgp_last_message = "out of memory"; goto cleanup; }
        memcpy(cur, coarse->data, (size_t)coarse->width * coarse->height * sizeof(double));
        uint32_t cw = coarse->width, ch = coarse->height;

        for (int64_t lvl = (int64_t)pyr->levels - 1; lvl >= 0; lvl--) {
            const bgp_plane_t *lap = &pyr->lap[(size_t)lvl * pyr->channels + c];
            uint32_t dropped = 0;
            for (uint32_t d = 0; d < drop_count; d++)
                if (drop_layers[d] == (uint32_t)(lvl + 1)) { dropped = 1; break; }

            bgp_plane_t cur_view = { cw, ch, cur };
            double *up = bgp_alloc_doubles((size_t)lap->width * lap->height);
            if (!up) { free(cur); rc = BGP_ERR_NOMEM; bgp_last_message = "out of memory"; goto cleanup; }
            bgp_plane_t up_view = { lap->width, lap->height, up };
            bgp_expand(&cur_view, &up_view);

            double *merged = bgp_alloc_doubles((size_t)lap->width * lap->height);
            if (!merged) { free(up); free(cur); rc = BGP_ERR_NOMEM; bgp_last_message = "out of memory"; goto cleanup; }
            for (size_t i = 0; i < (size_t)lap->width * lap->height; i++)
                merged[i] = up[i] + (dropped ? 0.0 : lap->data[i]);
            free(up);
            free(cur);
            cur = merged;
            cw = lap->width;
            ch = lap->height;
        }
        recon[c] = cur;
    }

    /* Interleave channels and clamp to [0,255]. */
    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (uint32_t c = 0; c < pyr->channels; c++) {
            double v = recon[c][i];
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            out_pixels[i * pyr->channels + c] = (uint8_t)(v + 0.5);
        }
    }
    bgp_last_message = "ok";

cleanup:
    for (uint32_t c = 0; c < pyr->channels; c++) free(recon[c]);
    free(recon);
    return rc;
}

/* Mean absolute horizontal+vertical gradient (finite differences), per plane. */
double bgp_mean_gradient(const uint8_t *image, uint32_t width, uint32_t height,
                         uint32_t channels)
{
    if (!image || width < 2 || height < 2 || (channels != 1 && channels != 3 && channels != 4))
        return 0.0;
    double sum = 0.0;
    size_t count = 0;
    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                if (x + 1 < width) {
                    int d = (int)image[((size_t)y * width + x) * channels + c]
                          - (int)image[((size_t)y * width + x + 1) * channels + c];
                    sum += (double)(d < 0 ? -d : d);
                    count++;
                }
                if (y + 1 < height) {
                    int d = (int)image[((size_t)y * width + x) * channels + c]
                          - (int)image[((size_t)(y + 1) * width + x) * channels + c];
                    sum += (double)(d < 0 ? -d : d);
                    count++;
                }
            }
        }
    }
    return count ? sum / (double)count : 0.0;
}

double bgp_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}
