/* PSNR, SSIM, edge-gradient retention and a three-stage preprocessing
 * pipeline (median -> bilateral -> unsharp) for 8-bit images.
 * No extra deps. */

#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "metrics.h"
#include "config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double bitgrain_psnr(const uint8_t *orig, const uint8_t *recon,
                     uint32_t width, uint32_t height, uint32_t channels)
{
    uint64_t n = (uint64_t)width * height * channels;
    if (n == 0) return 0.0;

    uint64_t sum_sq = 0;
    for (uint64_t i = 0; i < n; i++) {
        int d = (int)orig[i] - (int)recon[i];
        sum_sq += (uint64_t)(d * d);
    }
    double mse = (double)sum_sq / (double)n;
    if (mse <= 0.0) return 99.0; /* identical */
    return 10.0 * log10((255.0 * 255.0) / mse);
}

/* SSIM constants (standard). */
#define C1 6.5025    /* (0.01*255)^2 */
#define C2 58.5225   /* (0.03*255)^2 */

double bitgrain_ssim(const uint8_t *orig, const uint8_t *recon,
                     uint32_t width, uint32_t height, uint32_t channels)
{
    uint64_t n = (uint64_t)width * height * channels;
    if (n == 0) return 0.0;

    double mu_x = 0.0, mu_y = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        mu_x += orig[i];
        mu_y += recon[i];
    }
    mu_x /= (double)n;
    mu_y /= (double)n;

    double sigma_x2 = 0.0, sigma_y2 = 0.0, sigma_xy = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double dx = (double)orig[i] - mu_x;
        double dy = (double)recon[i] - mu_y;
        sigma_x2 += dx * dx;
        sigma_y2 += dy * dy;
        sigma_xy += dx * dy;
    }
    sigma_x2 /= (double)n;
    sigma_y2 /= (double)n;
    sigma_xy /= (double)n;

    double l = (2.0 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1);
    double sig_x = sqrt(sigma_x2), sig_y = sqrt(sigma_y2);
    double c = (2.0 * sig_x * sig_y + C2) / (sigma_x2 + sigma_y2 + C2);
    double s = (sigma_xy + C2 / 2.0) / (sig_x * sig_y + C2 / 2.0);

    return l * c * s;
}

/* ==================================================================== */
/* Three-stage preprocessing: median -> bilateral -> unsharp            */
/* ==================================================================== */

/*
 * Public types/prototypes for the preprocessing API.
 *
 * metrics.h cannot be edited (only this file may change), so the declarations
 * live here. The demo (demo/main.c) repeats them and carries compile-time
 * static assertions on struct size and member offsets, so any drift is a
 * build error rather than a silent ABI mismatch.
 */
typedef struct {
    int    median_enabled;     /* 0 = skip */
    int    median_radius;      /* [1, 3] */

    int    bilateral_enabled;  /* 0 = skip */
    int    bilateral_radius;   /* [1, 5] */
    double bilateral_sigma_s;  /* (0, 32] */
    double bilateral_sigma_r;  /* (0, 255] */

    int    sharpen_enabled;    /* 0 = skip */
    double sharpen_amount;     /* [0, 3] */
    int    sharpen_radius;     /* [1, 3] */
    double sharpen_threshold;  /* [0, 255] */
} bg_pipeline_params;

typedef struct {
    double median_ms;
    double bilateral_ms;
    double sharpen_ms;
    double total_ms;
} bg_pipeline_timings;

void   bg_pipeline_defaults(bg_pipeline_params *params);
int    bg_pipeline_validate(const bg_pipeline_params *params,
                            char *err, size_t errlen);
int    bg_pipeline_run(const uint8_t *in, uint8_t *out,
                       uint32_t width, uint32_t height, uint32_t channels,
                       const bg_pipeline_params *params,
                       bg_pipeline_timings *timings,
                       char *err, size_t errlen);
double bg_edge_gradient_retention(const uint8_t *reference,
                                  const uint8_t *processed,
                                  uint32_t width, uint32_t height,
                                  uint32_t channels);


static double bg_monotonic_ms(void)
{
#if defined(_WIN32)
    return (double)clock() * (1000.0 / (double)CLOCKS_PER_SEC);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#endif
}

/* The error buffer is optional; every failure path fills it when present. */
static int bg_preprocess_error(char *err, size_t errlen, const char *fmt, ...)
{
    if (err && errlen > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* Median stage: square window of radius r, border pixels replicate. */
static void bg_median_plane(const uint8_t *src, uint8_t *dst,
                            uint32_t width, uint32_t height, int radius)
{
    uint8_t window[7 * 7]; /* radius <= 3 -> at most 49 samples */

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int count = 0;
            for (int dy = -radius; dy <= radius; dy++) {
                int yy = (int)y + dy;
                if (yy < 0) yy = 0;
                if (yy >= (int)height) yy = (int)height - 1;
                const uint8_t *row = src + (size_t)yy * width;
                for (int dx = -radius; dx <= radius; dx++) {
                    int xx = (int)x + dx;
                    if (xx < 0) xx = 0;
                    if (xx >= (int)width) xx = (int)width - 1;
                    window[count++] = row[xx];
                }
            }
            /* Insertion sort; only the middle element is needed. */
            for (int i = 1; i < count; i++) {
                uint8_t v = window[i];
                int j = i - 1;
                while (j >= 0 && window[j] > v) {
                    window[j + 1] = window[j];
                    j--;
                }
                window[j + 1] = v;
            }
            dst[(size_t)y * width + x] = window[count / 2];
        }
    }
}

/* Bilateral stage: Gaussian in space and in intensity. Both weights use
 * lookup tables so the inner loop has no transcendental calls. */
static int bg_bilateral_plane(const uint8_t *src, uint8_t *dst,
                              uint32_t width, uint32_t height,
                              int radius, double sigma_s, double sigma_r)
{
    int side = 2 * radius + 1;
    double *spatial = (double *)malloc((size_t)side * (size_t)side * sizeof(double));
    double *range = (double *)malloc(256 * sizeof(double));
    if (!spatial || !range) {
        free(spatial);
        free(range);
        return -1;
    }

    const double inv_ss = 1.0 / (2.0 * sigma_s * sigma_s);
    const double inv_sr = 1.0 / (2.0 * sigma_r * sigma_r);
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int idx = (dy + radius) * side + (dx + radius);
            spatial[idx] = exp(-(double)(dx * dx + dy * dy) * inv_ss);
        }
    }
    for (int v = 0; v < 256; v++)
        range[v] = exp(-(double)(v * v) * inv_sr);

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int center = src[(size_t)y * width + x];
            double sum_w = 0.0;
            double sum_v = 0.0;
            for (int dy = -radius; dy <= radius; dy++) {
                int yy = (int)y + dy;
                if (yy < 0) yy = 0;
                if (yy >= (int)height) yy = (int)height - 1;
                const uint8_t *row = src + (size_t)yy * width;
                for (int dx = -radius; dx <= radius; dx++) {
                    int xx = (int)x + dx;
                    if (xx < 0) xx = 0;
                    if (xx >= (int)width) xx = (int)width - 1;
                    int v = row[xx];
                    int diff = v - center;
                    if (diff < 0) diff = -diff;
                    double w = spatial[(dy + radius) * side + (dx + radius)] * range[diff];
                    sum_w += w;
                    sum_v += w * (double)v;
                }
            }
            double out = (sum_w > 0.0) ? sum_v / sum_w : (double)center;
            long rounded = lrint(out);
            if (rounded < 0) rounded = 0;
            if (rounded > 255) rounded = 255;
            dst[(size_t)y * width + x] = (uint8_t)rounded;
        }
    }
    free(spatial);
    free(range);
    return 0;
}

/* Unsharp stage: separable Gaussian blur, then add the high-pass residual
 * (only where |residual| >= threshold). */
static int bg_unsharp_plane(const uint8_t *src, uint8_t *dst,
                            uint32_t width, uint32_t height,
                            int radius, double amount, double threshold)
{
    size_t n = (size_t)width * height;
    double *tmp = (double *)malloc(n * sizeof(double));
    double *kernel = (double *)malloc(((size_t)(2 * radius) + 1) * sizeof(double));
    if (!tmp || !kernel) {
        free(tmp);
        free(kernel);
        return -1;
    }

    double sigma = (double)radius;
    if (sigma < 0.5) sigma = 0.5;
    double sum = 0.0;
    for (int k = -radius; k <= radius; k++) {
        double w = exp(-(double)(k * k) / (2.0 * sigma * sigma));
        kernel[k + radius] = w;
        sum += w;
    }
    for (int k = 0; k <= 2 * radius; k++) kernel[k] /= sum;

    /* Horizontal pass into a double buffer (avoids double rounding). */
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *row = src + (size_t)y * width;
        for (uint32_t x = 0; x < width; x++) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; k++) {
                int xx = (int)x + k;
                if (xx < 0) xx = 0;
                if (xx >= (int)width) xx = (int)width - 1;
                acc += kernel[k + radius] * (double)row[xx];
            }
            tmp[(size_t)y * width + x] = acc;
        }
    }

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; k++) {
                int yy = (int)y + k;
                if (yy < 0) yy = 0;
                if (yy >= (int)height) yy = (int)height - 1;
                acc += kernel[k + radius] * tmp[(size_t)yy * width + x];
            }
            double original = (double)src[(size_t)y * width + x];
            double residual = original - acc;
            double value = original;
            if (fabs(residual) >= threshold)
                value = original + amount * residual;
            long rounded = lrint(value);
            if (rounded < 0) rounded = 0;
            if (rounded > 255) rounded = 255;
            dst[(size_t)y * width + x] = (uint8_t)rounded;
        }
    }

    free(tmp);
    free(kernel);
    return 0;
}

void bg_pipeline_defaults(bg_pipeline_params *params)
{
    if (!params) return;
    memset(params, 0, sizeof(*params));
    params->median_enabled = 1;
    params->median_radius = 1;
    params->bilateral_enabled = 1;
    params->bilateral_radius = 2;
    params->bilateral_sigma_s = 2.0;
    params->bilateral_sigma_r = 16.0;
    params->sharpen_enabled = 1;
    params->sharpen_amount = 0.5;
    params->sharpen_radius = 1;
    params->sharpen_threshold = 2.0;
}

/* Every field is range-checked, including stages that are switched off, so a
 * bad value is always reported instead of being silently ignored. */
int bg_pipeline_validate(const bg_pipeline_params *params,
                         char *err, size_t errlen)
{
    if (!params)
        return bg_preprocess_error(err, errlen, "params: NULL");

    if (params->median_enabled != 0 && params->median_enabled != 1)
        return bg_preprocess_error(err, errlen,
            "median: enabled must be 0 or 1 (got %d)", params->median_enabled);
    if (params->median_radius < 1 || params->median_radius > 3)
        return bg_preprocess_error(err, errlen,
            "median: radius=%d out of range [1,3]", params->median_radius);

    if (params->bilateral_enabled != 0 && params->bilateral_enabled != 1)
        return bg_preprocess_error(err, errlen,
            "bilateral: enabled must be 0 or 1 (got %d)", params->bilateral_enabled);
    if (params->bilateral_radius < 1 || params->bilateral_radius > 5)
        return bg_preprocess_error(err, errlen,
            "bilateral: radius=%d out of range [1,5]", params->bilateral_radius);
    if (!(params->bilateral_sigma_s > 0.0) || params->bilateral_sigma_s > 32.0)
        return bg_preprocess_error(err, errlen,
            "bilateral: sigma_s=%g out of range (0,32]", params->bilateral_sigma_s);
    if (!(params->bilateral_sigma_r > 0.0) || params->bilateral_sigma_r > 255.0)
        return bg_preprocess_error(err, errlen,
            "bilateral: sigma_r=%g out of range (0,255]", params->bilateral_sigma_r);

    if (params->sharpen_enabled != 0 && params->sharpen_enabled != 1)
        return bg_preprocess_error(err, errlen,
            "sharpen: enabled must be 0 or 1 (got %d)", params->sharpen_enabled);
    if (params->sharpen_radius < 1 || params->sharpen_radius > 3)
        return bg_preprocess_error(err, errlen,
            "sharpen: radius=%d out of range [1,3]", params->sharpen_radius);
    if (!(params->sharpen_amount >= 0.0) || params->sharpen_amount > 3.0)
        return bg_preprocess_error(err, errlen,
            "sharpen: amount=%g out of range [0,3]", params->sharpen_amount);
    if (!(params->sharpen_threshold >= 0.0) || params->sharpen_threshold > 255.0)
        return bg_preprocess_error(err, errlen,
            "sharpen: threshold=%g out of range [0,255]", params->sharpen_threshold);

    if (err && errlen > 0) err[0] = '\0';
    return 0;
}

int bg_pipeline_run(const uint8_t *in, uint8_t *out,
                    uint32_t width, uint32_t height, uint32_t channels,
                    const bg_pipeline_params *params,
                    bg_pipeline_timings *timings,
                    char *err, size_t errlen)
{
    double t_start = bg_monotonic_ms();
    if (timings) memset(timings, 0, sizeof(*timings));

    if (!in || !out)
        return bg_preprocess_error(err, errlen, "image: NULL pixel buffer");
    if (width == 0 || height == 0)
        return bg_preprocess_error(err, errlen, "image: width and height must be > 0");
    if (channels != 1 && channels != 3 && channels != 4)
        return bg_preprocess_error(err, errlen,
            "image: channels=%u unsupported (expected 1, 3 or 4)", (unsigned)channels);
    uint64_t bytes = (uint64_t)width * (uint64_t)height * (uint64_t)channels;
    if (bytes > (uint64_t)BITGRAIN_MAX_PIXEL_BYTES)
        return bg_preprocess_error(err, errlen, "image: %llu bytes exceeds limit",
                                   (unsigned long long)bytes);
    if (bg_pipeline_validate(params, err, errlen) != 0) return -1;

    size_t n = (size_t)width * (size_t)height;
    uint8_t *plane0 = (uint8_t *)malloc(n);
    uint8_t *plane1 = (uint8_t *)malloc(n);
    if (!plane0 || !plane1) {
        free(plane0);
        free(plane1);
        return bg_preprocess_error(err, errlen, "image: out of memory");
    }

    const uint32_t color_channels = (channels == 4) ? 3 : channels;
    double t_median = 0.0, t_bilateral = 0.0, t_sharpen = 0.0;

    for (uint32_t c = 0; c < color_channels; c++) {
        uint8_t *cur = plane0;
        uint8_t *scratch = plane1;

        if (channels == 1) {
            memcpy(cur, in, n);
        } else {
            for (size_t i = 0; i < n; i++) cur[i] = in[i * channels + c];
        }

        if (params->median_enabled) {
            double t0 = bg_monotonic_ms();
            bg_median_plane(cur, scratch, width, height, params->median_radius);
            t_median += bg_monotonic_ms() - t0;
            uint8_t *swap = cur; cur = scratch; scratch = swap;
        }
        if (params->bilateral_enabled) {
            double t0 = bg_monotonic_ms();
            if (bg_bilateral_plane(cur, scratch, width, height,
                                   params->bilateral_radius,
                                   params->bilateral_sigma_s,
                                   params->bilateral_sigma_r) != 0) {
                free(plane0);
                free(plane1);
                return bg_preprocess_error(err, errlen, "bilateral: out of memory");
            }
            t_bilateral += bg_monotonic_ms() - t0;
            uint8_t *swap = cur; cur = scratch; scratch = swap;
        }
        if (params->sharpen_enabled) {
            double t0 = bg_monotonic_ms();
            if (bg_unsharp_plane(cur, scratch, width, height,
                                 params->sharpen_radius,
                                 params->sharpen_amount,
                                 params->sharpen_threshold) != 0) {
                free(plane0);
                free(plane1);
                return bg_preprocess_error(err, errlen, "sharpen: out of memory");
            }
            t_sharpen += bg_monotonic_ms() - t0;
            uint8_t *swap = cur; cur = scratch; scratch = swap;
        }

        if (channels == 1) {
            memcpy(out, cur, n);
        } else {
            for (size_t i = 0; i < n; i++) out[i * channels + c] = cur[i];
        }
    }

    /* Alpha is passed through untouched. */
    if (channels == 4 && in != out) {
        for (size_t i = 0; i < n; i++) out[i * 4 + 3] = in[i * 4 + 3];
    }

    free(plane0);
    free(plane1);

    if (timings) {
        timings->median_ms = t_median;
        timings->bilateral_ms = t_bilateral;
        timings->sharpen_ms = t_sharpen;
        timings->total_ms = bg_monotonic_ms() - t_start;
    }
    if (err && errlen > 0) err[0] = '\0';
    return 0;
}

/* Sobel gradient magnitude at (x, y) of one channel. */
#define BG_EDGE_MASK_MIN 32.0  /* |grad| below this in the reference is flat */

static double bg_sobel_mag(const uint8_t *pixels,
                           uint32_t width, uint32_t height, uint32_t channels,
                           uint32_t c, uint32_t x, uint32_t y)
{
    static const int KX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int KY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    int gx = 0, gy = 0, k = 0;
    for (int dy = -1; dy <= 1; dy++) {
        int yy = (int)y + dy;
        if (yy < 0) yy = 0;
        if (yy >= (int)height) yy = (int)height - 1;
        for (int dx = -1; dx <= 1; dx++, k++) {
            int xx = (int)x + dx;
            if (xx < 0) xx = 0;
            if (xx >= (int)width) xx = (int)width - 1;
            int v = pixels[((size_t)yy * width + (size_t)xx) * channels + c];
            gx += KX[k] * v;
            gy += KY[k] * v;
        }
    }
    return sqrt((double)gx * gx + (double)gy * gy);
}

/*
 * Retention is measured only where the *reference* has a real edge. Flat
 * regions are excluded so that residual noise cannot pad the denominator:
 * the question is "did the pipeline keep the edges the source had?", not
 * "how much energy is there anywhere?".
 */
double bg_edge_gradient_retention(const uint8_t *reference,
                                  const uint8_t *processed,
                                  uint32_t width, uint32_t height,
                                  uint32_t channels)
{
    if (!reference || !processed) return 0.0;
    if (width < 2 || height < 2) return 0.0;
    if (channels != 1 && channels != 3 && channels != 4) return 0.0;

    const uint32_t color_channels = (channels == 4) ? 3 : channels;
    double ref_energy = 0.0, out_energy = 0.0;

    for (uint32_t c = 0; c < color_channels; c++) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                double ref_mag = bg_sobel_mag(reference, width, height,
                                              channels, c, x, y);
                if (ref_mag < BG_EDGE_MASK_MIN) continue;
                ref_energy += ref_mag;
                out_energy += bg_sobel_mag(processed, width, height,
                                           channels, c, x, y);
            }
        }
    }
    if (ref_energy <= 0.0) return 0.0;
    return out_energy / ref_energy;
}
