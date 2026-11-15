/* PSNR and SSIM for 8-bit images. No extra deps. */

#include "metrics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

/* ------------------------------------------------------------------ */
/* Guided filter (He, Sun, Tang, TPAMI 2013) for edge-preserving       */
/* denoise, plus a direct bilateral filter as a timing reference.      */
/* Self-guided, 8-bit grayscale.                                       */
/*                                                                     */
/* Per iteration:                                                      */
/*   1) local means of guide I and input p (box filter, radius r)      */
/*   2) local variance of I and covariance of (I, p)                   */
/*   3) linear coefficients a = cov/(var+eps), b = mean_p - a*mean_I   */
/*   4) q = box(a)*I + box(b)   (mean substitution back)               */
/* Iterations feed q back as the next guide/input.                     */
/*                                                                     */
/* Public prototypes (declared here until metrics.h grows them):       */
/*                                                                     */
/* int bitgrain_guided_filter_u8(const uint8_t *src, uint8_t *dst,     */
/*                               uint32_t width, uint32_t height,      */
/*                               int radius, double eps,               */
/*                               int iterations);                      */
/* int bitgrain_bilateral_filter_u8(const uint8_t *src, uint8_t *dst,  */
/*                                  uint32_t width, uint32_t height,   */
/*                                  int radius, double sigma_spatial,  */
/*                                  double sigma_range);               */
/*                                                                     */
/* Both return 0 on success, -1 on invalid arguments (NULL pointers,   */
/* zero dimensions, radius < 1, eps/sigma <= 0 or NaN,                 */
/* iterations < 1), -2 on allocation failure.                          */
/* ------------------------------------------------------------------ */

int bitgrain_guided_filter_u8(const uint8_t *src, uint8_t *dst,
                              uint32_t width, uint32_t height,
                              int radius, double eps, int iterations);
int bitgrain_bilateral_filter_u8(const uint8_t *src, uint8_t *dst,
                                 uint32_t width, uint32_t height,
                                 int radius, double sigma_spatial,
                                 double sigma_range);

/* Box mean via integral image; window clamped at borders. Safe in-place
 * (out may alias in) because the integral is built before writing out. */
static void box_mean_f64(const double *in, double *out, double *integ,
                         uint32_t w, uint32_t h, int r)
{
    const uint32_t iw = w + 1;
    for (uint32_t x = 0; x <= w; x++) integ[x] = 0.0;
    for (uint32_t y = 0; y < h; y++) {
        double rowsum = 0.0;
        integ[(size_t)(y + 1) * iw] = 0.0;
        for (uint32_t x = 0; x < w; x++) {
            rowsum += in[(size_t)y * w + x];
            integ[(size_t)(y + 1) * iw + (x + 1)] =
                integ[(size_t)y * iw + (x + 1)] + rowsum;
        }
    }
    for (uint32_t y = 0; y < h; y++) {
        int32_t y0 = (int32_t)y - r;
        int32_t y1 = (int32_t)y + r;
        if (y0 < 0) y0 = 0;
        if (y1 > (int32_t)h - 1) y1 = (int32_t)h - 1;
        for (uint32_t x = 0; x < w; x++) {
            int32_t x0 = (int32_t)x - r;
            int32_t x1 = (int32_t)x + r;
            if (x0 < 0) x0 = 0;
            if (x1 > (int32_t)w - 1) x1 = (int32_t)w - 1;
            double area = (double)(y1 - y0 + 1) * (double)(x1 - x0 + 1);
            double s = integ[(size_t)(y1 + 1) * iw + (size_t)(x1 + 1)]
                     - integ[(size_t)y0 * iw + (size_t)(x1 + 1)]
                     - integ[(size_t)(y1 + 1) * iw + (size_t)x0]
                     + integ[(size_t)y0 * iw + (size_t)x0];
            out[(size_t)y * w + x] = s / area;
        }
    }
}

/* One guided-filter pass: q = box(a)*I + box(b) with a,b from local
 * statistics of guide I and input p. Scratch must hold 5*w*h doubles
 * plus (w+1)*(h+1) doubles for the integral image. */
static void guided_step_f64(const double *guide, const double *p, double *q,
                            uint32_t w, uint32_t h, int r, double eps,
                            double *scratch)
{
    const size_t n = (size_t)w * h;
    double *mean_i  = scratch;
    double *mean_p  = scratch + n;
    double *corr_i  = scratch + 2 * n; /* reused for a, then box(a) */
    double *corr_ip = scratch + 3 * n; /* reused for b, then box(b) */
    double *prod    = scratch + 4 * n;
    double *integ   = scratch + 5 * n;

    box_mean_f64(guide, mean_i, integ, w, h, r);
    box_mean_f64(p, mean_p, integ, w, h, r);

    for (size_t i = 0; i < n; i++) prod[i] = guide[i] * guide[i];
    box_mean_f64(prod, corr_i, integ, w, h, r);
    for (size_t i = 0; i < n; i++) prod[i] = guide[i] * p[i];
    box_mean_f64(prod, corr_ip, integ, w, h, r);

    for (size_t i = 0; i < n; i++) {
        double var_i = corr_i[i] - mean_i[i] * mean_i[i];
        double cov_ip = corr_ip[i] - mean_i[i] * mean_p[i];
        double a = cov_ip / (var_i + eps);
        corr_i[i] = a;
        corr_ip[i] = mean_p[i] - a * mean_i[i];
    }

    box_mean_f64(corr_i, corr_i, integ, w, h, r);
    box_mean_f64(corr_ip, corr_ip, integ, w, h, r);

    for (size_t i = 0; i < n; i++)
        q[i] = corr_i[i] * guide[i] + corr_ip[i];
}

int bitgrain_guided_filter_u8(const uint8_t *src, uint8_t *dst,
                              uint32_t width, uint32_t height,
                              int radius, double eps, int iterations)
{
    if (!src || !dst || width == 0 || height == 0) return -1;
    if (radius < 1 || !(eps > 0.0) || iterations < 1) return -1;

    const size_t n = (size_t)width * height;
    const size_t integ_n = (size_t)(width + 1) * (height + 1);
    double *base = malloc((2 * n + 5 * n + integ_n) * sizeof(double));
    if (!base) return -2;
    double *cur = base;
    double *q = base + n;
    double *scratch = q + n;

    for (size_t i = 0; i < n; i++) cur[i] = (double)src[i];

    for (int it = 0; it < iterations; it++) {
        guided_step_f64(cur, cur, q, width, height, radius, eps, scratch);
        double *tmp = cur; cur = q; q = tmp;
    }

    for (size_t i = 0; i < n; i++) {
        double v = cur[i] + 0.5;
        if (v < 0.0) v = 0.0;
        if (v > 255.0) v = 255.0;
        dst[i] = (uint8_t)v;
    }
    free(base);
    return 0;
}

int bitgrain_bilateral_filter_u8(const uint8_t *src, uint8_t *dst,
                                 uint32_t width, uint32_t height,
                                 int radius, double sigma_spatial,
                                 double sigma_range)
{
    if (!src || !dst || width == 0 || height == 0) return -1;
    if (radius < 1 || !(sigma_spatial > 0.0) || !(sigma_range > 0.0)) return -1;

    const int ksz = 2 * radius + 1;
    double *spatial = malloc((size_t)ksz * ksz * sizeof(double));
    if (!spatial) return -2;
    double inv_s2 = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            spatial[(dy + radius) * ksz + (dx + radius)] =
                exp(-(double)(dy * dy + dx * dx) * inv_s2);

    double range_lut[256];
    double inv_r2 = 1.0 / (2.0 * sigma_range * sigma_range);
    for (int d = 0; d < 256; d++)
        range_lut[d] = exp(-(double)(d * d) * inv_r2);

    for (uint32_t y = 0; y < height; y++) {
        int32_t y0 = (int32_t)y - radius;
        int32_t y1 = (int32_t)y + radius;
        if (y0 < 0) y0 = 0;
        if (y1 > (int32_t)height - 1) y1 = (int32_t)height - 1;
        for (uint32_t x = 0; x < width; x++) {
            int32_t x0 = (int32_t)x - radius;
            int32_t x1 = (int32_t)x + radius;
            if (x0 < 0) x0 = 0;
            if (x1 > (int32_t)width - 1) x1 = (int32_t)width - 1;
            int center = src[(size_t)y * width + x];
            double wsum = 0.0, acc = 0.0;
            for (int32_t yy = y0; yy <= y1; yy++) {
                for (int32_t xx = x0; xx <= x1; xx++) {
                    int v = src[(size_t)yy * width + (uint32_t)xx];
                    int d = v > center ? v - center : center - v;
                    double w = spatial[(yy - (int32_t)y + radius) * ksz
                                       + (xx - (int32_t)x + radius)]
                             * range_lut[d];
                    wsum += w;
                    acc += w * (double)v;
                }
            }
            double out = acc / wsum + 0.5;
            if (out < 0.0) out = 0.0;
            if (out > 255.0) out = 255.0;
            dst[(size_t)y * width + x] = (uint8_t)out;
        }
    }
    free(spatial);
    return 0;
}
