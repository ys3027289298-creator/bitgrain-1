/*
 * phase_correlation_me.c — standalone test/benchmark: phase-correlation
 * motion estimation vs. classic full-search block matching.
 *
 * Self-contained (C11 + libm only, no third-party deps). Does not touch any
 * library file; build & run directly:
 *
 *   gcc -std=c11 -O2 -Wall -Wextra tests/phase_correlation_me.c -lm -o tests/out/phase_correlation_me
 *   ./tests/out/phase_correlation_me [search_window] [subpix_step]
 *
 * Generates 4 small synthetic 128x128 sequences (translation / rotation /
 * scaling / no motion), estimates the inter-frame shift with both methods,
 * compensates the reference frame, and reports estimation error, PSNR after
 * compensation, and runtime.
 *
 * Degenerate-spectrum policy (explicit):
 *   - A cross-power bin whose magnitude is ~0 (spectrum zero crossing) is
 *     set to exactly 0 instead of amplifying noise via normalization.
 *   - If ALL bins are degenerate (uniform frames), estimation is aborted and
 *     (0,0) is returned with reliable=0.
 *   - If the correlation peak is not prominent (prominence < PEAK_PROM_MIN),
 *     the estimate is rejected and (0,0) is returned with reliable=0.
 *   - Peaks outside the configured search window are clamped and flagged.
 */

#define _POSIX_C_SOURCE 199309L /* clock_gettime */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 128 /* frame width/height, power of two */
#define PEAK_PROM_MIN 3.0 /* min (peak-mean)/std for a trustworthy peak */
#define EPS 1e-12         /* zero-spectrum threshold for cross-power bins */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double re, im;
} cpx;

typedef struct {
    double dx, dy;     /* estimated shift: cur(x,y) ~= ref(x-dx, y-dy) */
    int reliable;      /* 0 => estimate rejected, (0,0) fallback used */
    double peak;       /* correlation peak height (PC only) */
    double prominence; /* (peak-mean)/std of surface (PC only) */
    int clamped;       /* estimate exceeded the search window */
} ME;

/* ---------------- deterministic PRNG ---------------- */
static uint32_t rng_state = 0x9E3779B9u;
static uint32_t xorshift(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static double frand(void) { return (xorshift() >> 8) / 16777216.0; }

/* ---------------- synthetic scene ---------------- */
/* Textured base frame: smooth gradient + gaussian blobs + fine grain. */
static void make_base(double *img)
{
    static const double blob[8][4] = { /* cx, cy, sigma, amp */
        { 32, 36, 10, 0.9 }, { 90, 30, 14, -0.8 }, { 60, 70, 8, 1.0 },
        { 100, 95, 12, 0.7 }, { 20, 100, 9, -0.9 }, { 75, 110, 7, 0.8 },
        { 45, 15, 6, -0.6 }, { 110, 60, 9, 0.6 },
    };
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            double v = 0.30 + 0.20 * (double)(x + y) / (2.0 * N);
            for (int b = 0; b < 8; b++) {
                double dx = x - blob[b][0], dy = y - blob[b][1];
                double s2 = blob[b][2] * blob[b][2];
                v += blob[b][3] * 0.35 * exp(-(dx * dx + dy * dy) / (2 * s2));
            }
            v += 0.03 * (frand() - 0.5);
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            img[y * N + x] = v * 255.0;
        }
    }
}

/* bilinear sample, clamp-to-edge */
static double sample(const double *img, double x, double y)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > N - 1) x = N - 1;
    if (y > N - 1) y = N - 1;
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1 < N ? x0 + 1 : x0;
    int y1 = y0 + 1 < N ? y0 + 1 : y0;
    double fx = x - x0, fy = y - y0;
    double a = img[y0 * N + x0] * (1 - fx) + img[y0 * N + x1] * fx;
    double b = img[y1 * N + x0] * (1 - fx) + img[y1 * N + x1] * fx;
    return a * (1 - fy) + b * fy;
}

/* cur(x,y) = ref(x-dx, y-dy) */
static void warp_translate(const double *ref, double *cur, double dx, double dy)
{
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            cur[y * N + x] = sample(ref, x - dx, y - dy);
}

static void warp_rotate(const double *ref, double *cur, double deg)
{
    double t = deg * M_PI / 180.0, c = cos(t), s = sin(t);
    double cx = (N - 1) / 2.0, cy = (N - 1) / 2.0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            double px = x - cx, py = y - cy;
            cur[y * N + x] = sample(ref, cx + c * px + s * py, cy - s * px + c * py);
        }
}

static void warp_scale(const double *ref, double *cur, double f)
{
    double cx = (N - 1) / 2.0, cy = (N - 1) / 2.0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            cur[y * N + x] = sample(ref, cx + (x - cx) / f, cy + (y - cy) / f);
}

/* ---------------- FFT (iterative radix-2, row-column 2D) ---------------- */
static void fft1d(cpx *a, int n, int inverse)
{
    for (int i = 1, j = 0; i < n; i++) { /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            cpx t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int j = 0; j < len / 2; j++) {
                cpx u = a[i + j];
                cpx v = { a[i + j + len / 2].re * cr - a[i + j + len / 2].im * ci,
                          a[i + j + len / 2].re * ci + a[i + j + len / 2].im * cr };
                a[i + j].re = u.re + v.re;
                a[i + j].im = u.im + v.im;
                a[i + j + len / 2].re = u.re - v.re;
                a[i + j + len / 2].im = u.im - v.im;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; i++) {
            a[i].re /= n;
            a[i].im /= n;
        }
    }
}

static void fft2d(cpx *a, int inverse)
{
    for (int y = 0; y < N; y++)
        fft1d(a + y * N, N, inverse);
    cpx *col = malloc(sizeof(cpx) * N);
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++)
            col[y] = a[y * N + x];
        fft1d(col, N, inverse);
        for (int y = 0; y < N; y++)
            a[y * N + x] = col[y];
    }
    free(col);
}

/* ---------------- motion estimators ---------------- */

/* Classic full-search integer-pel block matching (SAD), whole frame as one
 * block. This is the baseline path; it is NOT modified by the PC work. */
static ME block_match(const double *ref, const double *cur, int window)
{
    ME r = { 0, 0, 1, 0, 0, 0 };
    double best = 1e30;
    for (int dy = -window; dy <= window; dy++) {
        for (int dx = -window; dx <= window; dx++) {
            double sad = 0;
            for (int y = 0; y < N; y++) {
                int sy = y - dy;
                if (sy < 0) sy = 0;
                if (sy >= N) sy = N - 1;
                for (int x = 0; x < N; x++) {
                    int sx = x - dx;
                    if (sx < 0) sx = 0;
                    if (sx >= N) sx = N - 1;
                    sad += fabs(cur[y * N + x] - ref[sy * N + sx]);
                }
            }
            if (sad < best) {
                best = sad;
                r.dx = dx;
                r.dy = dy;
            }
        }
    }
    return r;
}

/* Phase correlation: FFT -> normalized cross-power spectrum -> IFFT -> peak.
 * subpix_step > 0 refines the peak on a local exact-IDFT grid of that step;
 * subpix_step == 0 disables sub-pixel refinement. */
static ME phase_correlate(const double *ref, const double *cur, int window,
                          double subpix_step)
{
    ME r = { 0, 0, 1, 0, 0, 0 };
    static cpx F[N * N], G[N * N], R[N * N], S[N * N];
    static double hann[N];

    for (int i = 0; i < N; i++)
        hann[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (N - 1)));

    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            double w = hann[x] * hann[y];
            F[y * N + x].re = ref[y * N + x] * w;
            F[y * N + x].im = 0;
            G[y * N + x].re = cur[y * N + x] * w;
            G[y * N + x].im = 0;
        }
    fft2d(F, 0);
    fft2d(G, 0);

    /* normalized cross-power spectrum, with explicit zero-spectrum guard */
    int zero_bins = 0;
    for (int i = 0; i < N * N; i++) {
        double re = F[i].re * G[i].re + F[i].im * G[i].im; /* conj(F)*G */
        double im = F[i].re * G[i].im - F[i].im * G[i].re;
        double mag = hypot(re, im);
        if (mag < EPS) { /* spectrum zero crossing: do not amplify noise */
            R[i].re = 0;
            R[i].im = 0;
            zero_bins++;
        } else {
            R[i].re = re / mag;
            R[i].im = im / mag;
        }
    }
    if (zero_bins == N * N) { /* degenerate: uniform frames, no phase info */
        r.reliable = 0;
        return r; /* (0,0) fallback */
    }

    memcpy(S, R, sizeof(R)); /* keep the spectrum; R is overwritten by IFFT */
    fft2d(R, 1); /* correlation surface in R[].re */

    double mean = 0, peak = -1e30;
    int px = 0, py = 0;
    for (int i = 0; i < N * N; i++)
        mean += R[i].re;
    mean /= N * N;
    double var = 0;
    for (int i = 0; i < N * N; i++) {
        var += (R[i].re - mean) * (R[i].re - mean);
        if (R[i].re > peak) {
            peak = R[i].re;
            px = i % N;
            py = i / N;
        }
    }
    double std = sqrt(var / (N * N));
    r.peak = peak;
    r.prominence = std > EPS ? (peak - mean) / std : 0.0;
    if (r.prominence < PEAK_PROM_MIN) { /* peak not prominent: reject */
        r.reliable = 0;
        return r; /* (0,0) fallback */
    }

    /* wrap to signed integer shift */
    int ix = px > N / 2 ? px - N : px;
    int iy = py > N / 2 ? py - N : py;

    /* sub-pixel: evaluate the exact correlation surface (inverse DFT of the
     * cross-power spectrum) on a local grid of step `subpix_step` around the
     * integer peak, and keep the argmax. subpix_step == 0 disables this.
     * Frequencies must be wrapped to [-N/2, N/2): for non-integer shifts
     * exp(j*2*pi*u*s/N) is not N-periodic in u, so unwrapped bins alias. */
    double fdx = 0, fdy = 0;
    if (subpix_step > 0) {
        static cpx wx[N], wy[N];
        double best = -1e30;
        for (double oy = -1.0; oy <= 1.0 + 1e-9; oy += subpix_step) {
            for (int v = 0; v < N; v++) {
                int ve = v > N / 2 ? v - N : v;
                double ph = 2.0 * M_PI * ve * (iy + oy) / N;
                wy[v].re = cos(ph);
                wy[v].im = sin(ph);
            }
            for (double ox = -1.0; ox <= 1.0 + 1e-9; ox += subpix_step) {
                for (int u = 0; u < N; u++) {
                    int ue = u > N / 2 ? u - N : u;
                    double ph = 2.0 * M_PI * ue * (ix + ox) / N;
                    wx[u].re = cos(ph);
                    wx[u].im = sin(ph);
                }
                double acc = 0;
                for (int v = 0; v < N; v++) {
                    double cyr = wy[v].re, cyi = wy[v].im;
                    const cpx *row = &S[v * N];
                    for (int u = 0; u < N; u++) {
                        double zr = wx[u].re * cyr - wx[u].im * cyi;
                        double zi = wx[u].re * cyi + wx[u].im * cyr;
                        acc += row[u].re * zr - row[u].im * zi;
                    }
                }
                if (acc > best) {
                    best = acc;
                    fdx = ox;
                    fdy = oy;
                }
            }
        }
    }

    r.dx = ix + fdx;
    r.dy = iy + fdy;

    /* search-window constraint */
    if (fabs(r.dx) > window || fabs(r.dy) > window) {
        r.clamped = 1;
        if (r.dx > window) r.dx = window;
        if (r.dx < -window) r.dx = -window;
        if (r.dy > window) r.dy = window;
        if (r.dy < -window) r.dy = -window;
    }
    return r;
}

/* ---------------- compensation & metrics ---------------- */
static void compensate(const double *ref, double *out, double dx, double dy)
{
    warp_translate(ref, out, dx, dy);
}

static double psnr(const double *a, const double *b)
{
    double mse = 0;
    for (int i = 0; i < N * N; i++) {
        double d = a[i] - b[i];
        mse += d * d;
    }
    mse /= N * N;
    if (mse < 1e-12) return 99.0; /* effectively identical */
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ---------------- driver ---------------- */
typedef void (*warp_fn)(const double *, double *, double);

int main(int argc, char **argv)
{
    int window = argc > 1 ? atoi(argv[1]) : 16; /* search window ±px */
    double subpix = argc > 2 ? atof(argv[2]) : 0.25; /* sub-pixel step, 0=off */
    const int reps = 50;

    static double base[N * N], cur[N * N], comp[N * N];
    make_base(base);

    struct {
        const char *name;
        warp_fn warp;
        double param, param2;
        double gt_dx, gt_dy; /* ground-truth global translation */
        int is_translate;
    } seq[4] = {
        { "translate(+3.50,-2.25)", NULL, 3.5, -2.25, 3.5, -2.25, 1 },
        { "rotate(1.5 deg)", NULL, 1.5, 0, 0, 0, 0 },
        { "scale(1.03)", NULL, 1.03, 0, 0, 0, 0 },
        { "no motion", NULL, 0, 0, 0, 0, 0 },
    };

    printf("config: frame=%dx%d  search_window=+/-%d px  subpix_step=%.3f%s  reps=%d\n",
           N, N, window, subpix, subpix > 0 ? "" : " (sub-pixel off)", reps);
    printf("ground truth: translation sequence shifts by (dx,dy)=(+3.50,-2.25); "
           "rotation/scale/none have zero global translation\n\n");

    double t_bm_total = 0, t_pc_total = 0;

    printf("%-24s | %-7s | %-16s %-7s %-9s | %-16s %-7s %-9s %s\n",
           "sequence", "PSNR raw", "BM est (dx,dy)", "BM err", "BM PSNR",
           "PC est (dx,dy)", "PC err", "PC PSNR", "PC flags");
    printf("---------------------------+----------+------------------"
           "---------+-----------+------------------|---------+-----------+"
           "---------------------\n");

    for (int s = 0; s < 4; s++) {
        switch (s) {
        case 0: warp_translate(base, cur, seq[s].param, seq[s].param2); break;
        case 1: warp_rotate(base, cur, seq[s].param); break;
        case 2: warp_scale(base, cur, seq[s].param); break;
        default: memcpy(cur, base, sizeof(base)); break;
        }

        double p_raw = psnr(base, cur);

        double t0 = now_ms();
        ME bm;
        for (int r = 0; r < reps; r++)
            bm = block_match(base, cur, window);
        double t_bm = (now_ms() - t0) / reps;

        t0 = now_ms();
        ME pc;
        for (int r = 0; r < reps; r++)
            pc = phase_correlate(base, cur, window, subpix);
        double t_pc = (now_ms() - t0) / reps;

        t_bm_total += t_bm;
        t_pc_total += t_pc;

        compensate(base, comp, bm.dx, bm.dy);
        double p_bm = psnr(comp, cur);
        compensate(base, comp, pc.dx, pc.dy);
        double p_pc = psnr(comp, cur);

        double e_bm = hypot(bm.dx - seq[s].gt_dx, bm.dy - seq[s].gt_dy);
        double e_pc = hypot(pc.dx - seq[s].gt_dx, pc.dy - seq[s].gt_dy);

        char flags[64] = "";
        if (!pc.reliable) strcat(flags, "REJECTED(weak peak)->(0,0) ");
        if (pc.clamped) strcat(flags, "CLAMPED-to-window ");
        if (!flags[0]) strcpy(flags, "-");

        printf("%-24s | %7.2f | (%6.2f,%6.2f)  %6.3f  %8.2f | (%6.2f,%6.2f)  "
               "%6.3f  %8.2f  %s\n",
               seq[s].name, p_raw, bm.dx, bm.dy, e_bm, p_bm, pc.dx, pc.dy,
               e_pc, p_pc, flags);
        if (s == 0 || !pc.reliable)
            printf("  %-22s | PC peak=%.4f prominence=%.1f (min %.1f)\n", "",
                   pc.peak, pc.prominence, PEAK_PROM_MIN);
    }

    printf("\navg time per frame pair (%d reps): block matching %.2f ms, "
           "phase correlation %.2f ms  (speedup %.1fx)\n",
           reps, t_bm_total / 4, t_pc_total / 4,
           t_pc_total > 0 ? t_bm_total / t_pc_total : 0.0);
    return 0;
}
