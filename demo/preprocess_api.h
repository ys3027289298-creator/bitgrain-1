/* Demo entry point for the three-stage preprocessing pipeline.
 *
 * This file is NOT part of the library. It declares the preprocessing API that
 * is implemented (additively) in the single modified library file c/metrics.c,
 * and prints the four requested measurements plus a single-stage comparison.
 *
 * Build:  make -f demo/Makefile      (or use demo/run.sh)
 */
#ifndef BITGRAIN_PREPROCESS_API
#define BITGRAIN_PREPROCESS_API

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

void bg_pipeline_defaults(bg_pipeline_params *params);
int  bg_pipeline_validate(const bg_pipeline_params *params, char *err, size_t errlen);
int  bg_pipeline_run(const uint8_t *in, uint8_t *out,
                     uint32_t width, uint32_t height, uint32_t channels,
                     const bg_pipeline_params *params,
                     bg_pipeline_timings *timings,
                     char *err, size_t errlen);
double bg_edge_gradient_retention(const uint8_t *reference,
                                  const uint8_t *processed,
                                  uint32_t width, uint32_t height,
                                  uint32_t channels);

#ifdef __cplusplus
}
#endif

#endif
