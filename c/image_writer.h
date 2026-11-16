#ifndef BITGRAIN_IMAGE_WRITER_H
#define BITGRAIN_IMAGE_WRITER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Write an image as JPEG.
 * comp: 1 = grayscale, 3 = RGB. quality: 1-100.
 * Returns 0 on success, non-zero on error.
 */
int bitgrain_write_jpg(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp,
                       int quality);

/**
 * Write an image as PNG (lossless).
 * comp: 1 = grayscale, 3 = RGB, 4 = RGBA.
 * Returns 0 on success, non-zero on error.
 */
int bitgrain_write_png(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp);

/**
 * Write an image as BMP (lossless).
 * comp: 1 = grayscale, 3 = RGB, 4 = RGBA.
 * Returns 0 on success, non-zero on error.
 */
int bitgrain_write_bmp(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp);

/**
 * Write an image as TGA.
 * comp: 1 = grayscale, 3 = RGB, 4 = RGBA.
 * Returns 0 on success, non-zero on error.
 */
int bitgrain_write_tga(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height,
                       int comp);

/**
 * Write an image as WebP. comp: 3 = RGB, 4 = RGBA. quality 1–100.
 * Returns 0 on success, non-zero on error.
 */
int bitgrain_write_webp(const char *path,
                        const uint8_t *pixels,
                        uint32_t width,
                        uint32_t height,
                        int comp,
                        int quality);

/**
 * Write grayscale as PGM (P5). comp must be 1.
 * Returns 0 on success.
 */
int bitgrain_write_pgm(const char *path,
                       const uint8_t *pixels,
                       uint32_t width,
                       uint32_t height);

/**
 * Write PGM (P5) from 1/3/4-channel pixels. RGB/RGBA are collapsed to
 * BT.601 luma because P5 stores a single channel.
 * Returns 0 on success.
 */
int bitgrain_write_pgm_planes(const char *path,
                              const uint8_t *pixels,
                              uint32_t width,
                              uint32_t height,
                              uint32_t channels);

/* ================================================================== */
/* Gaussian / Laplacian pyramid for pre-encode band trade-off.         */
/* Opaque handle; build with bgp_pyramid_build(), release with         */
/* bgp_pyramid_free(). Layer numbers are 1-based (L1 = finest detail). */
/* ================================================================== */
typedef struct bgp_pyramid bgp_pyramid_t;

/* Build a pyramid with `levels` Laplacian layers. Returns NULL on error
 * (bad arguments, levels too deep for the image, OOM); bgp_last_error(). */
bgp_pyramid_t *bgp_pyramid_build(const uint8_t *image, uint32_t width,
                                 uint32_t height, uint32_t channels,
                                 uint32_t levels);
void bgp_pyramid_free(bgp_pyramid_t *pyr);

/* Number of octaves whose smallest Gaussian layer stays >= 4 px. */
uint32_t bgp_max_levels(uint32_t width, uint32_t height);

uint32_t bgp_pyramid_levels(const bgp_pyramid_t *pyr);
uint32_t bgp_pyramid_channels(const bgp_pyramid_t *pyr);

/* Dimensions of Laplacian layer `layer` (1-based). Zeroes on bad index. */
void bgp_layer_dimensions(const bgp_pyramid_t *pyr, uint32_t layer,
                          uint32_t *out_width, uint32_t *out_height);

/* Dimensions of Gaussian layer g (0-based, 0 = source resolution). */
void bgp_gaussian_dimensions(const bgp_pyramid_t *pyr, uint32_t g,
                             uint32_t *out_width, uint32_t *out_height);

/* Fraction (0..1) of total Laplacian energy held by `layer`. */
int bgp_energy_share(const bgp_pyramid_t *pyr, uint32_t layer, double *out_share);

/* Rebuild, zeroing each Laplacian layer listed in drop_layers. out_pixels
 * must hold width*height*channels bytes. Returns BGP_OK (0) or negative. */
int bgp_pyramid_rebuild(const bgp_pyramid_t *pyr, const uint32_t *drop_layers,
                        uint32_t drop_count, uint8_t *out_pixels,
                        uint32_t out_capacity);

/* Mean absolute first-order gradient (horizontal+vertical), all channels. */
double bgp_mean_gradient(const uint8_t *image, uint32_t width, uint32_t height,
                         uint32_t channels);

/* Monotonic wall-clock timestamp in milliseconds (timing helper). */
double bgp_now_ms(void);

/* Human-readable message for the last pyramid call in this thread. */
const char *bgp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
