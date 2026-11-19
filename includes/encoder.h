#ifndef BITGRAIN_ENCODER_H
#define BITGRAIN_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes exposed by bitgrain_last_error_code(). */
enum {
    BITGRAIN_OK = 0,
    BITGRAIN_ERR_INVALID_ARG = 1,
    BITGRAIN_ERR_DECODE_FAILED = 2,
    BITGRAIN_ERR_THREAD_INIT = 3,
    BITGRAIN_ERR_PANIC = 100
};

/*
 * Encode a grayscale image (8 bpp) to .bg stream.
 * quality: 1–100 (higher = less quantization), 0 = default 85.
 * Returns 0 on success, -1 on error.
 */
int bitgrain_encode_grayscale(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality);

/*
 * Encode an RGB image (24 bpp, R G B per pixel) to .bg stream.
 * quality: 1–100, 0 = default 85.
 */
int bitgrain_encode_rgb(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality);

/*
 * Encode an RGBA image (32 bpp, R G B A per pixel) to .bg stream.
 * quality: 1–100, 0 = default 85.
 */
int bitgrain_encode_rgba(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality);

/*
 * Configure global worker thread count for the codec internals (Rayon).
 * Call before encode/decode to guarantee effect.
 * Returns 0 on success, -1 on invalid value or late/failed init.
 */
int bitgrain_set_threads(int32_t threads);

/*
 * Thread-local error status for API calls.
 * Every public API call updates this state. On success, code is BITGRAIN_OK.
 * bitgrain_last_error_message() returns a pointer owned by the library:
 * - valid only in the calling thread
 * - may change after the next bitgrain_* call on that same thread
 * Copy it immediately if you need to persist it.
 */
int bitgrain_last_error_code(void);
const char *bitgrain_last_error_message(void);
void bitgrain_clear_error(void);

/*
 * Decode a .bg stream into pixels (grayscale, RGB, or RGBA per header).
 * out_channels: output, 1 = grayscale (w*h), 3 = RGB (w*h*3), 4 = RGBA (w*h*4).
 * out_capacity must be >= width*height*out_channels.
 * On failure, output fields should be treated as undefined.
 */
int bitgrain_decode(
    const uint8_t *buffer,
    int32_t size,
    uint8_t *out_pixels,
    uint32_t out_capacity,
    uint32_t *out_width,
    uint32_t *out_height,
    uint32_t *out_channels);

/*
 * Decode a .bg stream to grayscale (version 1 only).
 */
int bitgrain_decode_grayscale(
    const uint8_t *buffer,
    int32_t size,
    uint8_t *out_pixels,
    uint32_t out_capacity,
    uint32_t *out_width,
    uint32_t *out_height);

/*
 * Encode RGB with optional ICC profile. icc may be NULL (no ICC).
 */
int bitgrain_encode_rgb_icc(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality,
    const uint8_t *icc,
    uint32_t icc_len);

/*
 * Encode RGBA with optional ICC profile. icc may be NULL (no ICC).
 */
int bitgrain_encode_rgba_icc(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality,
    const uint8_t *icc,
    uint32_t icc_len);

/*
 * Decode .bg and return embedded ICC. *out_icc receives malloc'd ICC (caller frees with bitgrain_free_icc).
 */
int bitgrain_decode_icc(
    const uint8_t *buffer,
    int32_t size,
    uint8_t *out_pixels,
    uint32_t out_capacity,
    uint32_t *out_width,
    uint32_t *out_height,
    uint32_t *out_channels,
    uint8_t **out_icc,
    uint32_t *out_icc_len);

void bitgrain_free_icc(uint8_t *icc, uint32_t len);

/*
 * Sample Adaptive Offset (SAO) configuration for the v20/v21 in-loop filter.
 * Layout mirrors the Rust SaoConfig and is ABI-stable (all fixed-width fields).
 */
typedef struct {
    /* Master switch (always 1 for v20/v21 streams). */
    uint8_t  enabled;
    /* Square region side in pixels; 0 means the default (64). */
    uint16_t region_size;
    /* Edge classes per EO region: 2, 4 or 5. */
    uint8_t  edge_classes;
    /* Four EO directions: horizontal, vertical, 135 deg, 45 deg (nonzero=on). */
    uint8_t  edge_directions[4];
    /* Nonzero enables the band-offset (contouring) candidate. */
    uint8_t  enable_band;
} bitgrain_sao_config_t;

/*
 * Fill an SAO config. Pass NULL for edge_directions to keep all four on.
 * The returned struct is a plain value type; no allocation or free needed.
 */
bitgrain_sao_config_t bitgrain_sao_config_new(
    uint16_t region_size,
    uint8_t edge_classes,
    const uint8_t *edge_directions,
    uint8_t enable_band);

/*
 * Encode RGB/RGBA with an explicit SAO configuration (v20/v21).
 * Pass NULL for sao_cfg to use the default (region 64, 5 classes, all
 * directions, band offset enabled).
 */
int bitgrain_encode_rgb_sao(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality,
    const bitgrain_sao_config_t *sao_cfg);

int bitgrain_encode_rgba_sao(
    const uint8_t *image,
    uint32_t width,
    uint32_t height,
    uint8_t *out_buffer,
    uint32_t out_capacity,
    int32_t *out_len,
    uint8_t quality,
    const bitgrain_sao_config_t *sao_cfg);

#ifdef __cplusplus
}
#endif

#endif
