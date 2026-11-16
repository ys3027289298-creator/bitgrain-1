#ifndef BITGRAIN_TIFF_IO_H
#define BITGRAIN_TIFF_IO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns non-zero if mem starts with a classic TIFF signature
   (little-endian "II*\0" or big-endian "MM\0*"). */
int bitgrain_tiff_probe_mem(const uint8_t *mem, size_t size);

/* Returns non-zero if the file at path starts with a TIFF signature. */
int bitgrain_tiff_probe_path(const char *path);

/* Decode a TIFF from memory.
 * Supports: uncompressed (Compression=1) and LZW (Compression=5),
 * 8/16-bit samples (16-bit is reduced to 8-bit via >>8),
 * grayscale (Photometric 0/1) and RGB(A) (Photometric 2),
 * strip-based images, Predictor 1 and 2.
 * req_channels: 1 (gray), 3 (RGB) or 4 (RGBA).
 * Returns malloc'd width*height*req_channels bytes (free with
 * bitgrain_image_free()), or NULL on error (a specific message is
 * printed to stderr, e.g. which strip is corrupt and why). */
uint8_t *bitgrain_tiff_load_mem(const uint8_t *mem, size_t size,
                                uint32_t *out_width, uint32_t *out_height,
                                int req_channels);

/* Same, reading the file from path. */
uint8_t *bitgrain_tiff_load(const char *path,
                            uint32_t *out_width, uint32_t *out_height,
                            int req_channels);

#ifdef __cplusplus
}
#endif

#endif
