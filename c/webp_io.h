#ifndef BITGRAIN_WEBP_IO_H
#define BITGRAIN_WEBP_IO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum WebP container accepted by the loader (256 MiB). */
#define BITGRAIN_WEBP_MAX_BYTES (256u * 1024u * 1024u)

/* RIFF/WebP container magic: "RIFF" .... "WEBP". */
int bitgrain_is_webp_mem(const uint8_t *mem, size_t mem_size);

/* Load WebP as RGB (width*height*3). Caller frees with bitgrain_image_free(). Returns NULL on error. */
uint8_t *bitgrain_load_webp_rgb(const char *path, uint32_t *out_width, uint32_t *out_height);

/* Load WebP as RGBA (width*height*4). Caller frees with bitgrain_image_free(). Returns NULL on error. */
uint8_t *bitgrain_load_webp_rgba(const char *path, uint32_t *out_width, uint32_t *out_height);

/* Memory variants of the loaders above. The buffer must contain a complete WebP container.
 * RGB output fills alpha implicitly (255); RGBA preserves any embedded alpha plane.
 * Returns NULL on error; call bitgrain_webp_last_error() for a human-readable reason. */
uint8_t *bitgrain_load_webp_rgb_mem(const uint8_t *mem, size_t mem_size,
                                    uint32_t *out_width, uint32_t *out_height);
uint8_t *bitgrain_load_webp_rgba_mem(const uint8_t *mem, size_t mem_size,
                                     uint32_t *out_width, uint32_t *out_height);

/* Probe a WebP container without fully decoding it.
 * Sets *out_has_alpha to 1 when the bitstream carries an alpha plane (VP8L/VP8X alpha flag).
 * Returns 0 on success, -1 when the buffer is not a decodable WebP stream
 * (see bitgrain_webp_last_error() for the reason). */
int bitgrain_webp_probe_mem(const uint8_t *mem, size_t mem_size,
                            uint32_t *out_width, uint32_t *out_height,
                            int *out_has_alpha);

/* Thread-local, human-readable description of the last WebP loader failure.
 * Never NULL; points to static/storage owned by the library and valid until the
 * next WebP call on the same thread. */
const char *bitgrain_webp_last_error(void);

/* Write RGB (comp=3) or RGBA (comp=4) to WebP. quality 1–100. Returns 0 on success. */
int bitgrain_write_webp(const char *path, const uint8_t *pixels,
                        uint32_t width, uint32_t height, int comp, int quality);

#ifdef __cplusplus
}
#endif

#endif
