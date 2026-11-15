/* WebP load/save. Compile only with -DBITGRAIN_USE_WEBP and -lwebp.
 *
 * Both WebP code paths are supported through libwebp:
 *   - lossy VP8  (simple RIFF "VP8 " chunk or VP8X container)
 *   - lossless VP8L ("VP8L" chunk or VP8X container)
 * The loader preserves an embedded alpha plane through the RGBA entry points
 * and reports explicit, thread-local error reasons instead of a bare NULL. */

#ifdef BITGRAIN_USE_WEBP

#include "webp_io.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webp/decode.h>
#include <webp/encode.h>

#if defined(__STDC_NO_THREADS__) || defined(_WIN32)
#define BITGRAIN_THREAD_LOCAL __thread
#else
#define BITGRAIN_THREAD_LOCAL _Thread_local
#endif

#define BITGRAIN_WEBP_ERR_CAP 160
static BITGRAIN_THREAD_LOCAL char g_webp_error[BITGRAIN_WEBP_ERR_CAP] = "no WebP error";

static void webp_set_error(const char *msg)
{
    if (!msg) msg = "unknown WebP error";
    size_t n = strlen(msg);
    if (n >= sizeof(g_webp_error)) n = sizeof(g_webp_error) - 1u;
    memcpy(g_webp_error, msg, n);
    g_webp_error[n] = '\0';
}

static void webp_set_errorf(const char *fmt, long v)
{
    char tmp[BITGRAIN_WEBP_ERR_CAP];
    snprintf(tmp, sizeof(tmp), fmt, v);
    webp_set_error(tmp);
}

const char *bitgrain_webp_last_error(void)
{
    return g_webp_error;
}

int bitgrain_is_webp_mem(const uint8_t *mem, size_t mem_size)
{
    /* RIFF header (12 bytes): "RIFF" + LE size + "WEBP". */
    return mem != NULL && mem_size >= 12u &&
           mem[0] == 'R' && mem[1] == 'I' && mem[2] == 'F' && mem[3] == 'F' &&
           mem[8] == 'W' && mem[9] == 'E' && mem[10] == 'B' && mem[11] == 'P';
}

static const char *webp_status_reason(VP8StatusCode st)
{
    switch (st) {
        case VP8_STATUS_OUT_OF_MEMORY:       return "out of memory while decoding WebP";
        case VP8_STATUS_INVALID_PARAM:       return "invalid WebP bitstream parameter";
        case VP8_STATUS_BITSTREAM_ERROR:     return "corrupt WebP bitstream";
        case VP8_STATUS_UNSUPPORTED_FEATURE: return "unsupported WebP feature";
        case VP8_STATUS_SUSPENDED:           return "truncated WebP bitstream (not enough data)";
        case VP8_STATUS_USER_ABORT:          return "WebP decode aborted";
        case VP8_STATUS_NOT_ENOUGH_DATA:     return "truncated WebP container (not enough data)";
        default:                             return "invalid WebP data";
    }
}

static uint8_t *webp_read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        webp_set_error("cannot open WebP file");
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        webp_set_error("cannot seek WebP file");
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        webp_set_error("WebP file is empty or unreadable");
        fclose(f);
        return NULL;
    }
    if ((size_t)sz > BITGRAIN_WEBP_MAX_BYTES) {
        webp_set_errorf("WebP file exceeds %u byte limit", (long)BITGRAIN_WEBP_MAX_BYTES);
        fclose(f);
        return NULL;
    }
    rewind(f);
    uint8_t *raw = (uint8_t *)malloc((size_t)sz);
    if (!raw) {
        webp_set_error("out of memory reading WebP file");
        fclose(f);
        return NULL;
    }
    if (fread(raw, 1, (size_t)sz, f) != (size_t)sz) {
        webp_set_error("short read while loading WebP file");
        free(raw);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return raw;
}

int bitgrain_webp_probe_mem(const uint8_t *mem, size_t mem_size,
                            uint32_t *out_width, uint32_t *out_height,
                            int *out_has_alpha)
{
    if (!bitgrain_is_webp_mem(mem, mem_size)) {
        webp_set_error("not a WebP container (missing RIFF/WEBP magic)");
        return -1;
    }
    WebPBitstreamFeatures features;
    VP8StatusCode st = WebPGetFeatures(mem, mem_size, &features);
    if (st != VP8_STATUS_OK) {
        webp_set_error(webp_status_reason(st));
        return -1;
    }
    if (features.width <= 0 || features.height <= 0) {
        webp_set_error("WebP reports non-positive dimensions");
        return -1;
    }
    if ((uint32_t)features.width > BITGRAIN_MAX_DIM ||
        (uint32_t)features.height > BITGRAIN_MAX_DIM) {
        webp_set_errorf("WebP dimensions exceed %u px limit", (long)BITGRAIN_MAX_DIM);
        return -1;
    }
    if (out_width)  *out_width  = (uint32_t)features.width;
    if (out_height) *out_height = (uint32_t)features.height;
    if (out_has_alpha) *out_has_alpha = features.has_alpha ? 1 : 0;
    return 0;
}

/* Decode into a plain malloc'd buffer so callers can release it through
 * bitgrain_image_free() like every other loader. channels is 3 (RGB) or 4 (RGBA). */
static uint8_t *webp_decode_mem(const uint8_t *mem, size_t mem_size, int channels,
                                uint32_t *out_width, uint32_t *out_height)
{
    uint32_t w = 0, h = 0;
    int has_alpha = 0;
    if (bitgrain_webp_probe_mem(mem, mem_size, &w, &h, &has_alpha) != 0)
        return NULL;

    size_t n = (size_t)w * (size_t)h * (size_t)channels;
    if (n > BITGRAIN_MAX_PIXEL_BYTES) {
        webp_set_error("decoded WebP image is too large");
        return NULL;
    }

    /* Full decode through the config API so the concrete VP8 status code is
     * available for precise failure reporting (truncation vs. corruption). */
    WebPDecoderConfig config;
    if (!WebPInitDecoderConfig(&config)) {
        webp_set_error("WebP decoder ABI version mismatch");
        return NULL;
    }
    config.output.colorspace = (channels == 4) ? MODE_RGBA : MODE_RGB;
    VP8StatusCode st = WebPDecode(mem, mem_size, &config);
    if (st != VP8_STATUS_OK) {
        webp_set_error(webp_status_reason(st));
        WebPFreeDecBuffer(&config.output);
        return NULL;
    }
    if (config.output.width != (int)w || config.output.height != (int)h ||
        config.output.u.RGBA.rgba == NULL) {
        WebPFreeDecBuffer(&config.output);
        webp_set_error("WebP decode returned inconsistent dimensions");
        return NULL;
    }

    uint8_t *out = (uint8_t *)malloc(n);
    if (!out) {
        WebPFreeDecBuffer(&config.output);
        webp_set_error("out of memory allocating WebP pixels");
        return NULL;
    }
    memcpy(out, config.output.u.RGBA.rgba, n);
    WebPFreeDecBuffer(&config.output);
    *out_width = w;
    *out_height = h;
    return out;
}

uint8_t *bitgrain_load_webp_rgb_mem(const uint8_t *mem, size_t mem_size,
                                    uint32_t *out_width, uint32_t *out_height)
{
    if (!mem || mem_size == 0) {
        webp_set_error("empty WebP input buffer");
        return NULL;
    }
    return webp_decode_mem(mem, mem_size, 3, out_width, out_height);
}

uint8_t *bitgrain_load_webp_rgba_mem(const uint8_t *mem, size_t mem_size,
                                     uint32_t *out_width, uint32_t *out_height)
{
    if (!mem || mem_size == 0) {
        webp_set_error("empty WebP input buffer");
        return NULL;
    }
    return webp_decode_mem(mem, mem_size, 4, out_width, out_height);
}

uint8_t *bitgrain_load_webp_rgb(const char *path, uint32_t *out_width, uint32_t *out_height)
{
    size_t sz = 0;
    uint8_t *raw = webp_read_file(path, &sz);
    if (!raw) return NULL;
    uint8_t *pixels = bitgrain_load_webp_rgb_mem(raw, sz, out_width, out_height);
    free(raw);
    return pixels;
}

uint8_t *bitgrain_load_webp_rgba(const char *path, uint32_t *out_width, uint32_t *out_height)
{
    size_t sz = 0;
    uint8_t *raw = webp_read_file(path, &sz);
    if (!raw) return NULL;
    uint8_t *pixels = bitgrain_load_webp_rgba_mem(raw, sz, out_width, out_height);
    free(raw);
    return pixels;
}

int bitgrain_write_webp(const char *path, const uint8_t *pixels,
                        uint32_t width, uint32_t height, int comp, int quality)
{
    if (!path || !pixels || width == 0 || height == 0) return -1;
    if (quality < 1) quality = 80;
    if (quality > 100) quality = 100;
    float q = (float)quality;

    uint8_t *out = NULL;
    size_t out_size = 0;
    if (comp == 4) {
        out_size = WebPEncodeRGBA(pixels, (int)width, (int)height, (int)width * 4, q, &out);
    } else {
        out_size = WebPEncodeRGB(pixels, (int)width, (int)height, (int)width * 3, q, &out);
    }
    if (!out || out_size == 0) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        WebPFree(out);
        return -1;
    }
    int ok = (fwrite(out, 1, out_size, f) == out_size);
    fclose(f);
    WebPFree(out);
    return ok ? 0 : -1;
}

#else  /* !BITGRAIN_USE_WEBP — libwebp unavailable at build time.
        * Keep the internal symbols resolvable; every call fails explicitly. */

#include "webp_io.h"
#include <stddef.h>

static const char *const g_webp_unavailable = "WebP support not built (libwebp missing)";

int bitgrain_is_webp_mem(const uint8_t *mem, size_t mem_size)
{
    (void)mem; (void)mem_size;
    return 0;
}

const char *bitgrain_webp_last_error(void) { return g_webp_unavailable; }

int bitgrain_webp_probe_mem(const uint8_t *mem, size_t mem_size,
                            uint32_t *out_width, uint32_t *out_height,
                            int *out_has_alpha)
{
    (void)mem; (void)mem_size; (void)out_width; (void)out_height; (void)out_has_alpha;
    return -1;
}

uint8_t *bitgrain_load_webp_rgb_mem(const uint8_t *mem, size_t mem_size,
                                    uint32_t *out_width, uint32_t *out_height)
{ (void)mem; (void)mem_size; (void)out_width; (void)out_height; return NULL; }

uint8_t *bitgrain_load_webp_rgba_mem(const uint8_t *mem, size_t mem_size,
                                     uint32_t *out_width, uint32_t *out_height)
{ (void)mem; (void)mem_size; (void)out_width; (void)out_height; return NULL; }

uint8_t *bitgrain_load_webp_rgb(const char *path, uint32_t *out_width, uint32_t *out_height)
{ (void)path; (void)out_width; (void)out_height; return NULL; }

uint8_t *bitgrain_load_webp_rgba(const char *path, uint32_t *out_width, uint32_t *out_height)
{ (void)path; (void)out_width; (void)out_height; return NULL; }

int bitgrain_write_webp(const char *path, const uint8_t *pixels,
                        uint32_t width, uint32_t height, int comp, int quality)
{ (void)path; (void)pixels; (void)width; (void)height; (void)comp; (void)quality; return -1; }

#endif
