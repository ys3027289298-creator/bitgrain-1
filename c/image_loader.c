#include "image_loader.h"
#include "webp_io.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* Thread-local, human-readable loader failure reason (internal CLI use). */
#if defined(__STDC_NO_THREADS__) || defined(_WIN32)
#define BITGRAIN_THREAD_LOCAL __thread
#else
#define BITGRAIN_THREAD_LOCAL _Thread_local
#endif

#define BITGRAIN_LOAD_ERR_CAP 192
static BITGRAIN_THREAD_LOCAL char g_load_error[BITGRAIN_LOAD_ERR_CAP] = "no loader error";

static void load_set_error(const char *msg)
{
    if (!msg) msg = "unknown image load error";
    size_t n = strlen(msg);
    if (n >= sizeof(g_load_error)) n = sizeof(g_load_error) - 1u;
    memcpy(g_load_error, msg, n);
    g_load_error[n] = '\0';
}

static void load_set_webp_error(void)
{
    const char *why = bitgrain_webp_last_error();
    char composed[BITGRAIN_LOAD_ERR_CAP];
    snprintf(composed, sizeof(composed), "WebP: %s", why ? why : "decode failed");
    load_set_error(composed);
}

const char *bitgrain_image_last_error(void)
{
    return g_load_error;
}

static int has_webp_extension(const char *path)
{
    if (!path) return 0;
    const char *dot = strrchr(path, '.');
    return dot && strcasecmp(dot, ".webp") == 0;
}

/* ------------------------------------------------------------------ */
/* Stream reader                                                        */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_read_stream(FILE *f, size_t *out_size)
{
    size_t cap = 65536;
    size_t len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;

    size_t n;
    while ((n = fread(buf + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap) {
            cap *= 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
    }
    *out_size = len;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Grayscale                                                            */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_grayscale(const char *path,
                                  uint32_t *out_width,
                                  uint32_t *out_height)
{
    if (has_webp_extension(path)) {
        size_t raw_size = 0;
        FILE *f = fopen(path, "rb");
        uint8_t *raw = f ? bitgrain_read_stream(f, &raw_size) : NULL;
        if (f) fclose(f);
        if (raw) {
            if (bitgrain_is_webp_mem(raw, raw_size)) {
                uint8_t *gray = bitgrain_load_grayscale_mem(raw, raw_size, out_width, out_height);
                free(raw);
                if (gray) return gray;
                return NULL;
            }
            free(raw);
        } else {
            load_set_error("cannot open image file");
            return NULL;
        }
    }
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 1);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_grayscale_mem(const uint8_t *mem, size_t mem_size,
                                      uint32_t *out_width, uint32_t *out_height)
{
    if (bitgrain_is_webp_mem(mem, mem_size)) {
        /* No native grayscale WebP: decode RGB and let the caller's 1-channel
         * stb path stay untouched; convert here via the RGB loader + luma. */
        uint8_t *rgb = bitgrain_load_webp_rgb_mem(mem, mem_size, out_width, out_height);
        if (!rgb) { load_set_webp_error(); return NULL; }
        size_t npix = (size_t)*out_width * (size_t)*out_height;
        uint8_t *gray = (uint8_t *)malloc(npix);
        if (!gray) {
            stbi_image_free(rgb);
            load_set_error("out of memory allocating grayscale pixels");
            return NULL;
        }
        for (size_t i = 0; i < npix; i++) {
            const uint8_t *p = rgb + i * 3u;
            gray[i] = (uint8_t)((299 * p[0] + 587 * p[1] + 114 * p[2] + 500) / 1000);
        }
        stbi_image_free(rgb);
        return gray;
    }
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 1);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

/* ------------------------------------------------------------------ */
/* RGB                                                                  */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_rgb(const char *path,
                           uint32_t *out_width,
                           uint32_t *out_height)
{
    if (has_webp_extension(path)) {
        uint32_t w = 0, h = 0;
        int has_alpha = 0;
        size_t probe_size = 0;
        FILE *f = fopen(path, "rb");
        uint8_t *raw = f ? bitgrain_read_stream(f, &probe_size) : NULL;
        if (f) fclose(f);
        if (raw) {
            if (bitgrain_webp_probe_mem(raw, probe_size, &w, &h, &has_alpha) == 0) {
                if (has_alpha) {
                    /* RGB request must silently drop an alpha-bearing WebP so the
                     * caller's RGBA fallback keeps the alpha plane intact. */
                    free(raw);
                    load_set_error("WebP image contains an alpha channel; use the RGBA loader");
                    return NULL;
                }
                uint8_t *data = bitgrain_load_webp_rgb_mem(raw, probe_size, out_width, out_height);
                free(raw);
                if (data) return data;
                load_set_webp_error();
                return NULL;
            }
            free(raw);
        }
        /* Fall through to stb_image for non-container diagnostics. */
    }
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 3);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_rgb_mem(const uint8_t *mem, size_t mem_size,
                                uint32_t *out_width, uint32_t *out_height)
{
    if (bitgrain_is_webp_mem(mem, mem_size)) {
        uint32_t w = 0, h = 0;
        int has_alpha = 0;
        if (bitgrain_webp_probe_mem(mem, mem_size, &w, &h, &has_alpha) == 0) {
            if (has_alpha) {
                load_set_error("WebP image contains an alpha channel; use the RGBA loader");
                return NULL;
            }
            uint8_t *data = bitgrain_load_webp_rgb_mem(mem, mem_size, out_width, out_height);
            if (data) return data;
            load_set_webp_error();
            return NULL;
        }
        load_set_webp_error();
        return NULL;
    }
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 3);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

/* ------------------------------------------------------------------ */
/* RGBA                                                                 */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_load_rgba(const char *path,
                            uint32_t *out_width,
                            uint32_t *out_height)
{
    if (has_webp_extension(path)) {
        size_t raw_size = 0;
        FILE *f = fopen(path, "rb");
        uint8_t *raw = f ? bitgrain_read_stream(f, &raw_size) : NULL;
        if (f) fclose(f);
        if (raw) {
            if (bitgrain_is_webp_mem(raw, raw_size)) {
                uint8_t *data = bitgrain_load_webp_rgba_mem(raw, raw_size, out_width, out_height);
                free(raw);
                if (data) return data;
                load_set_webp_error();
                return NULL;
            }
            free(raw);
        } else {
            load_set_error("cannot open image file");
            return NULL;
        }
    }
    int w, h, n;
    unsigned char *data = stbi_load(path, &w, &h, &n, 4);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

uint8_t *bitgrain_load_rgba_mem(const uint8_t *mem, size_t mem_size,
                                 uint32_t *out_width, uint32_t *out_height)
{
    if (bitgrain_is_webp_mem(mem, mem_size)) {
        uint8_t *data = bitgrain_load_webp_rgba_mem(mem, mem_size, out_width, out_height);
        if (data) return data;
        load_set_webp_error();
        return NULL;
    }
    int w, h, n;
    unsigned char *data = stbi_load_from_memory(mem, (int)mem_size, &w, &h, &n, 4);
    if (!data || w <= 0 || h <= 0) {
        load_set_error(stbi_failure_reason() ? stbi_failure_reason() : "stb_image failed to load image");
        stbi_image_free(data);
        return NULL;
    }
    *out_width  = (uint32_t)w;
    *out_height = (uint32_t)h;
    return (uint8_t *)data;
}

void bitgrain_image_free(void *pixels)
{
    stbi_image_free(pixels);
}
