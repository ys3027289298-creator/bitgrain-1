/* SPDX-License-Identifier: GPL-3.0-or-later */
/* TIFF reader: uncompressed (Compression=1) and LZW (Compression=5),
 * 8/16-bit samples, grayscale (Photometric 0/1) and RGB(A) (Photometric 2),
 * strip-based, Predictor 1/2. 16-bit samples are reduced to the internal
 * 8-bit format via >>8 (same rule stb_image uses). No external deps. */

#include "tiff_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define TIFF_MAX_FILE   (256u * 1024u * 1024u)  /* same cap as webp_io */
#define TIFF_MAX_PIXELS (1u << 28)              /* 268M px sanity cap */

/* ------------------------------------------------------------------ */
/* Error reporting (loader API returns NULL; details go to stderr).    */
/* The CLI retries the same file under RGB/RGBA/gray loaders, so an    */
/* identical consecutive message is printed only once.                 */
/* ------------------------------------------------------------------ */

static void tiff_err(const char *fmt, ...)
{
    static _Thread_local char last[256];
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (strcmp(msg, last) == 0) return;
    memcpy(last, msg, sizeof last);
    fprintf(stderr, "Error: TIFF: %s.\n", msg);
}

/* ------------------------------------------------------------------ */
/* Probing                                                              */
/* ------------------------------------------------------------------ */

int bitgrain_tiff_probe_mem(const uint8_t *mem, size_t size)
{
    if (!mem || size < 4) return 0;
    return (mem[0] == 'I' && mem[1] == 'I' && mem[2] == 42 && mem[3] == 0) ||
           (mem[0] == 'M' && mem[1] == 'M' && mem[2] == 0 && mem[3] == 42);
}

int bitgrain_tiff_probe_path(const char *path)
{
    uint8_t sig[4];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(sig, 1, sizeof sig, f);
    fclose(f);
    return bitgrain_tiff_probe_mem(sig, n);
}

/* ------------------------------------------------------------------ */
/* Reader context                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *d;
    size_t n;
    int le; /* 1 = little-endian ("II"), 0 = big-endian ("MM") */
} tiff_t;

static uint16_t rd16(const tiff_t *t, size_t off)
{
    if (t->le) return (uint16_t)(t->d[off] | (t->d[off + 1] << 8));
    return (uint16_t)((t->d[off] << 8) | t->d[off + 1]);
}

static uint32_t rd32(const tiff_t *t, size_t off)
{
    if (t->le)
        return (uint32_t)t->d[off] | ((uint32_t)t->d[off + 1] << 8) |
               ((uint32_t)t->d[off + 2] << 16) | ((uint32_t)t->d[off + 3] << 24);
    return ((uint32_t)t->d[off] << 24) | ((uint32_t)t->d[off + 1] << 16) |
           ((uint32_t)t->d[off + 2] << 8) | (uint32_t)t->d[off + 3];
}

/* IFD field types we read (SHORT/LONG only; everything else is ignored). */
#define T_SHORT 3
#define T_LONG  4

/* Find an IFD entry. Returns entry byte offset in the file, or 0 if absent. */
static size_t find_tag(const tiff_t *t, size_t ifd_off, uint16_t count, uint16_t tag)
{
    for (uint32_t i = 0; i < count; i++) {
        size_t e = ifd_off + 2 + (size_t)i * 12;
        if (rd16(t, e) == tag) return e;
    }
    return 0;
}

/* Read a SHORT/LONG tag as an array of uint32. Returns malloc'd array
 * (caller frees) and sets *out_count, or NULL if absent/invalid. */
static uint32_t *tag_array(const tiff_t *t, size_t ifd_off, uint16_t ifd_count,
                           uint16_t tag, uint32_t *out_count)
{
    size_t e = find_tag(t, ifd_off, ifd_count, tag);
    if (!e) return NULL;
    uint16_t type = rd16(t, e + 2);
    uint32_t count = rd32(t, e + 4);
    size_t unit = (type == T_SHORT) ? 2 : (type == T_LONG) ? 4 : 0;
    if (!unit || count == 0 || count > (1u << 22)) return NULL;

    size_t vals_off;
    if ((size_t)count * unit <= 4) {
        vals_off = e + 8; /* inline, left-justified in the 4-byte field */
    } else {
        uint32_t off = rd32(t, e + 8);
        if ((uint64_t)off + (uint64_t)count * unit > t->n) return NULL;
        vals_off = off;
    }
    uint32_t *vals = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
    if (!vals) return NULL;
    for (uint32_t i = 0; i < count; i++)
        vals[i] = (type == T_SHORT) ? rd16(t, vals_off + (size_t)i * 2)
                                    : rd32(t, vals_off + (size_t)i * 4);
    *out_count = count;
    return vals;
}

/* Scalar tag with default. */
static uint32_t tag_u32(const tiff_t *t, size_t ifd_off, uint16_t ifd_count,
                        uint16_t tag, uint32_t dflt)
{
    uint32_t count = 0;
    uint32_t *v = tag_array(t, ifd_off, ifd_count, tag, &count);
    if (!v || count < 1) { free(v); return dflt; }
    uint32_t r = v[0];
    free(v);
    return r;
}

/* ------------------------------------------------------------------ */
/* LZW decoder (TIFF 6.0 variant: MSB-first codes, 9..12 bits,
 * "early change": width grows when the decoder table gains entry
 * 511/1023/2047, i.e. one code before the current width would fill up) */
/* ------------------------------------------------------------------ */

#define LZW_CLEAR 256
#define LZW_EOI   257
#define LZW_FIRST 258
#define LZW_MAX   4096

static int lzw_decode(const uint8_t *src, size_t src_len,
                      uint8_t *dst, size_t dst_need,
                      char *err, size_t errsz)
{
    uint16_t prefix[LZW_MAX];
    uint8_t  suffix[LZW_MAX];
    uint8_t  tmp[LZW_MAX + 1];
    uint8_t  rev[LZW_MAX + 1];
    size_t pos = 0, out = 0;
    uint32_t bitbuf = 0;
    int bitcnt = 0, width = 9, next = LZW_FIRST, prev = -1;

    for (;;) {
        while (bitcnt < width) {
            if (pos >= src_len) {
                if (out == dst_need) return 0; /* tolerate missing EOI */
                snprintf(err, errsz, "LZW stream ended early "
                         "(%zu of %zu bytes decoded)", out, dst_need);
                return -1;
            }
            bitbuf = (bitbuf << 8) | src[pos++];
            bitcnt += 8;
        }
        bitcnt -= width;
        int code = (int)((bitbuf >> bitcnt) & ((1u << width) - 1));
        bitbuf &= (uint32_t)(((uint64_t)1 << bitcnt) - 1);

        if (code == LZW_CLEAR) { width = 9; next = LZW_FIRST; prev = -1; continue; }
        if (code == LZW_EOI) break;

        int emit_code;
        int append_first = 0;
        if (code < next) {
            emit_code = code;
        } else if (code == next && prev >= 0) {
            emit_code = prev;      /* KwKwK: str(prev) + first(str(prev)) */
            append_first = 1;
        } else {
            snprintf(err, errsz, "invalid LZW code %d (table size %d)", code, next);
            return -1;
        }

        /* expand string for emit_code into tmp (forward order) */
        size_t n = 0;
        int c = emit_code;
        while (c >= 256) {
            if (n >= LZW_MAX) { snprintf(err, errsz, "LZW prefix loop"); return -1; }
            rev[n++] = suffix[c];
            c = prefix[c];
        }
        rev[n++] = (uint8_t)c;
        for (size_t i = 0; i < n; i++) tmp[i] = rev[n - 1 - i];
        size_t len = n;
        int first = tmp[0];
        if (append_first) tmp[len++] = (uint8_t)first;

        if (out + len > dst_need) {
            snprintf(err, errsz, "LZW data exceeds expected strip size "
                     "(%zu > %zu)", out + len, dst_need);
            return -1;
        }
        memcpy(dst + out, tmp, len);
        out += len;

        if (prev >= 0 && next < LZW_MAX) {
            prefix[next] = (uint16_t)prev;
            suffix[next] = (uint8_t)first;
            next++;
            /* TIFF "early change": the decoder's table lags the encoder's by
             * one entry, so widen one code before the width would fill up. */
            if (next == (1 << width) - 1 && width < 12) width++;
        }
        prev = code;
    }
    if (out != dst_need) {
        snprintf(err, errsz, "LZW decoded %zu bytes, expected %zu", out, dst_need);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Channel conversion to the internal 8-bit format                      */
/* ------------------------------------------------------------------ */

static uint8_t luma(uint32_t r, uint32_t g, uint32_t b)
{
    return (uint8_t)((r * 77 + g * 150 + b * 29) >> 8); /* stb's coefficients */
}

static uint8_t *convert8(const uint8_t *raw, uint32_t w, uint32_t h,
                         int spp, int req)
{
    size_t npix = (size_t)w * h;
    uint8_t *out = (uint8_t *)malloc(npix * (size_t)req);
    if (!out) return NULL;
    for (size_t i = 0; i < npix; i++) {
        const uint8_t *p = raw + i * (size_t)spp;
        uint8_t r, g, b, a;
        if (spp <= 2) { r = g = b = p[0]; a = (spp == 2) ? p[1] : 255; }
        else          { r = p[0]; g = p[1]; b = p[2]; a = (spp >= 4) ? p[3] : 255; }
        uint8_t *o = out + i * (size_t)req;
        if (req == 1)      o[0] = (spp <= 2) ? r : luma(r, g, b);
        else if (req == 3) { o[0] = r; o[1] = g; o[2] = b; }
        else               { o[0] = r; o[1] = g; o[2] = b; o[3] = a; }
    }
    return out;
}

static uint8_t *convert16(const uint16_t *img, uint32_t w, uint32_t h,
                          int spp, int req)
{
    size_t npix = (size_t)w * h;
    uint8_t *out = (uint8_t *)malloc(npix * (size_t)req);
    if (!out) return NULL;
    for (size_t i = 0; i < npix; i++) {
        const uint16_t *p = img + i * (size_t)spp;
        uint8_t r, g, b, a;
        if (spp <= 2) {
            r = g = b = (uint8_t)(p[0] >> 8);
            a = (spp == 2) ? (uint8_t)(p[1] >> 8) : 255;
        } else {
            r = (uint8_t)(p[0] >> 8);
            g = (uint8_t)(p[1] >> 8);
            b = (uint8_t)(p[2] >> 8);
            a = (spp >= 4) ? (uint8_t)(p[3] >> 8) : 255;
        }
        uint8_t *o = out + i * (size_t)req;
        if (req == 1)      o[0] = (spp <= 2) ? r : luma(r, g, b);
        else if (req == 3) { o[0] = r; o[1] = g; o[2] = b; }
        else               { o[0] = r; o[1] = g; o[2] = b; o[3] = a; }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Main decode                                                          */
/* ------------------------------------------------------------------ */

uint8_t *bitgrain_tiff_load_mem(const uint8_t *mem, size_t size,
                                uint32_t *out_width, uint32_t *out_height,
                                int req_channels)
{
    tiff_t t;
    t.d = mem;
    t.n = size;

    if (!mem || size < 8 || !bitgrain_tiff_probe_mem(mem, size)) {
        tiff_err("not a TIFF file");
        return NULL;
    }
    t.le = (mem[0] == 'I');

    uint32_t ifd_off = rd32(&t, 4);
    if (ifd_off < 8 || (uint64_t)ifd_off + 2 > size) {
        tiff_err("bad IFD offset %u (file is %zu bytes)", ifd_off, size);
        return NULL;
    }
    uint16_t ifd_count = rd16(&t, ifd_off);
    if ((uint64_t)ifd_off + 2 + (uint64_t)ifd_count * 12 + 4 > size) {
        tiff_err("IFD with %u entries extends past end of file", ifd_count);
        return NULL;
    }

    /* ---- required geometry ---- */
    uint32_t w = tag_u32(&t, ifd_off, ifd_count, 256, 0);
    uint32_t h = tag_u32(&t, ifd_off, ifd_count, 257, 0);
    if (w == 0 || h == 0 || (uint64_t)w * h > TIFF_MAX_PIXELS) {
        tiff_err("bad or missing dimensions %ux%u", w, h);
        return NULL;
    }

    /* ---- bits per sample (all samples must share one depth: 8 or 16) ---- */
    uint32_t bits_count = 0;
    uint32_t *bits_arr = tag_array(&t, ifd_off, ifd_count, 258, &bits_count);
    uint32_t bits = bits_arr ? bits_arr[0] : 8;
    if (bits_arr)
        for (uint32_t i = 1; i < bits_count; i++)
            if (bits_arr[i] != bits) {
                free(bits_arr);
                tiff_err("mixed BitsPerSample not supported");
                return NULL;
            }
    free(bits_arr);
    if (bits != 8 && bits != 16) {
        tiff_err("unsupported BitsPerSample %u (only 8 and 16)", bits);
        return NULL;
    }

    uint32_t compression = tag_u32(&t, ifd_off, ifd_count, 259, 1);
    if (compression != 1 && compression != 5) {
        tiff_err("unsupported Compression %u (only 1=none and 5=LZW)", compression);
        return NULL;
    }

    uint32_t spp = tag_u32(&t, ifd_off, ifd_count, 277, 1);
    uint32_t photo = tag_u32(&t, ifd_off, ifd_count, 262, spp >= 3 ? 2 : 1);
    if (photo == 2) {
        if (spp != 3 && spp != 4) {
            tiff_err("RGB photometric with %u samples per pixel not supported", spp);
            return NULL;
        }
    } else if (photo <= 1) {
        if (spp != 1 && spp != 2) {
            tiff_err("grayscale photometric with %u samples per pixel not supported", spp);
            return NULL;
        }
    } else {
        tiff_err("unsupported PhotometricInterpretation %u "
                 "(only 0/1=grayscale and 2=RGB)", photo);
        return NULL;
    }

    uint32_t planar = tag_u32(&t, ifd_off, ifd_count, 284, 1);
    if (planar != 1) {
        tiff_err("planar (band-interleaved) images not supported");
        return NULL;
    }
    uint32_t predictor = tag_u32(&t, ifd_off, ifd_count, 317, 1);
    if (predictor != 1 && predictor != 2) {
        tiff_err("unsupported Predictor %u (only 1 and 2)", predictor);
        return NULL;
    }

    uint32_t rps = tag_u32(&t, ifd_off, ifd_count, 278, h);
    if (rps == 0) rps = h;
    uint32_t nstrips = (h + rps - 1) / rps;

    uint32_t off_count = 0, bc_count = 0;
    uint32_t *offsets = tag_array(&t, ifd_off, ifd_count, 273, &off_count);
    uint32_t *bytecnt = tag_array(&t, ifd_off, ifd_count, 279, &bc_count);
    if (!offsets || off_count < nstrips) {
        int present = find_tag(&t, ifd_off, ifd_count, 273) != 0;
        free(offsets); free(bytecnt);
        if (present)
            tiff_err("StripOffsets corrupt (entry points outside the file)");
        else
            tiff_err("missing/short StripOffsets (need %u strips)", nstrips);
        return NULL;
    }
    if (!bytecnt || bc_count < nstrips) {
        int present = find_tag(&t, ifd_off, ifd_count, 279) != 0;
        free(offsets); free(bytecnt);
        if (present)
            tiff_err("StripByteCounts corrupt (entry points outside the file)");
        else
            tiff_err("missing/short StripByteCounts (need %u strips)", nstrips);
        return NULL;
    }

    size_t bpp = (size_t)spp * (bits / 8);       /* bytes per pixel in file */
    size_t row_bytes = (size_t)w * bpp;
    uint8_t *raw = (uint8_t *)malloc(row_bytes * h);
    if (!raw) {
        free(offsets); free(bytecnt);
        tiff_err("out of memory");
        return NULL;
    }

    /* ---- decode strips ---- */
    for (uint32_t s = 0; s < nstrips; s++) {
        uint32_t rows = (s == nstrips - 1) ? (h - s * rps) : rps;
        size_t need = (size_t)rows * row_bytes;
        uint64_t off = offsets[s];
        uint64_t cnt = bytecnt[s];
        if (off > size || cnt > size - off) {
            tiff_err("strip %u corrupt: offset %llu + byte count %llu out of "
                     "range (file is %zu bytes)",
                     s, (unsigned long long)off, (unsigned long long)cnt, size);
            goto fail;
        }
        uint8_t *dst = raw + (size_t)s * rps * row_bytes;
        if (compression == 1) {
            if (cnt < need) {
                tiff_err("strip %u corrupt: truncated (%llu bytes stored, "
                         "%zu expected)", s, (unsigned long long)cnt, need);
                goto fail;
            }
            memcpy(dst, mem + off, need);
        } else {
            char lerr[128] = {0};
            if (lzw_decode(mem + off, (size_t)cnt, dst, need,
                           lerr, sizeof lerr) != 0) {
                tiff_err("strip %u corrupt: %s", s, lerr);
                goto fail;
            }
        }
    }
    free(offsets);
    free(bytecnt);

    /* ---- horizontal differencing predictor (per row, per sample) ---- */
    if (predictor == 2) {
        if (bits == 8) {
            for (uint32_t y = 0; y < h; y++) {
                uint8_t *row = raw + (size_t)y * row_bytes;
                for (size_t i = bpp; i < row_bytes; i++)
                    row[i] = (uint8_t)(row[i] + row[i - bpp]);
            }
        } else {
            /* operate on host-order u16 below; handled after endian fixup */
        }
    }

    uint8_t *out = NULL;
    if (bits == 8) {
        if (photo == 0) { /* WhiteIsZero */
            size_t total = row_bytes * h;
            for (size_t i = 0; i < total; i++) raw[i] = (uint8_t)(255 - raw[i]);
        }
        out = convert8(raw, w, h, (int)spp, req_channels);
    } else {
        size_t nsamp = (size_t)w * h * spp;
        uint16_t *img = (uint16_t *)malloc(nsamp * sizeof(uint16_t));
        if (!img) { free(raw); tiff_err("out of memory"); return NULL; }
        for (size_t i = 0; i < nsamp; i++) {
            uint16_t v = t.le ? (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8))
                              : (uint16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);
            img[i] = v;
        }
        free(raw);
        if (predictor == 2) {
            size_t rb = (size_t)w * spp;
            for (uint32_t y = 0; y < h; y++) {
                uint16_t *row = img + (size_t)y * rb;
                for (size_t i = spp; i < rb; i++)
                    row[i] = (uint16_t)(row[i] + row[i - spp]);
            }
        }
        if (photo == 0)
            for (size_t i = 0; i < nsamp; i++) img[i] = (uint16_t)(65535 - img[i]);
        out = convert16(img, w, h, (int)spp, req_channels);
        free(img);
        raw = NULL;
    }
    free(raw);

    if (!out) { tiff_err("out of memory"); return NULL; }
    *out_width = w;
    *out_height = h;
    return out;

fail:
    free(offsets);
    free(bytecnt);
    free(raw);
    return NULL;
}

uint8_t *bitgrain_tiff_load(const char *path,
                            uint32_t *out_width, uint32_t *out_height,
                            int req_channels)
{
    FILE *f = fopen(path, "rb");
    if (!f) { tiff_err("cannot open '%s'", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || (unsigned long)sz > TIFF_MAX_FILE) {
        fclose(f);
        tiff_err("bad file size %ld for '%s'", sz, path);
        return NULL;
    }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); tiff_err("out of memory"); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        tiff_err("could not read '%s'", path);
        return NULL;
    }
    fclose(f);
    uint8_t *out = bitgrain_tiff_load_mem(buf, (size_t)sz,
                                          out_width, out_height, req_channels);
    free(buf);
    return out;
}
