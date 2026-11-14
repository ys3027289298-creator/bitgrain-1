/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Quantization-dithering verification harness (public C API only).
 *
 * Builds a small, deliberately banding-prone grayscale gradient in memory
 * (256x8, one input step every 32 pixels), encodes/decodes it under every
 * dither configuration and reports two banding metrics on the gradient row:
 *
 *   levels  - distinct reconstructed gray levels (the requested banding
 *             metric; dithering raises this versus the undithered baseline),
 *   maxrun  - longest run of identical adjacent pixels (the width of the
 *             widest visible band; dithering shortens or holds this),
 *
 * plus the encoded .bg byte size. It also verifies that disabling dithering
 * restores the baseline and that out-of-range strength / unknown modes are
 * rejected with BITGRAIN_ERR_INVALID_ARG.
 */

#include "encoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 256
#define HEIGHT 8
#define RAMP_STEP 32
#define QUALITY 5

static void build_gradient(uint8_t *image, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t x = (uint32_t)(i % WIDTH);
        image[i] = (uint8_t)((x / RAMP_STEP) * 255u / ((WIDTH - 1u) / RAMP_STEP));
    }
}

static size_t count_levels(const uint8_t *decoded)
{
    uint8_t seen[256];
    memset(seen, 0, sizeof(seen));
    const uint8_t *row = decoded + (size_t)(HEIGHT / 2) * WIDTH;
    size_t distinct = 0;
    for (uint32_t x = 0; x < WIDTH; x++) {
        if (!seen[row[x]]) {
            seen[row[x]] = 1;
            distinct++;
        }
    }
    return distinct;
}

static size_t longest_run(const uint8_t *decoded)
{
    const uint8_t *row = decoded + (size_t)(HEIGHT / 2) * WIDTH;
    size_t best = 1, cur = 1;
    for (uint32_t x = 1; x < WIDTH; x++) {
        if (row[x] == row[x - 1]) {
            cur++;
            if (cur > best)
                best = cur;
        } else {
            cur = 1;
        }
    }
    return best;
}

static int run_case(int mode, int strength, const uint8_t *image,
                    size_t pixel_count, size_t *levels, size_t *run,
                    size_t *bytes)
{
    if (bitgrain_set_dither(mode, strength) != 0) {
        fprintf(stderr, "set_dither(%d,%d) rejected: %s\n",
                mode, strength, bitgrain_last_error_message());
        return -1;
    }

    size_t cap = pixel_count * 2 + 1024 * 1024;
    uint8_t *bg = (uint8_t *)malloc(cap);
    uint8_t *decoded = (uint8_t *)malloc(pixel_count);
    if (!bg || !decoded) {
        free(bg);
        free(decoded);
        return -1;
    }

    int32_t encoded_len = 0;
    if (bitgrain_encode_grayscale(image, WIDTH, HEIGHT, bg, (uint32_t)cap,
                                  &encoded_len, QUALITY) != 0) {
        fprintf(stderr, "encode failed: %s\n", bitgrain_last_error_message());
        free(bg);
        free(decoded);
        return -1;
    }

    uint32_t dw = 0, dh = 0, dc = 0;
    if (bitgrain_decode(bg, encoded_len, decoded, (uint32_t)pixel_count,
                        &dw, &dh, &dc) != 0 ||
        dw != WIDTH || dh != HEIGHT || dc != 1) {
        fprintf(stderr, "decode failed\n");
        free(bg);
        free(decoded);
        return -1;
    }

    *levels = count_levels(decoded);
    *run = longest_run(decoded);
    *bytes = (size_t)encoded_len;
    free(bg);
    free(decoded);
    return 0;
}

static void print_row(const char *label, int strength, size_t levels,
                      size_t run, size_t bytes)
{
    printf("%-16s %8d %10zu %10zu %10zu\n",
           label, strength, levels, run, bytes);
}

int main(void)
{
    const size_t pixel_count = (size_t)WIDTH * HEIGHT;
    uint8_t *image = (uint8_t *)malloc(pixel_count);
    if (!image)
        return 2;
    build_gradient(image, pixel_count);

    static const int strengths[] = {25, 50, 75, 100};

    printf("quality=%d image=%ux%u ramp_step=%d metric=gradient_row\n",
           QUALITY, WIDTH, HEIGHT, RAMP_STEP);
    printf("%-16s %8s %10s %10s %10s\n",
           "config", "strength", "levels", "max_run", "bg_bytes");

    size_t base_levels = 0, base_run = 0, base_bytes = 0;
    if (run_case(BITGRAIN_DITHER_OFF, 0, image, pixel_count,
                 &base_levels, &base_run, &base_bytes) != 0)
        return 2;
    print_row("off", 0, base_levels, base_run, base_bytes);

    for (size_t i = 0; i < sizeof(strengths) / sizeof(strengths[0]); i++) {
        size_t levels = 0, run = 0, bytes = 0;
        if (run_case(BITGRAIN_DITHER_ORDERED, strengths[i], image,
                     pixel_count, &levels, &run, &bytes) != 0)
            return 2;
        print_row("ordered", strengths[i], levels, run, bytes);
        if (levels < base_levels) {
            fprintf(stderr, "ordered dither reduced levels below baseline\n");
            return 2;
        }
    }

    for (size_t i = 0; i < sizeof(strengths) / sizeof(strengths[0]); i++) {
        size_t levels = 0, run = 0, bytes = 0;
        if (run_case(BITGRAIN_DITHER_ERROR_DIFFUSION, strengths[i], image,
                     pixel_count, &levels, &run, &bytes) != 0)
            return 2;
        print_row("error-diffusion", strengths[i], levels, run, bytes);
        if (levels < base_levels) {
            fprintf(stderr, "error diffusion reduced levels below baseline\n");
            return 2;
        }
    }

    /* Clearing dither must restore the exact baseline metrics. */
    bitgrain_clear_dither();
    size_t clear_levels = 0, clear_run = 0, clear_bytes = 0;
    if (run_case(BITGRAIN_DITHER_OFF, 0, image, pixel_count,
                 &clear_levels, &clear_run, &clear_bytes) != 0)
        return 2;
    if (clear_levels != base_levels || clear_run != base_run ||
        clear_bytes != base_bytes) {
        fprintf(stderr, "clear_dither did not restore baseline behavior\n");
        return 2;
    }

    /* Failure paths. */
    if (bitgrain_set_dither(BITGRAIN_DITHER_ORDERED, 101) == 0 ||
        bitgrain_last_error_code() != BITGRAIN_ERR_INVALID_ARG) {
        fprintf(stderr, "strength 101 must be rejected with INVALID_ARG\n");
        return 2;
    }
    if (bitgrain_set_dither(9, 50) == 0 ||
        bitgrain_last_error_code() != BITGRAIN_ERR_INVALID_ARG) {
        fprintf(stderr, "unknown mode must be rejected with INVALID_ARG\n");
        return 2;
    }
    if (bitgrain_set_dither(BITGRAIN_DITHER_ERROR_DIFFUSION, -1) == 0 ||
        bitgrain_last_error_code() != BITGRAIN_ERR_INVALID_ARG) {
        fprintf(stderr, "negative strength must be rejected with INVALID_ARG\n");
        return 2;
    }

    printf("ALL CHECKS PASSED\n");
    free(image);
    return 0;
}
