/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "bg_utils.h"
#include "config.h"

int parse_bg_header(const uint8_t *buf, uint32_t *width, uint32_t *height, uint32_t *channels)
{
    if (buf[0] != 'B' || buf[1] != 'G') return -1;
    uint8_t ver = buf[2];
    if (ver < 1 || ver > 23) return -1;
    *width   = (uint32_t)buf[3] | ((uint32_t)buf[4]<<8) | ((uint32_t)buf[5]<<16) | ((uint32_t)buf[6]<<24);
    *height  = (uint32_t)buf[7] | ((uint32_t)buf[8]<<8) | ((uint32_t)buf[9]<<16) | ((uint32_t)buf[10]<<24);
    /* v1=gray(1ch), v2=RGB(3ch), v3=RGBA(4ch),
       even v4..v22 = YCbCr RGB profiles (3ch), odd v5..v23 = +A RGBA profiles (4ch);
       v18/v19=4:2:0, v20/v21=4:2:2, v22/v23=4:4:4 */
    switch (ver) {
        case 1: *channels = 1; break;
        case 2: *channels = 3; break;
        case 3: *channels = 4; break;
        case 4: *channels = 3; break;  /* YCbCr420 decodes to RGB */
        case 5: *channels = 4; break;  /* YCbCr420A decodes to RGBA */
        case 6: *channels = 3; break;  /* YCbCr420 + chroma AC decodes to RGB */
        case 7: *channels = 4; break;  /* YCbCr420A + chroma AC decodes to RGBA */
        case 8: *channels = 3; break;  /* perceptual quant + chroma AC decodes to RGB */
        case 9: *channels = 4; break;  /* perceptual quant + chroma AC decodes to RGBA */
        case 10: *channels = 3; break; /* perceptual + chroma AC + DC delta decodes to RGB */
        case 11: *channels = 4; break; /* perceptual + chroma AC + DC delta decodes to RGBA */
        case 12: *channels = 3; break; /* stronger perceptual profile decodes to RGB */
        case 13: *channels = 4; break; /* stronger perceptual profile decodes to RGBA */
        case 14: *channels = 3; break; /* aggressive perceptual profile decodes to RGB */
        case 15: *channels = 4; break; /* aggressive perceptual profile decodes to RGBA */
        case 16: *channels = 3; break; /* very aggressive perceptual profile decodes to RGB */
        case 17: *channels = 4; break; /* very aggressive perceptual profile decodes to RGBA */
        case 18: *channels = 3; break; /* ultra profile decodes to RGB */
        case 19: *channels = 4; break; /* ultra profile decodes to RGBA */
        case 20: *channels = 3; break; /* YCbCr422 decodes to RGB */
        case 21: *channels = 4; break; /* YCbCr422A decodes to RGBA */
        case 22: *channels = 3; break; /* YCbCr444 decodes to RGB */
        case 23: *channels = 4; break; /* YCbCr444A decodes to RGBA */
        default: return -1;
    }
    return 0;
}

int check_image_size(uint32_t width, uint32_t height, uint32_t channels)
{
    if (width == 0 || height == 0 || width > BITGRAIN_MAX_DIM || height > BITGRAIN_MAX_DIM) return -1;
    uint64_t bytes = (uint64_t)width * height * channels;
    if (bytes > BITGRAIN_MAX_PIXEL_BYTES) return -1;
    return 0;
}
