#!/usr/bin/env python3
"""Tiny deterministic RGB samples for the chroma-subsampling verification.
Two 32x32 images: a natural-ish smooth gradient and a high-frequency
checkerboard, so both typical and worst-case chroma behavior are visible.
Writes PNGs using only zlib/struct from the stdlib."""
import struct, zlib, os

OUT = os.path.dirname(os.path.abspath(__file__))

def write_png(path, w, h, rows):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 2 = truecolor RGB
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)

def smooth(w, h):
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            # slow-varying gradients + mild sine detail, mostly low frequency
            r = int(128 + 90 * math_sin(x * 0.25))
            g = int(128 + 90 * math_sin(y * 0.21 + 1.0))
            b = int(128 + 80 * math_sin((x + y) * 0.13 + 2.0))
            row += bytes((r & 255, g & 255, b & 255))
        rows.append(row)
    return rows

def checker(w, h):
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            r = (x * 11 + y * 3 + ((x ^ y) & 1) * 90 + 17) & 255
            g = (y * 13 + x * 5 + ((x + y) & 1) * 70 + 34) & 255
            b = ((x + y) * 7 + ((x ^ y) & 1) * 60 + 51) & 255
            row += bytes((r, g, b))
        rows.append(row)
    return rows

def math_sin(v):
    # avoid importing math at module level twice; tiny sin via stdlib
    import math
    return math.sin(v)

write_png(os.path.join(OUT, "smooth32.png"), 32, 32, smooth(32, 32))
write_png(os.path.join(OUT, "checker32.png"), 32, 32, checker(32, 32))
print("samples written")
