#!/usr/bin/env python3
"""PSNR + size report over the three chroma modes. Stdlib only."""
import math, os, struct, sys, zlib

HERE = sys.argv[1]

def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, w, h, color = 8, 0, 0, 2
    idat = b""
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos+4])[0]
        tag = data[pos+4:pos+8]
        body = data[pos+8:pos+8+ln]
        if tag == b"IHDR":
            w, h, _bit, color = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + ln
    raw = zlib.decompress(idat)
    channels = {2: 3, 6: 4}[color]
    stride = w * channels
    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _ in range(h):
        ftype = raw[p]; p += 1
        line = bytearray(raw[p:p+stride]); p += stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if ftype == 1: line[i] = (line[i] + a) & 255
            elif ftype == 2: line[i] = (line[i] + b) & 255
            elif ftype == 3: line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif ftype == 4:
                xp = a + b - c
                pa, pb, pc = abs(xp-a), abs(xp-b), abs(xp-c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out += line
        prev = line
    return w, h, channels, bytes(out)

def read_bmp(path):
    """Minimal uncompressed 24/32-bit BMP reader (stbi_write_bmp output)."""
    data = open(path, "rb").read()
    assert data[:2] == b"BM"
    offset = struct.unpack("<I", data[10:14])[0]
    w = struct.unpack("<i", data[18:22])[0]
    h = struct.unpack("<i", data[22:26])[0]
    bpp = struct.unpack("<H", data[28:30])[0]
    assert struct.unpack("<I", data[30:34])[0] == 0
    top_down = h < 0
    h = abs(h)
    channels = bpp // 8
    row_size = (w * bpp + 31) // 32 * 4
    rows = []
    for ry in range(h):
        src_y = ry if top_down else (h - 1 - ry)
        start = offset + src_y * row_size
        row = bytearray()
        for x in range(w):
            b, g, r = data[start+x*channels], data[start+x*channels+1], data[start+x*channels+2]
            row += bytes((r, g, b))
        rows.append(bytes(row))
    return w, h, b"".join(rows)

def psnr(orig, recon):
    n = min(len(orig), len(recon))
    sse = sum((orig[i] - recon[i]) ** 2 for i in range(n))
    if sse == 0:
        return float("inf")
    return 10 * math.log10((255.0 ** 2) * n / sse)

for name in ("smooth32", "checker32"):
    w, h, ch, orig = read_png(os.path.join(HERE, name + ".png"))
    print(f"\n{name}  ({w}x{h}, RGB input, quality 85)")
    print(f"{'mode':<7}{'version':>8}{'bytes':>8}{'vs 4:4:4':>10}{'PSNR dB':>10}")
    sizes = {}
    for mode, tag, ver in (("4:4:4", "444", 22), ("4:2:2", "422", 20), ("4:2:0", "420", 18)):
        bg = os.path.join(HERE, f"{name}-{tag}.bg")
        bmp = os.path.join(HERE, f"{name}-{tag}.bmp")
        size = os.path.getsize(bg)
        sizes[mode] = size
        actual_ver = open(bg, "rb").read(3)[2]
        assert actual_ver == ver, (mode, actual_ver, ver)
        _, _, recon = read_bmp(bmp)
        p = psnr(orig, recon)
        print(f"{mode:<7}{ver:>8}{size:>8}{'':>10}{'inf' if math.isinf(p) else f'{p:.2f}':>10}")
    base = sizes["4:4:4"]
    print("share of 4:4:4 size:",
          "  ".join(f"{m} {sizes[m]/base*100:.1f}%" for m in ("4:4:4", "4:2:2", "4:2:0")))
