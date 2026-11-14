use crate::chroma::ChromaSubsampling;
use crate::decoder;
use crate::encoder;

fn rgb_image(w: usize, h: usize) -> Vec<u8> {
    let mut img = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            img.push((x * 17) as u8);
            img.push((y * 29) as u8);
            img.push(((x + y) * 13) as u8);
        }
    }
    img
}

fn rgba_image(w: usize, h: usize) -> Vec<u8> {
    let mut img = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            img.push((x * 17) as u8);
            img.push((y * 29) as u8);
            img.push(((x + y) * 13) as u8);
            img.push(((x * 9 + y * 5) ^ 0x55) as u8);
        }
    }
    img
}

#[test]
fn chroma_mode_roundtrip_all_versions() {
    for (w, h) in [(16usize, 16usize), (9, 7), (8, 1), (1, 8)] {
        let image = rgb_image(w, h);
        for mode in [
            ChromaSubsampling::H2V2,
            ChromaSubsampling::H2V1,
            ChromaSubsampling::H1V1,
        ] {
            let mut buf = vec![0u8; w * h * 3 * 2 + 4096];
            let mut pos = 0i32;
            encoder::encode_rgb_ycbcr_mode(
                &image, w, h, 85, &mut buf, &mut pos, None, mode,
            );
            let stream = &buf[..pos as usize];
            let version = stream[2];
            assert!(version >= 18 && version <= 22);

            let mut out = vec![0u8; w * h * 3];
            let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
            let ok = decoder::decode(
                stream, &mut out, &mut ow, &mut oh, &mut oc, None,
            );
            assert!(ok, "decode failed for {w}x{h} mode {mode:?} version {version}");
            assert_eq!((ow as usize, oh as usize, oc), (w, h, 3));
        }
    }
}

#[test]
fn chroma_dims_and_cells_match() {
    assert_eq!(ChromaSubsampling::H1V1.chroma_dims(9, 7), (9, 7));
    assert_eq!(ChromaSubsampling::H2V1.chroma_dims(9, 7), (5, 7));
    assert_eq!(ChromaSubsampling::H2V2.chroma_dims(9, 7), (5, 4));
}

#[test]
fn default_env_is_420() {
    std::env::remove_var("BITGRAIN_CHROMA");
    assert_eq!(ChromaSubsampling::from_env(), ChromaSubsampling::H2V2);
    std::env::set_var("BITGRAIN_CHROMA", "422");
    assert_eq!(ChromaSubsampling::from_env(), ChromaSubsampling::H2V1);
    std::env::set_var("BITGRAIN_CHROMA", "444");
    assert_eq!(ChromaSubsampling::from_env(), ChromaSubsampling::H1V1);
    std::env::set_var("BITGRAIN_CHROMA", "garbage");
    assert_eq!(ChromaSubsampling::from_env(), ChromaSubsampling::H2V2);
    std::env::remove_var("BITGRAIN_CHROMA");
}

#[test]
fn h2v2_legacy_bytes_unchanged() {
    let (w, h) = (16usize, 16usize);
    let image = rgb_image(w, h);
    let mut legacy = vec![0u8; w * h * 3 * 2 + 4096];
    let mut p1 = 0i32;
    encoder::encode_rgb_ycbcr(&image, w, h, 85, &mut legacy, &mut p1, None);

    let mut via_mode = vec![0u8; w * h * 3 * 2 + 4096];
    let mut p2 = 0i32;
    encoder::encode_rgb_ycbcr_mode(
        &image, w, h, 85, &mut via_mode, &mut p2, None, ChromaSubsampling::H2V2,
    );
    assert_eq!(&legacy[..p1 as usize], &via_mode[..p2 as usize]);
}

#[test]
fn rgba_chroma_mode_roundtrip_all_versions() {
    for (w, h) in [(16usize, 16usize), (9, 7)] {
        let image = rgba_image(w, h);
        for mode in [
            ChromaSubsampling::H2V2,
            ChromaSubsampling::H2V1,
            ChromaSubsampling::H1V1,
        ] {
            let mut buf = vec![0u8; w * h * 4 * 2 + 4096];
            let mut pos = 0i32;
            encoder::encode_rgba_ycbcr_mode(
                &image, w, h, 85, &mut buf, &mut pos, None, mode,
            );
            let stream = &buf[..pos as usize];
            let version = stream[2];
            assert!([19u8, 21, 23].contains(&version));

            let mut out = vec![0u8; w * h * 4];
            let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
            let ok = decoder::decode(
                stream, &mut out, &mut ow, &mut oh, &mut oc, None,
            );
            assert!(ok, "RGBA decode failed for {w}x{h} mode {mode:?}");
            assert_eq!((ow as usize, oh as usize, oc), (w, h, 4));
            // Alpha is a full-resolution plane in every mode; its
            // reconstruction quality is independent of chroma mode (lossy
            // DCT/quant, same profile as luma). Exactness for a uniform
            // plane is covered by uniform_alpha_is_exact_in_every_mode.
        }
    }
}

#[test]
fn rgba_h2v2_legacy_bytes_unchanged() {
    let (w, h) = (16usize, 16usize);
    let image = rgba_image(w, h);
    let mut legacy = vec![0u8; w * h * 4 * 2 + 4096];
    let mut p1 = 0i32;
    encoder::encode_rgba_ycbcr(&image, w, h, 85, &mut legacy, &mut p1, None);

    let mut via_mode = vec![0u8; w * h * 4 * 2 + 4096];
    let mut p2 = 0i32;
    encoder::encode_rgba_ycbcr_mode(
        &image, w, h, 85, &mut via_mode, &mut p2, None, ChromaSubsampling::H2V2,
    );
    assert_eq!(&legacy[..p1 as usize], &via_mode[..p2 as usize]);
}

#[test]
fn uniform_alpha_is_exact_in_every_mode() {
    let (w, h) = (16usize, 16usize);
    let mut image = rgba_image(w, h);
    for p in 0..(w * h) {
        image[p * 4 + 3] = 137;
    }
    for mode in [
        ChromaSubsampling::H2V2,
        ChromaSubsampling::H2V1,
        ChromaSubsampling::H1V1,
    ] {
        let mut buf = vec![0u8; w * h * 4 * 2 + 4096];
        let mut pos = 0i32;
        encoder::encode_rgba_ycbcr_mode(&image, w, h, 85, &mut buf, &mut pos, None, mode);
        let stream = &buf[..pos as usize];
        let mut out = vec![0u8; w * h * 4];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        assert!(decoder::decode(stream, &mut out, &mut ow, &mut oh, &mut oc, None));
        for p in 0..(w * h) {
            assert_eq!(out[p * 4 + 3], 137, "uniform alpha changed in {mode:?}");
        }
    }
}
