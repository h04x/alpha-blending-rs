//use std::process::Command;
use image::{Rgba, RgbaImage};
use std::time::Instant;
use rand::Rng;

use std::mem::transmute;
use core::convert::TryInto;
use core::convert::TryFrom;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

fn gen_images(w: u32, h: u32) -> (RgbaImage, RgbaImage) {
    let mut im1 = RgbaImage::new(w, h);
    for p in im1.pixels_mut() {
        *p = Rgba([101, 102, 103, 255])
    }
    let mut im2 = RgbaImage::new(w, h);
    for p in im2.pixels_mut() {
        *p = Rgba([10, 217, 100, 200])
    }
    (im1, im2)
}

fn blend_on_floats_universal(im1: &mut RgbaImage, im2: &RgbaImage) {
    // from image crate
    let max_t = u8::max_value();
    let max_t = max_t as f32;
    for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) {
        let (bg_r, bg_g, bg_b, bg_a) = (im1p.0[0], im1p.0[1], im1p.0[2], im1p.0[3]);
        let (fg_r, fg_g, fg_b, fg_a) = (im2p.0[0], im2p.0[1], im2p.0[2], im2p.0[3]);
        let (bg_r, bg_g, bg_b, bg_a) = (
            bg_r as f32 / max_t,
            bg_g as f32 / max_t,
            bg_b as f32 / max_t,
            bg_a as f32 / max_t,
        );
        let (fg_r, fg_g, fg_b, fg_a) = (
            fg_r as f32 / max_t,
            fg_g as f32 / max_t,
            fg_b as f32 / max_t,
            fg_a as f32 / max_t,
        );

        // Work out what the final alpha level will be
        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };

        // We premultiply our channels by their alpha, as this makes it easier to calculate
        let (bg_r_a, bg_g_a, bg_b_a) = (bg_r * bg_a, bg_g * bg_a, bg_b * bg_a);
        let (fg_r_a, fg_g_a, fg_b_a) = (fg_r * fg_a, fg_g * fg_a, fg_b * fg_a);

        // Standard formula for src-over alpha compositing
        let (out_r_a, out_g_a, out_b_a) = (
            fg_r_a + bg_r_a * (1.0 - fg_a),
            fg_g_a + bg_g_a * (1.0 - fg_a),
            fg_b_a + bg_b_a * (1.0 - fg_a),
        );

        // Unmultiply the channels by our resultant alpha channel
        let (out_r, out_g, out_b) = (
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
        );

        // Cast back to our initial type on return
        *im1p = Rgba([
            (max_t * out_r) as u8,
            (max_t * out_g) as u8,
            (max_t * out_b) as u8,
            (max_t * alpha_final) as u8,
        ])
    }
}

fn blend_optimized_universal(im1: &mut RgbaImage, im2: &RgbaImage) {
    //for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) {
        for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) {
        let src_r = im2p.0[0] as u32;
        let src_g = im2p.0[1] as u32;
        let src_b = im2p.0[2] as u32;
        let src_a = im2p.0[3] as u32;

        let dst_r = im1p.0[0] as u32;
        let dst_g = im1p.0[1] as u32;
        let dst_b = im1p.0[2] as u32;
        let dst_a = im1p.0[3] as u32;

        // Premul src_color/256 * src_alpha/256
        let src_r_p = (src_r * src_a) >> 8;
        let src_g_p = (src_g * src_a) >> 8;
        let src_b_p = (src_b * src_a) >> 8;

        let dst_r_p = (dst_r * dst_a) >> 8;
        let dst_g_p = (dst_g * dst_a) >> 8;
        let dst_b_p = (dst_b * dst_a) >> 8;

        let src_a_not = 255 - src_a;

        let r = (src_r_p) + ((src_a_not * (dst_r_p)) >> 8);
        let g = (src_g_p) + ((src_a_not * (dst_g_p)) >> 8);
        let b = (src_b_p) + ((src_a_not * (dst_b_p)) >> 8);

        let alpha_final = dst_a + src_a - ((dst_a * src_a) >> 8);

        // demul
        let r = ((r as f32 / alpha_final as f32) * 256f32) as u8;
        let g = ((g as f32 / alpha_final as f32) * 256f32) as u8;
        let b = ((b as f32 / alpha_final as f32) * 256f32) as u8;

        // lookup table a bit more faster
        /*let r = lookup[r as usize][alpha_final as usize];
        let g = lookup[g as usize][alpha_final as usize];
        let b = lookup[b as usize][alpha_final as usize];*/

        *im1p = Rgba([
            r as u8,
            g as u8,
            b as u8,
            if alpha_final > 255 {
                255
            } else {
                alpha_final as u8
            },
        ]);
    }
}

fn blend_optimized_bg_opaque(im1: &mut RgbaImage, im2: &RgbaImage) {
    for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) {
        let src_r = im2p.0[0] as u32;
        let src_g = im2p.0[1] as u32;
        let src_b = im2p.0[2] as u32;
        let src_a = im2p.0[3] as u32;

        let dst_r = im1p.0[0] as u32;
        let dst_g = im1p.0[1] as u32;
        let dst_b = im1p.0[2] as u32;

        let src_a_not = 256 - src_a;

        let r = ((src_r * src_a) + (src_a_not * dst_r)) >> 8;
        let g = ((src_g * src_a) + (src_a_not * dst_g)) >> 8;
        let b = ((src_b * src_a) + (src_a_not * dst_b)) >> 8;

        *im1p = Rgba([r as u8, g as u8, b as u8, 255]);
    }
}


unsafe fn blend_unsafe_bg_opaque(im1: &mut [u8], im2: &[u8]) {

    for chunk in im1.chunks_exact_mut(4).zip(im2.chunks_exact(4)) {
        let src_a = *chunk.1.get_unchecked(3) as u32 ;
        let src_a_not = 255 - src_a;

            let dst_r = chunk.0.get_unchecked_mut(0);
            let src_r = *chunk.1.get_unchecked(0) as u32;
            let r = ((src_r * src_a) + (src_a_not * *dst_r as u32)) >> 8;
            *dst_r = r as u8;

            let dst_g = chunk.0.get_unchecked_mut(1) ;
            let src_g = *chunk.1.get_unchecked(1) as u32 ;
            let g = ((src_g * src_a) + (src_a_not * *dst_g as u32)) >> 8;
            *dst_g = g as u8;

            let dst_b = chunk.0.get_unchecked_mut(2) ;
            let src_b = *chunk.1.get_unchecked(2) as u32 ;
            let b = ((src_b * src_a) + (src_a_not * *dst_b as u32)) >> 8;
            *dst_b = b as u8;

    }
}

#[target_feature(enable = "sse2,ssse3")]
unsafe fn blend_sse2_ssse3(im1: &mut [u8], im2: &[u8]) {
    let (dst_prefix, dst_arr, dst_suffix) = im1.align_to_mut::<i64>();
    let (src_prefix, src_arr, src_suffix) = im2.align_to::<i64>();
    //println!("{:?} {:?}", dst_prefix.len(), dst_suffix.len());

    let rgb_shuffler:[u8;16] = [
        0b00000000,0b10000000,
        0b00000001,0b10000000,
        0b00000010,0b10000000,
        0b00000100,0b10000000,
        0b00000101,0b10000000,
        0b00000110,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000];
    //rgb_shuffler.reverse();
    let rgb_shuffler = _mm_load_si128(rgb_shuffler.as_ptr() as *const std::arch::x86_64::__m128i);


    let alpha_shuffler: [u8; 16] =  [
        0b00000011,0b10000000,
        0b00000011,0b10000000,
        0b00000011,0b10000000,
        0b00000111,0b10000000,
        0b00000111,0b10000000,
        0b00000111,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000];
    //alpha_shuffler.reverse();
    let alpha_shuffler = _mm_load_si128(alpha_shuffler.as_ptr() as *const std::arch::x86_64::__m128i);

    let unpacker: [u8; 16] = [
        0b00000001,0b00000011,
        0b00000101,0b10000000,
        0b00000111,0b00001001,
        0b00001011,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000];
    //unpacker.reverse();
    let unpacker = _mm_load_si128(unpacker.as_ptr() as *const std::arch::x86_64::__m128i);

    //let dst_arr = dst_arr.chunks_exact_mut(2);
    //let src_arr = src_arr.chunks_exact(2);
    for chunk in dst_arr.iter_mut().zip(src_arr)  {
        let src_pix_pack_2 = _mm_set_epi64x(
            0,
            *chunk.1
        );

        let dst_pix_pack_2 = _mm_set_epi64x(
            0,
            *chunk.0
        );

        let dec255 = _mm_set1_epi16(255 as i16);

        let src_pix_pack2_shuffle = _mm_shuffle_epi8(src_pix_pack_2, rgb_shuffler);
        let dst_pix_pack2_shuffle = _mm_shuffle_epi8(dst_pix_pack_2, rgb_shuffler);
        let src_alpha_shuffle = _mm_shuffle_epi8(src_pix_pack_2, alpha_shuffler);
        let src_alpha_not = _mm_subs_epu8(dec255, src_alpha_shuffle);
        let src_premul = _mm_mullo_epi16(src_pix_pack2_shuffle, src_alpha_shuffle);
        let dst_mul_alpha_not = _mm_mullo_epi16(dst_pix_pack2_shuffle, src_alpha_not);
        let added = _mm_add_epi16(src_premul, dst_mul_alpha_not);
        let unpacked = _mm_shuffle_epi8(added, unpacker);

        /*let xxx: (u16,u16,u16,u16,u16,u16,u16,u16) = transmute(added);
        println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                 xxx.0, xxx.1, xxx.2, xxx.3, xxx.4, xxx.5, xxx.6, xxx.7);*/

        /*let xxx: (u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8) = transmute(added);
        println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                xxx.0, xxx.1, xxx.2, xxx.3, xxx.4, xxx.5, xxx.6, xxx.7,
                xxx.8, xxx.9, xxx.10, xxx.11, xxx.12, xxx.13, xxx.14, xxx.15);*/

        let r: (i64,i64) = transmute(unpacked);
        //println!("{:b}", r.0);
        *chunk.0 = (r.0 | 255 << 56) | 255 << 24;
    }
}

#[target_feature(enable = "avx,avx2")]
unsafe fn blend_avx_avx2(im1: &mut [u8], im2: &[u8]) {
    let (dst_prefix, dst_arr, dst_suffix) = im1.align_to_mut::<i64>();
    let (src_prefix, src_arr, src_suffix) = im2.align_to::<i64>();
    //println!("{:?} {:?}", dst_prefix.len(), dst_suffix.len());

    let mut rgb_shuffler:[u8;32] = [
        0b10000000,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b10000000,
        0b10000000,0b00001110,
        0b10000000,0b00001101,
        0b10000000,0b00001100,
        0b10000000,0b00001010,
        0b10000000,0b00001001,
        0b10000000,0b00001000,
        0b10000000,0b00000110,
        0b10000000,0b00000101,
        0b10000000,0b00000100,
        0b10000000,0b00000010,
        0b10000000,0b00000001,
        0b10000000,0b00000000
    ];
    rgb_shuffler.reverse();
    let rgb_shuffler = _mm256_load_si256(rgb_shuffler.as_ptr() as *const std::arch::x86_64::__m256i);


    let mut alpha_shuffler: [u16; 16] =  [
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_00001111,
        0b10000000_00001111,
        0b10000000_00001111,
        0b10000000_00001011,
        0b10000000_10001011,
        0b10000000_10001011,
        0b10000000_00000111,
        0b10000000_00000111,
        0b10000000_00000111,
        0b10000000_00000011,
        0b10000000_00000011,
        0b10000000_00000011,
    ];
    alpha_shuffler.reverse();
    let alpha_shuffler = _mm256_load_si256(alpha_shuffler.as_ptr() as *const std::arch::x86_64::__m256i);

    let mut unpacker: [u16; 16] = [
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_10000000,
        0b10000000_00010111,
        0b00010101_10010011,
        0b10000000_10010001,
        0b00001111_00001101,
        0b10000000_00001011,
        0b00001001_00000111,
        0b10000000_00000101,
        0b00000011_00000001,
    ];
    unpacker.reverse();
    let unpacker = _mm256_load_si256(unpacker.as_ptr() as *const std::arch::x86_64::__m256i);

    //let dst_arr = dst_arr.chunks_exact_mut(2);
    //let src_arr = src_arr.chunks_exact(2);
    let dst_arr = dst_arr.chunks_exact_mut(2);
    let src_arr = src_arr.chunks_exact(2);
    for chunk in dst_arr.zip(src_arr)  {
        let src_pix_pack_4 = _mm256_set_epi64x(
            0,
            0,
            *chunk.1.get_unchecked(0),
            *chunk.1.get_unchecked(1)
        );

        let dst_pix_pack_4 = _mm256_set_epi64x(
            0,
            0,
            *chunk.0.get_unchecked(0),
            *chunk.0.get_unchecked(1)
        );

        let dec255 = _mm256_set1_epi16(255 as i16);

        let src_pix_pack4_shuffle = _mm256_shuffle_epi8(src_pix_pack_4, rgb_shuffler);
        let dst_pix_pack4_shuffle = _mm256_shuffle_epi8(dst_pix_pack_4, rgb_shuffler);
        let src_alpha_shuffle = _mm256_shuffle_epi8(src_pix_pack_4, alpha_shuffler);
        let src_alpha_not = _mm256_subs_epu8(dec255, src_alpha_shuffle);
        let src_premul = _mm256_mullo_epi16(src_pix_pack4_shuffle, src_alpha_shuffle);
        let dst_mul_alpha_not = _mm256_mullo_epi16(dst_pix_pack4_shuffle, src_alpha_not);
        let added = _mm256_add_epi16(src_premul, dst_mul_alpha_not);
        let unpacked = _mm256_shuffle_epi8(added, unpacker);
        //_mm256_shuffle_epi8

        /*let xxx: (u16,u16,u16,u16,u16,u16,u16,u16) = transmute(added);
        println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                 xxx.0, xxx.1, xxx.2, xxx.3, xxx.4, xxx.5, xxx.6, xxx.7);*/

        /*let xxx: (u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8,u8) = transmute(added);
        println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                xxx.0, xxx.1, xxx.2, xxx.3, xxx.4, xxx.5, xxx.6, xxx.7,
                xxx.8, xxx.9, xxx.10, xxx.11, xxx.12, xxx.13, xxx.14, xxx.15);*/

        let r: (i64,i64,i128) = transmute(unpacked);
        //println!("{:b}", r.0);
        *chunk.0.get_unchecked_mut(0) = (r.0 | 255 << 56) | 255 << 24;
        *chunk.0.get_unchecked_mut(1) = (r.1 | 255 << 56) | 255 << 24;
    }
}

fn main() {
    let w = 1920;
    let h = 1200;

    let (im1, im2) = gen_images(w, h);

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_on_floats_universal(&mut im1c, &im2);
    println!(
        "blend_on_floats_universal\t{:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_optimized_universal(&mut im1c, &im2);
    println!(
        "blend_optimized_universal\t{:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_optimized_bg_opaque(&mut im1c, &im2);
    println!(
        "blend_optimized_bg_opaque\t{:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = Instant::now();
    unsafe { blend_unsafe_bg_opaque(&mut im1c, &im2); }
    println!(
        "blend_unsafe_bg_opaque\t\t{:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    if is_x86_feature_detected!("sse2") & is_x86_feature_detected!("ssse3") {
        let mut im1c = im1.clone();
        let t = Instant::now();
        unsafe { blend_sse2_ssse3(&mut im1c, &im2); };
        println!(
            "blend_sse2_ssse3\t\t{:?} / {:?}",
            t.elapsed(),
            im1c.get_pixel(1, 2)
        );
    } else {
        println!("sse2/ssse3 not supported");
    }

    if is_x86_feature_detected!("avx") & is_x86_feature_detected!("avx2") {
        let mut im1c = im1.clone();
        let t = Instant::now();
        unsafe { blend_avx_avx2(&mut im1c, &im2); };
        println!(
            "blend_avx_avx2\t\t\t{:?} / {:?}",
            t.elapsed(),
            im1c.get_pixel(0, 0)
        );
    } else {
        println!("avx/avx2 not supported");
    }

    //let _ = Command::new("cmd.exe").arg("/c").arg("pause").status();


}
