use image::{Rgba, RgbaImage};
use ocl::enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType};
use ocl::{Context, Device, Image, Kernel, Program, Queue};
//use std::process::Command;
use core::ops::Deref;
use core::ops::DerefMut;
use std::time::Instant;

fn gen_images(w: u32, h: u32) -> (RgbaImage, RgbaImage) {
    let mut im1 = RgbaImage::new(w, h);
    for p in im1.pixels_mut() {
        *p = Rgba([101, 102, 103, 255])
    }
    let mut im2 = RgbaImage::new(w, h);
    for p in im2.pixels_mut() {
        *p = Rgba([10, 217, 100, 123])
    }
    (im1, im2)
}

fn blend_on_floats_universal(im1: &mut RgbaImage, im2: &RgbaImage) {
    // from image library
    // build in img crate algorythm
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

        let src_a_not = 256 - src_a;

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

fn blend_opencl_universal(
    im1: &mut RgbaImage,
    im2: &RgbaImage,
) -> ocl::Result<std::time::Duration> {
    let src = r#"    
    __constant sampler_t sampler_const =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_NEAREST;;
    
    __kernel void kern_func(
        write_only image2d_t dst_image_writable,
        read_only image2d_t dst_image_readable,
        read_only image2d_t src_image_readable)
    {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        float4 src_pixel = read_imagef(src_image_readable, sampler_const, coord);
        float4 dst_pixel = read_imagef(dst_image_readable, sampler_const, coord);
        float alpha_final = dst_pixel.w + src_pixel.w - dst_pixel.w * src_pixel.w;
        float bg_r_a = dst_pixel.x * dst_pixel.w;
        float bg_g_a = dst_pixel.y * dst_pixel.w;
        float bg_b_a = dst_pixel.z * dst_pixel.w;
        float fg_r_a = src_pixel.x * src_pixel.w;
        float fg_g_a = src_pixel.y * src_pixel.w;
        float fg_b_a = src_pixel.z * src_pixel.w;
        float out_r_a = fg_r_a + bg_r_a * (1.0 - src_pixel.w);
        float out_g_a = fg_g_a + bg_g_a * (1.0 - src_pixel.w);
        float out_b_a = fg_b_a + bg_b_a * (1.0 - src_pixel.w);
        float4 blended = (float4)(
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
            alpha_final
        );
        write_imagef(dst_image_writable, coord, blended);  
    }"#;

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(src)
        .devices(device)
        .build(&context)
        .unwrap();
    /*let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {:?}.", sup_img_formats);*/
    let dims = im1.dimensions();

    let dst_image_writable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build()
        .unwrap();

    let dst_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build()
        .unwrap();

    let src_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im2)
        .queue(queue.clone())
        .build()
        .unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("kern_func")
        .queue(queue)
        .global_work_size(&dims)
        //.arg_sampler(&sampler)
        .arg(&dst_image_writable)
        .arg(&dst_image_readable)
        .arg(&src_image_readable)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }
    dst_image_writable.read(im1).enq().unwrap();

    // hope double invoke is a right way to calc time
    let t = Instant::now();
    unsafe {
        kernel.enq().unwrap();
    }
    dst_image_writable.read(im1).enq().unwrap();
    let t = t.elapsed();
    Ok(t)
}

fn blend_opencl_bg_opaque(
    im1: &mut RgbaImage,
    im2: &RgbaImage,
) -> ocl::Result<std::time::Duration> {
    let src = r#"    
    __constant sampler_t sampler_const =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_NEAREST;;
    
    __kernel void kern_func(
        write_only image2d_t dst_image_writable,
        read_only image2d_t dst_image_readable,
        read_only image2d_t src_image_readable)
    {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        float4 src_pixel = read_imagef(src_image_readable, sampler_const, coord);
        float4 dst_pixel = read_imagef(dst_image_readable, sampler_const, coord);
        float4 blended = mix(dst_pixel, src_pixel, src_pixel.w);
        blended.w = 255;
        write_imagef(dst_image_writable, coord, blended);  
    }"#;

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(src)
        .devices(device)
        .build(&context)
        .unwrap();
    let dims = im1.dimensions();

    let dst_image_writable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build()
        .unwrap();

    let dst_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build()
        .unwrap();

    let src_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(im2)
        .queue(queue.clone())
        .build()
        .unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("kern_func")
        .queue(queue)
        .global_work_size(&dims)
        //.arg_sampler(&sampler)
        .arg(&dst_image_writable)
        .arg(&dst_image_readable)
        .arg(&src_image_readable)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }
    dst_image_writable.read(im1).enq().unwrap();

    // hope double invoke is a right way to calc time
    let t = Instant::now();
    unsafe {
        kernel.enq().unwrap();
    }
    dst_image_writable.read(im1).enq().unwrap();
    let t = t.elapsed();
    Ok(t)
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

fn blend_unsafe_bg_opaque(im1: &mut RgbaImage, im2: &RgbaImage) {
    let w = im1.width();
    let h = im1.height();
    let im1_raw = im1.deref_mut();
    let im2_raw = im2.deref();
    let len = (w * h) / 4;
    for i in 0..len as usize {
        let src_a = unsafe { *im2_raw.get_unchecked(i * 4 + 3) as u32 };
        let src_a_not = 255 - src_a;
        {
            let dst_r = unsafe { im1_raw.get_unchecked_mut(i * 4) };
            let src_r = unsafe { *im2_raw.get_unchecked(i * 4) as u32 };
            let r = ((src_r * src_a) + (src_a_not * *dst_r as u32)) >> 8;
            *dst_r = r as u8;
        }
        {
            let dst_g = unsafe { im1_raw.get_unchecked_mut(i * 4 + 1) };
            let src_g = unsafe { *im2_raw.get_unchecked(i * 4 + 1) as u32 };
            let g = ((src_g * src_a) + (src_a_not * *dst_g as u32)) >> 8;
            *dst_g = g as u8;
        }
        {
            let dst_b = unsafe { im1_raw.get_unchecked_mut(i * 4 + 2) };
            let src_b = unsafe { *im2_raw.get_unchecked(i * 4 + 2) as u32 };
            let b = ((src_b * src_a) + (src_a_not * *dst_b as u32)) >> 8;
            *dst_b = b as u8;
        }
    }
}

fn blend_raw(im1_raw: &mut Vec<u8>, im2_raw: &Vec<u8>) {
    for i in 0..im1_raw.len() / 4 {
        let src_a = unsafe { *im2_raw.get_unchecked(i * 4 + 3) as u32 };
        let src_a_not = 255 - src_a;
        {
            let dst_r = unsafe { im1_raw.get_unchecked_mut(i * 4) };
            let src_r = unsafe { *im2_raw.get_unchecked(i * 4) as u32 };
            let r = ((src_r * src_a) + (src_a_not * *dst_r as u32)) >> 8;
            *dst_r = r as u8;
        }
        {
            let dst_g = unsafe { im1_raw.get_unchecked_mut(i * 4 + 1) };
            let src_g = unsafe { *im2_raw.get_unchecked(i * 4 + 1) as u32 };
            let g = ((src_g * src_a) + (src_a_not * *dst_g as u32)) >> 8;
            *dst_g = g as u8;
        }
        {
            let dst_b = unsafe { im1_raw.get_unchecked_mut(i * 4 + 2) };
            let src_b = unsafe { *im2_raw.get_unchecked(i * 4 + 2) as u32 };
            let b = ((src_b * src_a) + (src_a_not * *dst_b as u32)) >> 8;
            *dst_b = b as u8;
        }
    }
}

fn main() {
    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    println!("{}", device.name().unwrap());

    let w = 1920;
    let h = 1200;

    let (im1, im2) = gen_images(w, h);
    im1.save("im1.png").unwrap();
    im2.save("im2.png").unwrap();

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_on_floats_universal(&mut im1c, &im2);
    println!(
        "blend_on_floats_universal {:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_optimized_universal(&mut im1c, &im2);
    println!(
        "blend_optimized_universal {:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = blend_opencl_universal(&mut im1c, &im2).unwrap();
    println!(
        "blend_opencl_universal\t {:?} / {:?}",
        t,
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = blend_opencl_bg_opaque(&mut im1c, &im2).unwrap();
    println!(
        "blend_opencl_bg_opaque\t {:?} / {:?}",
        t,
        im1c.get_pixel(0, 0)
    );
    im1c.save("blend_opencl_bg_opaque.png").unwrap();

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_optimized_bg_opaque(&mut im1c, &im2);
    println!(
        "blend_optimized_bg_opaque {:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_unsafe_bg_opaque(&mut im1c, &im2);
    println!(
        "blend_unsafe_bg_opaque {:?} / {:?}",
        t.elapsed(),
        im1c.get_pixel(0, 0)
    );

    let mut im1: Vec<u8> = Vec::with_capacity(((w * h) as usize * 4) as usize);
    let mut im2: Vec<u8> = Vec::with_capacity(((w * h) as usize * 4) as usize);
    for _ in 0..w * h {
        im1.push(101);
        im1.push(102);
        im1.push(103);
        im1.push(255);

        im2.push(10);
        im2.push(217);
        im2.push(100);
        im2.push(123);
    }
    let t = Instant::now();
    blend_raw(&mut im1, &im2);
    let elapsed = t.elapsed();
    println!(
        "simpel vectors {:?} / {:?}",
        elapsed,
        (im1[4], im1[5], im1[6], im1[7])
    );
}
