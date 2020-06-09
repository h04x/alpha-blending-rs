use std::process::Command;
use image::{GenericImage, Pixel, RgbImage, Rgba, RgbaImage};
use rand::prelude::*;
use std::time::Instant;
use core::arch::x86_64::*;
use std::mem::transmute;
use ocl::{Buffer, Context, Device, Kernel, Platform, ProQue, Program, Queue, Image};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, AddressingMode, FilterMode, MemObjectType};

fn gen_images(w: u32, h: u32) -> (RgbaImage, RgbaImage) {
    let mut im1 = RgbaImage::new(w, h);
    for p in im1.pixels_mut() {
        *p = Rgba([123, 46, 75, 78])
    }
    let mut im2 = RgbaImage::new(w, h);
    for p in im2.pixels_mut() {
        *p = Rgba([44, 175, 221, 185])
    }
    (im1, im2)
}

fn blend_on_floats (im1: &mut RgbaImage, im2: &RgbaImage) {
    // build in img crate algorythm
    let max_t = u8::max_value();
    let max_t = max_t  as f32;
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

fn blend_optimized_integers (im1: &mut RgbaImage, im2: &RgbaImage, lookup: &[[u8; 256]; 256]) {
    for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) { 
        let src_r = im2p.0[0] as u32;
        let src_g = im2p.0[1] as u32;
        let src_b = im2p.0[2] as u32;
        let src_a = im2p.0[3] as u32;

        let dst_r = im1p.0[0] as u32;
        let dst_g = im1p.0[1] as u32;
        let dst_b = im1p.0[2] as u32;
        let dst_a = im1p.0[3] as u32;

        // Premul src_color/255 * src_alpha/255
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

        let alpha_final = (dst_a + src_a - ((dst_a * src_a) >> 8));

        // demul
        /*let r = ((r as f32 / alpha_final as f32) * 256f32) as u8;
        let g = ((g as f32 / alpha_final as f32) * 256f32) as u8;
        let b = ((b as f32 / alpha_final as f32) * 256f32) as u8;*/
        
        // lookup table a bit more faster
        let r = lookup[r as usize][alpha_final as usize];
        let g = lookup[g as usize][alpha_final as usize];
        let b = lookup[b as usize][alpha_final as usize];
        
        *im1p = Rgba([
            r as u8,
            g as u8,
            b as u8,
            // alpha_final some 
            if alpha_final > 255 { 255 } else { alpha_final as u8 },
        ]);
    }
}

fn blend_ocl (im1: &mut RgbaImage, im2: &RgbaImage) -> ocl::Result<(std::time::Duration)> {
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
    
    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();
    
    let program = Program::builder()
        .src(src)
        .devices(device)
        .build(&context).unwrap();        
        /*let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
            MemObjectType::Image2d).unwrap();
        println!("Image formats supported: {:?}.", sup_img_formats);*/
        let dims = im1.dimensions();

        let dst_image_writable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build().unwrap();

        let dst_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(im1)
        .queue(queue.clone())
        .build().unwrap();
        
        let src_image_readable = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(im2)
        .queue(queue.clone())
        .build().unwrap();

        let kernel = Kernel::builder()
        .program(&program)
        .name("kern_func")
        .queue(queue.clone())
        .global_work_size(&dims)
        //.arg_sampler(&sampler)
        .arg(&dst_image_writable)
        .arg(&dst_image_readable)
        .arg(&src_image_readable)
        .build().unwrap();

        unsafe { kernel.enq().unwrap(); }        
        dst_image_writable.read(im1).enq().unwrap();          

        // hope double invoke is a right way to calc time
        let t = Instant::now();
        unsafe { kernel.enq().unwrap(); }        
        dst_image_writable.read(im1).enq().unwrap();          
        let t = t.elapsed();
        Ok(t)    
}

fn blend_test(im1: &mut RgbaImage, im2: &RgbaImage) {
    // https://stackoverflow.com/a/27141669
    const AMASK: u32  = 0xFF000000;
    const RBMASK: u32 = 0x00FF00FF;
    const GMASK: u32  = 0x0000FF00;
    const AGMASK: u32 = AMASK | GMASK;
    const ONEALPHA: u32 = 0x01000000;

    for (im1p, im2p) in im1.pixels_mut().zip(im2.pixels()) {            
        let im1p32: u32 = ((im1p.0[3] as u32) << 24) | ((im1p.0[0] as u32) << 16) | ((im1p.0[1] as u32) << 8) | (im1p.0[2] as u32) ;
        let im2p32: u32 = ((im2p.0[3] as u32) << 24) | ((im2p.0[0] as u32) << 16) | ((im2p.0[1] as u32) << 8) | (im2p.0[2] as u32);
        let a: u32 = (im2p32 & AMASK) >> 24;
        let na: u32 = 255 - a;
        let rb: u32 = ((na * (im1p32 & RBMASK)) + (a * (im2p32 & RBMASK))) >> 8;
        let ag: u32 = (na * ((im1p32 & AGMASK) >> 8)) + (a * (ONEALPHA | ((im2p32 & GMASK) >> 8) as u32));
        //let res = ((rb & RBMASK) | (ag & AGMASK));        
        let a: u8 = (ag >> 24) as u8;
        let r: u8 = (rb >> 16 & 0x0000ff)  as u8;
        let g: u8 = (ag >> 8 & 0x0000ff)  as u8;        
        let b: u8 = (rb & 0x0000ff)  as u8;
        *im1p = Rgba([r,g,b,a]);

    }
}

fn main() {
    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    println!("{}", device.name().unwrap());
    
    
    let w = 1920;
    let h = 1200;

    let (im1, im2) = gen_images(w,h);
    im1.save("im1.png").unwrap();    
    im2.save("im2.png").unwrap();  

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_on_floats(&mut im1c, &im2);
    println!("blend_on_floats\t\t {:?} / {:?}", t.elapsed(), im1c.get_pixel(0,0));
    im1c.save("blend_on_floats.png").unwrap();

    let mut im1c = im1.clone();
    // prepare demul table
    let mut lookup_divisions = [[0u8; 256]; 256];
    for l in 0..255 {
        for k in 0..255 {
            lookup_divisions[l][k] = ((l as f32 / k as f32) * 256f32) as u8;
        }
    }
    let t = Instant::now();
    blend_optimized_integers(&mut im1c, &im2, &lookup_divisions);
    println!("blend_optimized_integers {:?} / {:?}", t.elapsed(), im1c.get_pixel(0,0));
    im1c.save("blend_optimized_integers.png").unwrap();

    let mut im1c = im1.clone();
    let t = blend_ocl(&mut im1c, &im2).unwrap();
    println!("blend_ocl\t\t {:?} / {:?}", t, im1c.get_pixel(0,0));
    im1c.save("im1_blended_ocl.png").unwrap();

    let mut im1c = im1.clone();
    let t = Instant::now();
    blend_test(&mut im1c, &im2);
    println!("trick blend\t\t {:?} / {:?}", t.elapsed(), im1c.get_pixel(0,0));
    im1c.save("im1_blended_trick.png").unwrap();




  
    if is_x86_feature_detected!("ssse3") {

    } else {
        println!("avx blend not supported");    
    }

    //let _ = Command::new("cmd.exe").arg("/c").arg("pause").status();

    let t = Instant::now();
    let c = unsafe {
        let mut accum = _mm_set_epi32(0,0,0,0);
        let inc = _mm_set_epi32(1,1,1,1);
        for _ in 0..1_000_000_000 {
            accum = _mm_add_epi32(accum, inc);
        }
        accum
    };
    let r: (i32, i32, i32, i32) = unsafe { transmute(c) };
    println!("trick blend {:?} {:?}", t.elapsed(), r);
 
    let t = Instant::now();
    let mut accum: (i32, i32, i32, i32) = (0, 0, 0, 0);
    for _ in 0..1_000_000_000 {
        accum.0 += 1;
        accum.1 += 1;
        accum.2 += 1;
        accum.3 += 1;
    }
    println!("trick blend {:?} {:?}", t.elapsed(), accum);

}
