// clang-cl.exe /O2 blend.cpp -o blend.exe

#include <iostream>
#include <chrono>
#include <emmintrin.h>

void blend_background_opaque(unsigned char* im1, unsigned char* im2, unsigned w, unsigned h) {
    for(unsigned long i=0; i < w*h*4; i+=4) {
        unsigned short src_r = im2[i];
        unsigned short src_g = im2[i+1];
        unsigned short src_b = im2[i+2];
        unsigned short src_a = im2[i+3];

        unsigned short dst_r = im1[i];
        unsigned short dst_g = im1[i+1];
        unsigned short dst_b = im1[i+2];

        unsigned short src_a_not = 255 - im2[i+3];

        im1[i] = ((src_r * src_a) + (src_a_not * dst_r)) >> 8;
        im1[i+1] = ((src_g * src_a) + (src_a_not * dst_g)) >> 8;
        im1[i+2] = ((src_b * src_a) + (src_a_not * dst_b)) >> 8;
    }
}

void blend_background_opaque2(unsigned char* im1, unsigned char* im2, unsigned w, unsigned h) {
    for(unsigned long i=0; i < w*h*4; i+=4*8) {
        /*unsigned src_r = im2[i];
        unsigned src_g = im2[i+1];
        unsigned src_b = im2[i+2];
        unsigned src_a = im2[i+3];

        unsigned dst_r = im1[i];
        unsigned dst_g = im1[i+1];
        unsigned dst_b = im1[i+2];*/

        __m128i src_r = _mm_setr_epi16(
            (short)im2[i],
            (short)im2[i+4],
            (short)im2[i+8],
            (short)im2[i+12],
            (short)im2[i+16],
            (short)im2[i+20],
            (short)im2[i+24],
            (short)im2[i+28]);

        __m128i src_g = _mm_setr_epi16(
            (short)im2[i+1],
            (short)im2[i+5],
            (short)im2[i+9],
            (short)im2[i+13],
            (short)im2[i+17],
            (short)im2[i+21],
            (short)im2[i+25],
            (short)im2[i+29]);

        __m128i src_b = _mm_setr_epi16(
            (short)im2[i+2],
            (short)im2[i+6],
            (short)im2[i+10],
            (short)im2[i+14],
            (short)im2[i+18],
            (short)im2[i+22],
            (short)im2[i+26],
            (short)im2[i+30]);

        __m128i src_a = _mm_setr_epi16(
            (short)im2[i+3],
            (short)im2[i+7],
            (short)im2[i+11],
            (short)im2[i+15],
            (short)im2[i+19],
            (short)im2[i+23],
            (short)im2[i+27],
            (short)im2[i+31]);

        __m128i src_a_not = _mm_setr_epi16(
            (short)(255 - im2[i+3]),
            (short)(255 - im2[i+7]),
            (short)(255 - im2[i+11]),
            (short)(255 - im2[i+15]),
            (short)(255 - im2[i+19]),
            (short)(255 - im2[i+23]),
            (short)(255 - im2[i+27]),
            (short)(255 - im2[i+31]));

        __m128i dst_r = _mm_setr_epi16(
            (short)im1[i],
            (short)im1[i+4],
            (short)im1[i+8],
            (short)im1[i+12],
            (short)im1[i+16],
            (short)im1[i+20],
            (short)im1[i+24],
            (short)im1[i+28]);

        __m128i dst_g = _mm_setr_epi16(
            (short)im1[i+1],
            (short)im1[i+5],
            (short)im1[i+9],
            (short)im1[i+13],
            (short)im1[i+17],
            (short)im1[i+21],
            (short)im1[i+25],
            (short)im1[i+29]);

        __m128i dst_b = _mm_setr_epi16(
            (short)im1[i+2],
            (short)im1[i+6],
            (short)im1[i+10],
            (short)im1[i+14],
            (short)im1[i+18],
            (short)im1[i+22],
            (short)im1[i+26],
            (short)im1[i+30]);

        __m128i src_r_premul = _mm_mullo_epi16(src_r, src_a);
        __m128i src_g_premul = _mm_mullo_epi16(src_g, src_a);
        __m128i src_b_premul = _mm_mullo_epi16(src_b, src_a);

        __m128i dst_r_mul_src_a_not = _mm_mullo_epi16(dst_r, src_a_not);
        __m128i dst_g_mul_src_a_not = _mm_mullo_epi16(dst_g, src_a_not);
        __m128i dst_b_mul_src_a_not = _mm_mullo_epi16(dst_b, src_a_not);

        __m128i added_r = _mm_add_epi16(src_r_premul, dst_r_mul_src_a_not);
        __m128i added_g = _mm_add_epi16(src_g_premul, dst_g_mul_src_a_not);
        __m128i added_b = _mm_add_epi16(src_b_premul, dst_b_mul_src_a_not);

        im1[i] = ((unsigned char*)&added_r)[1];
        im1[i+4] = ((unsigned char*)&added_r)[3];
        im1[i+8] = ((unsigned char*)&added_r)[5];
        im1[i+12] = ((unsigned char*)&added_r)[7];
        im1[i+16] = ((unsigned char*)&added_r)[9];
        im1[i+20] = ((unsigned char*)&added_r)[11];
        im1[i+24] = ((unsigned char*)&added_r)[13];
        im1[i+28] = ((unsigned char*)&added_r)[15];

        im1[i+1] = ((unsigned char*)&added_g)[1];
        im1[i+5] = ((unsigned char*)&added_g)[3];
        im1[i+9] = ((unsigned char*)&added_g)[5];
        im1[i+13] = ((unsigned char*)&added_g)[7];
        im1[i+17] = ((unsigned char*)&added_g)[9];
        im1[i+21] = ((unsigned char*)&added_g)[11];
        im1[i+25] = ((unsigned char*)&added_g)[13];
        im1[i+29] = ((unsigned char*)&added_g)[15];

        im1[i+2] = ((unsigned char*)&added_b)[1];
        im1[i+6] = ((unsigned char*)&added_b)[3];
        im1[i+10] = ((unsigned char*)&added_b)[5];
        im1[i+14] = ((unsigned char*)&added_b)[7];
        im1[i+18] = ((unsigned char*)&added_b)[9];
        im1[i+22] = ((unsigned char*)&added_b)[11];
        im1[i+26] = ((unsigned char*)&added_b)[13];
        im1[i+30] = ((unsigned char*)&added_b)[15];

        //unsigned src_a_not = 255 - im2[i+3];

        //printf("%d\n", ((unsigned char*)&src_b)[2]);


    }
}

int main() {
    unsigned w = 1920;
    unsigned h = 1200;

    unsigned char* im1 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 4);
    unsigned char* im2 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 4);
    unsigned char* im1_cpy = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 4);

    for(unsigned long i=0; i < w*h*4; i+=4) {
        im1[i] = 101; 
        im1[i+1] = 102;
        im1[i+2] = 103;
        im1[i+3] = 255;
    }

        for(unsigned long i=0; i < w*h*4; i+=4) {
        im2[i] = 10; 
        im2[i+1] = 217;
        im2[i+2] = 100;
        im2[i+3] = 123;
    }

    memcpy(im1_cpy, im1, sizeof(unsigned char) * w * h * 4);
    auto begin = std::chrono::high_resolution_clock::now();
    blend_background_opaque(im1_cpy, im2, w, h);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "microseconds " << elapsed.count() * 1e-9
        << " " << (int)im1_cpy[0]
        << " " << (int)im1_cpy[1]
        << " " << (int)im1_cpy[2]
        << " " << (int)im1_cpy[3]
        << std::endl;

    memcpy(im1_cpy, im1, sizeof(unsigned char) * w * h * 4);
    begin = std::chrono::high_resolution_clock::now();
    blend_background_opaque2(im1_cpy, im2, w, h);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "microseconds " << elapsed.count() * 1e-9
        << " " << (int)im1_cpy[0]
        << " " << (int)im1_cpy[1]
        << " " << (int)im1_cpy[2]
        << " " << (int)im1_cpy[3]
        << std::endl;
    return 0;
}