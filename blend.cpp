// clang-cl.exe /O2 blend.cpp -o blend.exe

#include <iostream>
#include <chrono>

void blend_background_opaque(unsigned char* im1, unsigned char* im2, unsigned w, unsigned h) {
    for(unsigned long i=0; i < (w*h / 4); i+=4) {
        unsigned src_r = im2[i];
        unsigned src_g = im2[i+1];
        unsigned src_b = im2[i+2];
        unsigned src_a = im2[i+3];

        unsigned dst_r = im1[i];
        unsigned dst_g = im1[i+1];
        unsigned dst_b = im1[i+2];

        unsigned src_a_not = 255 - im2[i+3];

        im1[i] = ((src_r * src_a) + (src_a_not * dst_r)) >> 8;
        im1[i+1] = ((src_g * src_a) + (src_a_not * dst_g)) >> 8;
        im1[i+2] = ((src_b * src_a) + (src_a_not * dst_b)) >> 8;
    }
}

int main() {
    unsigned w = 1920;
    unsigned h = 1200;

    unsigned char* im1 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 4);
    unsigned char* im2 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 4);

    for(unsigned long i=0; i < (w*h / 4); i+=4) {
        im1[i] = 101; 
        im1[i+1] = 102;
        im1[i+2] = 103;
        im1[i+3] = 255;
    }

        for(unsigned long i=0; i < (w*h / 4); i+=4) {
        im2[i] = 10; 
        im2[i+1] = 217;
        im2[i+2] = 100;
        im2[i+3] = 123;
    }

    auto begin = std::chrono::high_resolution_clock::now();
    blend_background_opaque(im1, im2, w, h);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "microseconds " << elapsed.count() * 1e-6
        << " " << (int)im1[0]
        << " " << (int)im1[1]
        << " " << (int)im1[2]
        << " " << (int)im1[3]
        << std::endl;
    return 0;
}