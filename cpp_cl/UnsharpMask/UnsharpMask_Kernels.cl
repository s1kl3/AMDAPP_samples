/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

__constant sampler_t imageSampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE;



__kernel 
void unsharp_mask(__read_only image2d_t input, __write_only image2d_t output
                            , __global float* gaussianKernel, int gaussianKernelRadius
                            , float threshold, float amount) {

    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);

    float4 pixel = 0.0f;
    int2 tc;
    tc.y = coords.y - gaussianKernelRadius;
    for (int j=0; j<=2*gaussianKernelRadius; j++, tc.y++) {
        tc.x = coords.x - gaussianKernelRadius;
        for (int i=0; i<=2*gaussianKernelRadius; i++, tc.x++) {
            float4 tPixel = convert_float4(read_imageui(input, imageSampler, tc));
            pixel.xyz += tPixel.xyz*gaussianKernel[j*(gaussianKernelRadius*2+1)+i];
        }
    }

    float4 currentPixel = convert_float4(read_imageui(input, imageSampler, coords));

    float diffx = currentPixel.x-pixel.x;
    currentPixel.x +=  fabs(diffx) > threshold ? diffx*amount:0.0f;

    float diffy = currentPixel.y-pixel.y;
    currentPixel.y += fabs(diffy) > threshold ? diffy*amount:0.0f;

    float diffz = currentPixel.z-pixel.z;
    currentPixel.z += fabs(diffz) > threshold ? diffz*amount:0.0f;

    currentPixel = clamp(currentPixel+0.5f, 0.0f, 255.0f);
    write_imageui(output, coords, convert_uint4(currentPixel));
}


__kernel
void unsharp_mask_pass1(__read_only image2d_t input, __global float4* temp
                , int width
                , __global float* gaussianKernel, int gaussianKernelRadius) {
    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);

    float4 pixel = 0.0f;
    int2 tc;
    tc.y = coords.y;
    tc.x = coords.x - gaussianKernelRadius;
    for (int i=0; i<=2*gaussianKernelRadius; i++, tc.x++) {
        float4 tPixel = convert_float4(read_imageui(input, imageSampler, tc));
        pixel.xyz += tPixel.xyz*gaussianKernel[i];
    }
    temp[coords.y*width+coords.x] = pixel;
}


__kernel 
void unsharp_mask_pass2(__read_only image2d_t input
                , __global float4* temp
                , __write_only image2d_t output
                    , int width, int height
                , __global float* gaussianKernel, int gaussianKernelRadius
                , float threshold, float amount) {
    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);

    float4 pixel = 0.0f;
    int2 tc;
    tc.y = coords.y - gaussianKernelRadius;
    tc.x = coords.x;
    for (int i=0; i<=2*gaussianKernelRadius; i++, tc.y++) {
          int2 clamped_tc;
        clamped_tc.x = tc.x;
        clamped_tc.y = clamp(tc.y, 0, height-1);
        float4 tPixel = temp[clamped_tc.y * width + clamped_tc.x];
        pixel.xyz += tPixel.xyz*gaussianKernel[i];
    }

    float4 currentPixel = convert_float4(read_imageui(input, imageSampler, coords));

    float diffx = currentPixel.x-pixel.x;
    currentPixel.x +=  fabs(diffx) > threshold ? diffx*amount:0.0f;

    float diffy = currentPixel.y-pixel.y;
    currentPixel.y += fabs(diffy) > threshold ? diffy*amount:0.0f;

    float diffz = currentPixel.z-pixel.z;
    currentPixel.z += fabs(diffz) > threshold ? diffz*amount:0.0f;

    currentPixel = clamp(currentPixel+0.5f, 0.0f, 255.0f);
    write_imageui(output, coords, convert_uint4(currentPixel));
}


__kernel 
void unsharp_mask_filter(__global uchar4* input, __global uchar4* output
                                , int width, int height
                                , __global float* gaussianKernel, int gaussianKernelRadius
                                , float threshold, float amount) {

    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);

    float4 pixel = 0.0f;
    int2 tc;
    tc.y = coords.y - gaussianKernelRadius;
    for (int j=0; j<=2*gaussianKernelRadius; j++, tc.y++) {
        tc.x = coords.x - gaussianKernelRadius;
        for (int i=0; i<=2*gaussianKernelRadius; i++, tc.x++) {

            int2 clamped_tc;
            clamped_tc.x = clamp(tc.x, 0, width-1);
            clamped_tc.y = clamp(tc.y, 0, height-1);
            float4 tPixel = convert_float4(input[clamped_tc.y*width+clamped_tc.x]);
            pixel.xyz += tPixel.xyz*gaussianKernel[j*(gaussianKernelRadius*2+1)+i];
        }
    }

    float4 currentPixel = convert_float4(input[coords.y*width+coords.x]);

    float diffx = currentPixel.x-pixel.x;
    currentPixel.x +=  fabs(diffx) > threshold ? diffx*amount:0.0f;

    float diffy = currentPixel.y-pixel.y;
    currentPixel.y += fabs(diffy) > threshold ? diffy*amount:0.0f;

    float diffz = currentPixel.z-pixel.z;
    currentPixel.z += fabs(diffz) > threshold ? diffz*amount:0.0f;

    currentPixel = clamp(currentPixel+0.5f, 0.0f, 255.0f);

    output[coords.y*width+coords.x] = convert_uchar4(currentPixel);
}