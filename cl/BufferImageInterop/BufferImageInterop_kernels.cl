/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void sepiaToning(__global uchar4 *inputImage,
                          __global uchar4 *outputImage)
{
    int dim = get_global_id(0);
    int dim1 = get_global_id(1);

    int width = get_global_size(0);

    int red, green, blue;

    int index = dim + width * dim1;

    red = inputImage[index].x;
    green = inputImage[index].y;
    blue = inputImage[index].z;

    float grayscale = (0.3 * red + 0.59 * green + 0.11 * blue);

// red = green = blue = grayscale;

    float depth = 1.8;

    red = (int)(grayscale + depth * 56.6 );
    if (red > 255)
    {
        red = 255;
    }

    green = (int)(grayscale + depth * 33.3 );
    if (green > 255)
    {
        green = 255;
    }

    blue = (int)(grayscale + depth * 10.1);

    if (blue > 255)
    {
        blue = 255;
    }
    if (dim == 1 && dim1 == 0)
    {
        //printf("blue_gpu= %f, gray= %f", (float)( grayscale + depth * 33.3) , (float)grayscale);
    }
    outputImage[index].x = red;
    outputImage[index].y = green;
    outputImage[index].z = blue;
    outputImage[index].w = inputImage[index].w;
}


__kernel void imageReverse(__read_only image2d_t inputImage,
                           __write_only image2d_t outputImage)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    width--;
    int2 coord;
    coord.x = x;
    coord.y = y;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP |
                        CLK_FILTER_NEAREST;

    uint4 pixel = read_imageui(inputImage, sampler, coord);

    coord.x = width-x;

    write_imageui(outputImage, coord, pixel);
}
