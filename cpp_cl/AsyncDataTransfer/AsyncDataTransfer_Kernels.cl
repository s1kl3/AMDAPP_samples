/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/**
* @brief Copy data from input array to output array with some dummy operations
         (Main purpose spent some more time on kernel execution).
* @param input      Input char array
* @param output     Output char array
* @param multiplier multiply on input value to make sure different outputs for different kernels
* @return 0-failure, 1-success
*/
__kernel void work(__global unsigned char* input,
                   __global unsigned char* output,
                   int multiplier)
{
    int gSize = get_global_size(0);
    int gId = get_global_id(0);

    // Consume some time
    unsigned int tmp = 1;
    volatile int zero = 0;
    for(int i=5;i<100;i++)
    {
        tmp += input[((i*1) % gSize)]
            +  input[((i*2) % gSize)]
            +  input[((i*3) % gSize)]
            +  input[((i*4) % gSize)]
            +  input[((i*5) % gSize)];
    }

    // Copy the input to output
    output[gId] = ((input[gId] * multiplier) % 256) + (zero * tmp);
}