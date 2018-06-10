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
* @param bufferSize Input array size
* @param multiplier 
* @return 0-failure, 1-success
*/
void work1(__global unsigned char* input,
           __global unsigned char* output,
           unsigned int bufferSize,
           int multiplier)
{
    int gSize = get_global_size(0);
    int gId = get_global_id(0);
    int lId = get_local_id(0);

    // Consume some time
    unsigned int tmp = 1;
    volatile int zero = 0;
    for(int i=1;i<5000;i++)
    {
        tmp += input[((i*1) % bufferSize)]
            +  input[((i*2) % bufferSize)]
            +  input[((i*3) % bufferSize)]
            +  input[((i*4) % bufferSize)]
            +  input[((i*5) % bufferSize)];
    }

    // Copy the input to output
    for(unsigned int i = gId; i < bufferSize; i += gSize)
    {
        output[i] = ((input[i] * multiplier) % 256) + (zero * tmp);
    }
}

#define MAX_LDS_USE (8 * 1024)

__kernel void work2(__global unsigned char* input,
					__global unsigned char* output,
					unsigned int bufferSize,
					int multiplier)
{
	__local unsigned char lds[MAX_LDS_USE];
    int gSize = get_global_size(0);
    int gId = get_global_id(0);
    int lId = get_local_id(0);

	// Consume some time
	int tmp = 1;
	volatile int zero = 0;
	for(int i=1;i<10000;i++)
	{
		tmp += input[((i*2) % bufferSize)]
			-  input[((i*4) % bufferSize)]
			+  input[((i*6) % bufferSize)]
			-  input[((i*8) % bufferSize)]
			+  input[((i*10) % bufferSize)];
	}

	// Copy the input to LDS then from LDS to output
	for(unsigned int i = gId; i < bufferSize; i += gSize)
	{
		lds[lId] = input[i];
		output[i] = ((lds[lId] * multiplier) % 256) + (zero * tmp);
	}
}

__kernel void K1(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
	work1(input, output, bufferSize, multiplier);
}

__kernel void K2(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work2(input, output, bufferSize, multiplier);
}

__kernel void K3(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work1(input, output, bufferSize, multiplier);
}

__kernel void K4(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work2(input, output, bufferSize, multiplier);
}

__kernel void K5(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work1(input, output, bufferSize, multiplier);
}

__kernel void K6(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work2(input, output, bufferSize, multiplier);
}

__kernel void K7(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work1(input, output, bufferSize, multiplier);
}

__kernel void K8(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work2(input, output, bufferSize, multiplier);
}

__kernel void K9(__global unsigned char* input,
                 __global unsigned char* output,
                 unsigned int bufferSize,
                 int multiplier)
{
    work1(input, output, bufferSize, multiplier);
}

__kernel void K10(__global unsigned char* input,
                  __global unsigned char* output,
                  unsigned int bufferSize,
                  int multiplier)
{
    work2(input, output, bufferSize, multiplier);
}