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

#ifndef BUFFER_IMAGE_INTEROP_H_
#define BUFFER_IMAGE_INTEROP_H_

#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define INPUT_IMAGE "BufferImageInterop_Input.bmp"
#define OUTPUT_IMAGE "BufferImageInterop_Output.bmp"

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"


class BufferImageInterop
{
        cl_context context;
        cl_device_id *devices;
        cl_command_queue commandQueue;
        cl_program program;
        cl_kernel  sepiaKernel,imageReverseKernel;

        cl_mem inputImageBuffer;
        cl_mem outputImageBuffer;

        SDKBitMap inputBitmap;

        uchar4 *pixelData;
        cl_uchar4 *inputImageData;
        cl_uchar4 *outputImageData;
        cl_uint pixelSize;
        cl_uint width;
        cl_uint height;

        uchar4 *verificationInput;
        uchar4 *verificationOutput;
        SDKTimer *sampleTimer;

        int iterations;
        double totalKernelTime;

        cl_image_format imageFormat;
        cl_image_desc imageDesc;

        cl_mem inputImage;
        cl_mem outputImage;

        SDKDeviceInfo deviceInfo;

    public:

        CLCommandArgs *sampleArgs;


        BufferImageInterop()
        {
            sampleTimer = new SDKTimer();
            sampleArgs = new CLCommandArgs();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            iterations =1;
        }


        /**
        * Override from SDKSample, Generate binary image of given kernel
        * and exit application
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int genBinaryImage();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCL();
        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runCLKernels();

        /**
        * CPU verification
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int CPUReference();

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run OpenCL BinarySearch
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * sampleArgs->verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();
        /**
         * Read the input image
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int readInputImage(std::string inputImageName);

        /**
         * write the output image
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int writeOutputImage(std::string outputImageName);

        /**
         * copy data storage from a buffer object to an image object.
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int copyBufferToImage(cl_mem Buffer, cl_mem Image);

        /**
        * clEnqueueMapBuffer
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);

        /**
         * clEnqueueUnmapMemObject
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);
};

#endif
