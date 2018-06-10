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


#ifndef FFT_H_
#define FFT_H_

//Header Files
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

using namespace appsdk;

/**
 * FFT
 * Class implements OpenCL FFT sample
 */

class FFT
{
        cl_double
        setupTime;                         /**< time taken to setup OpenCL resources and building kernel */
        cl_double
        kernelTime;                        /**< time taken to run kernel and read result back */
        cl_float     *input_i;                          /**< Input array */
        cl_float     *input_r;                          /**< Input array */
        cl_float     *output_i;                         /**< Input array */
        cl_float     *output_r;                         /**< Input array */
        cl_float     *verificationOutput_i;             /**< Input array */
        cl_float     *verificationOutput_r;             /**< Input array */
        cl_uint      length;
        cl_context   context;                           /**< CL context */
        cl_device_id *devices;                          /**< CL device list */

        cl_mem       buffer_r;                          /**< CL memory input buffer */
        cl_mem       buffer_i;                          /**< CL memory input buffer */

        cl_command_queue commandQueue;                  /**< CL command queue */
        cl_program   program;                           /**< CL program  */
        cl_kernel    kernel;                            /**< CL kernel */
        cl_ulong availableLocalMemory;
        cl_ulong    neededLocalMemory;
        int
        iterations;                  /**< Number of iterations for kernel execution */
        SDKDeviceInfo
        deviceInfo;                       /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;                 /**< Structure to store kernel related info */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        FFT()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            input_i = NULL;
            input_r = NULL;
            output_i = NULL;
            output_r = NULL;
            verificationOutput_i = NULL;
            verificationOutput_r = NULL;
            length = 1024;
            iterations = 1;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupFFT();

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
         * Reference CPU implementation of FFT
         */
        void fftCPUReference(
            cl_float *output_r,
            cl_float *output_i,
            cl_float *input_r,
            cl_float *input_i,
            cl_uint  w);

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
         * Run OpenCL Bitonic Sort
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
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();
};
#endif
