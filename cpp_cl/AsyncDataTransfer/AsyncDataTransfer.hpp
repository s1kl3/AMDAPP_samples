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


#ifndef ASYNCDATATRANSFER_H_
#define ASYNCDATATRANSFER_H_

#include "CLUtil.hpp"
#include <CL/cl.hpp>

#define BYTES_PER_MB (1024 * 1024)
#define WORKGROUP_SIZE 64

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

using namespace appsdk;

/**
 * AsyncDataTransfer Class implementation
 * Derived from SDKSample base class
 */
class AsyncDataTransfer
{
        int                           cpuTimer;    /**< Timer */
        double                       setupTime;    /**< Time for setting up OpenCL */
        double                        syncTime;    /**< Sync Time DMA + kernel  */
        double                       asyncTime;    /**< Async Time DMA + kernel */

        unsigned int                bufferSize;    /**< Buffer size in MB */
        int
        iterations;    /**< Number of iteration for kernel execution */
        cl_uchar
        **input;    /**< map pointer to input pinBuffer1  */
        cl_uchar
        **output;    /**< map pointer to output pinBuffer1 */
        cl_uchar
        *verificationOutput;    /**< Input array for reference implementation1 */
        int                         numKernels;

        std::vector<cl::Device>        devices;    /**< OpenCL device list */

        cl::NDRange              globalThreads;    /**< Global threads  */
        cl::NDRange               localThreads;    /**< Work group size */
        cl::Context                    context;    /**< OpenCL context  */
        cl::Program                    program;    /**< OpenCL program  */
        cl::Kernel                      kernel;    /**< OpenCL kernel   */
        cl::CommandQueue           kernelQueue,    /**< OpenCL kernel command queue  */
        readQueue,    /**< OpenCL Read command queue  */
        writeQueue;    /**< OpenCL Write command queue  */
        cl::Buffer                *inputBuffer,    /**< Input CL buffer */
        *outputBuffer,    /**< Output CL buffer */
        *inPrepinBuffer,
        *outPrepinBuffer;

        SDKDeviceInfo    deviceInfo;    /**< Structure to store device information*/

        SDKTimer                    *sampleTimer;
    public:
        CLCommandArgs              *sampleArgs;
        /**
         * Constructor
         * Initialize member variables
         */
        AsyncDataTransfer()
            :
            bufferSize(1 * BYTES_PER_MB),
            iterations(100),
            input(NULL),
            output(NULL),
            verificationOutput(NULL),
            numKernels(3),
            inputBuffer(NULL),
            outputBuffer(NULL),
            inPrepinBuffer(NULL),
            outPrepinBuffer(NULL)
        {
            sampleTimer = new SDKTimer();
            sampleArgs =  new CLCommandArgs();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


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
         * Reference CPU implementation for performance comparison
         */
        int cpuReferenceImpl(int numBuffer);

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
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
         * Override from SDKSample, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run OpenCL AsyncDataTransfer
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

        /**
         * Execute Synchronous kernel execution with DMA transfer
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int executeSyncKernel();

        /**
         * Execute Asynchronous kernel execution with DMA transfer
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int executeAsyncKernel();

        /**
         * Set device information
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setDeviceInfo(cl::Device& device);
};

#endif
