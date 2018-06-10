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

#define BYTES_PER_MB   (1024 * 1024)
#define NUM_CL_KERNELS 10

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

using namespace appsdk;

/**
 * ConcurrentKernel Class implementation
 * Derived from SDKSample base class
 */
class ConcurrentKernel
{
        int                           cpuTimer;    /**< Timer */
        double                       setupTime;    /**< Time for setting up OpenCL */
        double                         seqTime;    /**< Sequential kernel run time */
        double                         conTime;    /**< Concurrent kernel run time */

        cl_uint                     bufferSize;    /**< Buffer size in Bytes */
        int                         iterations;    /**< Number of iteration for kernel execution */
        cl_uchar           *verificationOutput;    /**< CPU verification output result */

        int                         numKernels;    /**< Number of kernels */
        int                          numQueues;    /**< Number of Queues for kernels */
        int                    numWGsPerKernel;    /**< Number of WorkGroups per kernel run */
        cl_uint                      localSize;
        int                    numComputeUnits;    /**< Number of compute units in the GPU */

        std::vector<cl::Device>        devices;    /**< OpenCL device list */
        std::vector<cl::Event>       eventList;    /**< Event list */

        cl::NDRange              globalThreads;    /**< Global threads  */
        cl::NDRange               localThreads;    /**< Work group size */
        cl::Context                    context;    /**< OpenCL context  */
        cl::Program                    program;    /**< OpenCL program  */
        cl::Kernel                     *kernel;    /**< OpenCL kernel   */
        cl::CommandQueue         *commandQueue;    /**< OpenCL kernel command queue  */
        cl::Buffer                *inputBuffer,    /**< Input CL buffer */
                                 *outputBuffer;    /**< Output CL buffer */

        SDKDeviceInfo               deviceInfo;    /**< Structure to store device information*/
        SDKTimer                  *sampleTimer;
        
    public:
        CLCommandArgs              *sampleArgs;
        /**
         * Constructor
         * Initialize member variables
         */
        ConcurrentKernel()
            :
            bufferSize(BYTES_PER_MB),
            iterations(5),
            verificationOutput(NULL),
            numKernels(10),
            numQueues(2),
            numWGsPerKernel(10),
            localSize(64),
            kernel(NULL),
            commandQueue(NULL),
            inputBuffer(NULL),
            outputBuffer(NULL)
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
         * @param numBuffer Buffer number to generate CPU output data.
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
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
         * Run OpenCL ConcurrentKernel
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
         * Execute kernels sequentially
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runSequentialKernels();

        /**
         * Execute kernels parallely in different queues
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runConcurrentKernels();

        /**
         * Set device information
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setDeviceInfo(cl::Device& device);
};

#endif
