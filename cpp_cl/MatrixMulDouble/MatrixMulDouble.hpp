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


#ifndef MATRIXMULDOUBLE_H_
#define MATRIXMULDOUBLE_H_

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

/**
 * MatrixMulDouble
 * Class implements OpenCL Matrix Multiplication sample
 */

class MatrixMulDouble
{
        cl_uint
        seed;      /**< Seed value for random number generation */
        cl_double           setupTime;      /**< Time for setting up OpenCL */
        cl_double             appTime;      /**< Time for transfer + kernel execution */
        cl_double          kernelTime;      /**< Time for kernel execution */
        cl_double              *inputA;      /**< Input array */
        cl_int                 widthA;      /**< width of input Array */
        cl_int                heightA;      /**< height of input Array */
        cl_double             *inputB;      /**< Input array */
        cl_int                 widthB;      /**< width of Input Array */
        cl_int                heightB;      /**< height of Input Array */
        cl_double             *output;      /**< Output Array */
        cl_double*
        refOutput;      /**< Output array for reference implementation */
        cl_uint
        blockSize;      /**< Size of the block used for shared memory */
        cl::Context           context;      /**< CL context */
        std::vector<cl::Device> devices;    /**< CL device list */
        std::vector<cl::Device> device;     /**< CL device to be used */
        std::vector<cl::Platform> platforms;    /**< vector of platforms */
        cl::Buffer          inputBufA;      /**< CL memory buffer  for matrix inputA*/
        cl::Buffer          inputBufB;      /**< CL memory buffer  for matrix inputB*/
        cl::Buffer
        outputBuf;      /**< CL memory buffer  for storing the output*/
        cl::CommandQueue commandQueue;      /**< CL command queue */
        cl::Program           program;      /**< CL program  */
        cl::Kernel             kernel;      /**< CL kernel */

        bool                      lds;      /**< Local data store availability */

        cl_int                      n;      /**< for command line args */
        cl_int                      m;
        cl_int                      k;

        size_t       maxWorkGroupSize;      /**< Device Specific Information */
        size_t    kernelWorkGroupSize;      /**< Group Size returned by kernel */
        cl_uint         maxDimensions;
        size_t *     maxWorkItemSizes;
        cl_ulong     totalLocalMemory;
        cl_ulong      usedLocalMemory;
        cl_ulong availableLocalMemory;
        cl_ulong    neededLocalMemory;
        int
        iterations;      /**< Number of iterations for kernel execution */
        bool eAppGFLOPS;

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        MatrixMulDouble()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            seed   = 123;
            inputA = NULL;
            inputB = NULL;
            output = NULL;
            refOutput = NULL;
            n = 64;
            m = 64;
            k = 64;
            blockSize = 8;
            setupTime = 0;
            appTime = 0;
            kernelTime = 0;
            iterations = 1;
            lds = false;
            eAppGFLOPS = false;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupMatrixMulDouble();

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
         * Reference CPU implementation of Matrix Multiplication
         * @param output stores the output of the multiplied matrices depthxheight
         * @param inputA input matrix of size width x height
         * @param inputB input matrix of size depth x width
         * @param height height of the output matrix
         * @param width  length of the common dimension of the matrices inputA and inputB
         * @param depth  width  of the output matrix
         */
        void MatrixMulDoubleCPUReference(
            cl_double * output,
            cl_double * inputA,
            cl_double * inputB,
            const cl_uint height,
            const cl_uint width,
            const cl_uint depth);
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
         * Run OpenCL Matrix Multiplication
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
