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


#include "BinarySearch.hpp"
#include <malloc.h>

/*
 * \brief set up program input data
 */
int BinarySearch::setupBinarySearch()
{
    // allocate and init memory used by host
    cl_uint max = length * 20;
    cl_uint inputSizeBytes = length *  sizeof(cl_uint);

    int status = mapBuffer( inputBuffer, input, inputSizeBytes,
                            CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(inputBuffer in setupBinarySearch)");

    // random initialisation of input
    input[0] = 0;
    for(cl_uint i = 1; i < length; i++)
    {
        input[i] = input[i - 1] + (cl_uint) ((max * rand()) / (float)RAND_MAX);
    }

    /*
     * Unless sampleArgs->quiet mode has been enabled, print the INPUT array.
     */
    if(!sampleArgs->quiet)
    {
        printArray<cl_uint>(
            "Sorted Input",
            input,
            length,
            1);
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(inputBuffer in setupBinarySearch)");


    return SDK_SUCCESS;
}

template<typename T>
int BinarySearch::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
                            size_t sizeInBytes, cl_map_flags flags)
{
    cl_int status;
    hostPointer = (T*) clEnqueueMapBuffer(commandQueue,
                                          deviceBuffer,
                                          CL_TRUE,
                                          flags,
                                          0,
                                          sizeInBytes,
                                          0,
                                          NULL,
                                          NULL,
                                          &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed");

    return SDK_SUCCESS;
}

int
BinarySearch::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    return SDK_SUCCESS;
}

int
BinarySearch::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("BinarySearch_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


/*
 * \brief OpenCL related initialisations are done here.
 *        Context, Device list, Command Queue are set up.
 *        Calls are made to set up OpenCL memory buffers that this program uses
 *        and to load the programs into memory and get kernel handles.
 *          Load and build OpenCL program and get kernel handles.
 *        Set up OpenCL memory buffers used by this program.
 */

int
BinarySearch::setupCL(void)
{
    cl_int status = 0;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */

    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


    // If we could find our platform, use it. Otherwise use just available platform.

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = clCreateCommandQueue(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "clCreateCommandQueue failed.");
    }

    localThreads[0] = 256;
    numSubdivisions = length / (cl_uint)localThreads[0];

    if(numSubdivisions < localThreads[0])
    {
        numSubdivisions = (cl_uint)localThreads[0];
    }

    inlength = numSubdivisions*2*sizeof(cl_uint);

    inputBuffer = clCreateBuffer(
                      context,
                      CL_MEM_READ_ONLY,
                      sizeof(cl_uint) * inlength,
                      NULL,
                      &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer)");

    outputBuffer = clCreateBuffer(
                       context,
                       CL_MEM_WRITE_ONLY,
                       sizeof(cl_uint4),
                       NULL,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("BinarySearch_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    kernel = clCreateKernel(program, "binarySearch", &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

    return SDK_SUCCESS;
}

int
BinarySearch::runCLKernels(void)
{
    cl_int   status;

    size_t globalThreads[1];
    globalThreads[0] = numSubdivisions;

    // Check group size against kernelWorkGroupSize
    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "clGetKernelWorkGroupInfo failed.");

    if((cl_uint)(localThreads[0]) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << localThreads[0] << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Changing the group size to "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
        }

        localThreads[0] = kernelInfo.kernelWorkGroupSize;
        numSubdivisions = length / (cl_uint)localThreads[0];
        if(numSubdivisions < localThreads[0])
        {
            numSubdivisions = (cl_uint)localThreads[0];
        }
        globalThreads[0] = numSubdivisions;
    }


    /**
     * Since a plain binary search on the GPU would not achieve much benefit over the GPU
     * we are doing an N'ary search. We split the array into N segments every pass and therefore
     * get log (base N) passes instead of log (base 2) passes.
     *
     * In every pass, only the thread that can potentially have the element we are looking for
     * writes to the output array. For ex: if we are looking to find 4567 in the array and every
     * thread is searching over a segment of 1000 values and the input array is 1, 2, 3, 4,...
     * then the first thread is searching in 1 to 1000, the second one from 1001 to 2000, etc.
     * The first one does not write to the output. The second one doesn't either. The fifth one however is from
     * 4001 to 5000. So it can potentially have the element 4567 which lies between them.
     *
     * This particular thread writes to the output the lower bound, upper bound and whether the element equals the lower bound element.
     * So, it would be 4001, 5000, 0
     *
     * The next pass would subdivide 4001 to 5000 into smaller segments and continue the same process from there.
     *
     * When a pass returns 1 in the third element, it means the element has been found and we can stop executing the kernel.
     * If the element is not found, then the execution stops after looking at segment of size 1.
     */


    globalLowerBound = 0;
    globalUpperBound = length - 1;
    cl_uint subdivSize = (globalUpperBound - globalLowerBound + 1) /
                         numSubdivisions;
    isElementFound = 0;
    cl_uint inputSizeBytes = length *  sizeof(cl_uint);

    status = mapBuffer( inputBuffer, input, inputSizeBytes, CL_MAP_READ);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(inputBuffer in setupBinarySearch");

    if((input[0] > findMe) || (input[length - 1] < findMe))
    {
        status = unmapBuffer( inputBuffer, input);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

        return SDK_SUCCESS;
    }
    status = mapBuffer( outputBuffer, output, sizeof(cl_uint4), CL_MAP_WRITE );
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuffer)");

    output[3] = 1;
    status = unmapBuffer( outputBuffer, output);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(outputBuffer)");

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(inputBuffer in setupBinarySearch)");


    // Set appropriate arguments to the kernel

    /*
     * First argument of the kernel is the output buffer
     */
    status = clSetKernelArg(
                 kernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg 0(OutputBuffer) failed.");

    status = mapBuffer( outputBuffer, output, sizeof(cl_uint4), CL_MAP_WRITE );
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuffer)");

    while(subdivSize > 1 && output[3] != 0)
    {
        /*select the data that will be used in the kernel and only write these data to the input buffer*/
        cl_uint *in=NULL;

        // Set input data
        status = mapBuffer( inputBuffer, in, inlength, CL_MAP_WRITE );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

        for(cl_uint i=0 ; i<numSubdivisions; i++)
        {
            int indexa = i*subdivSize;
            int indexb = (i+1)*subdivSize-1;
            in[2*i] = input[indexa];
            in[2*i+1] = input[indexb];
        }

        status = unmapBuffer( inputBuffer, in);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

        /*
        * Second argument is input buffer
        */
        status = clSetKernelArg(
                     kernel,
                     1,
                     sizeof(cl_mem),
                     (void *)&inputBuffer);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg 1(inputBuffer) failed.");

        /*
        * Third is the element we are looking for
        */
        status = clSetKernelArg(
                     kernel,
                     2,
                     sizeof(cl_uint),
                     (void *)&findMe);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg 2(findMe) failed.");
        output[3] = 0;

        status = unmapBuffer( outputBuffer, output );
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(outputBuffer)");

        /*
        * Enqueue a kernel run call
        */
        cl_event ndrEvt;
        status = clEnqueueNDRangeKernel(commandQueue,
                                        kernel,
                                        1,
                                        NULL,
                                        globalThreads,
                                        localThreads,
                                        0,
                                        NULL,
                                        &ndrEvt);
        CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush failed.");

        status = waitForEventAndRelease(&ndrEvt);
        CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

        status = mapBuffer( outputBuffer, output, sizeof(cl_uint4), CL_MAP_WRITE );
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputBuffer)");

        globalLowerBound = output[0]*subdivSize;
        globalUpperBound = globalLowerBound+subdivSize-1;
        subdivSize = (globalUpperBound - globalLowerBound + 1)/numSubdivisions;
    }

    status = mapBuffer( inputBuffer, input, inputSizeBytes, CL_MAP_READ);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    for(cl_uint i=globalLowerBound; i<= globalUpperBound; i++)
    {
        if(input[i] == findMe)
        {
            elementIndex = i;
            globalLowerBound = i;
            globalUpperBound = i+1;
            isElementFound = 1;
            break;
        }
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    status = unmapBuffer( outputBuffer, output );
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(outputBuffer)");

    return SDK_SUCCESS;
}


/**
 * CPU verification for the BinarySearch algorithm
 */
int
BinarySearch::binarySearchCPUReference()
{
    cl_uint inputSizeBytes = length *  sizeof(cl_uint);

    int status = mapBuffer( inputBuffer, input, inputSizeBytes, CL_MAP_WRITE);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to map device buffer.(inputBuffer in binarySearchCPUReference)");

    if(isElementFound)
    {
        if(input[globalLowerBound] == findMe)
        {
            return SDK_SUCCESS;
        }
        else
        {
            return SDK_FAILURE;
        }
    }
    else
    {
        for(cl_uint i = 0; i < length; i++)
        {
            if(input[i] == findMe)
            {
                return SDK_FAILURE;
            }
        }
        return SDK_SUCCESS;
    }

    status = unmapBuffer( inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed to unmap device buffer.(inputBuffer in binarySearchCPUReference)");
}

int BinarySearch::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    // Now add customized options
    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error.\n");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Lenght of the input array";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;

    sampleArgs->AddOption(array_length);

    delete array_length;

    Option* find_me = new Option;
    CHECK_ALLOCATION(find_me, "Memory allocation error.\n");

    find_me->_sVersion = "f";
    find_me->_lVersion = "find";
    find_me->_description = "element to be found";
    find_me->_type = CA_ARG_INT;
    find_me->_value = &findMe;
    sampleArgs->AddOption(find_me);

    delete find_me;

    Option* sub_div = new Option;
    CHECK_ALLOCATION(sub_div, "Memory allocation error.\n");

    sub_div->_sVersion = "s";
    sub_div->_lVersion = "subdivisions";
    sub_div->_description = "number of subdivisions";
    sub_div->_type = CA_ARG_INT;
    sub_div->_value = &numSubdivisions;
    sampleArgs->AddOption(sub_div);

    delete sub_div;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);

    delete num_iterations;

    return SDK_SUCCESS;
}

int BinarySearch::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    //length should bigger then 0 and  divisible by 256
    if(length<=0)
    {
        length = 512;
    }
    length = length>256? (length/256*256):length;

    if(!isPowerOf2(length))
    {
        length = roundToPowerOf2(length);
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(setupBinarySearch() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    setupTime = (cl_double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int BinarySearch::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations <<
              " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer));


    if(!sampleArgs->quiet)
    {
        std::cout << "Lower bound = " << globalLowerBound << ", Upper bound = " <<
                  globalUpperBound <<
                  ", Element to be searched = " << findMe << std::endl;
        if(isElementFound == 1)
        {
            std::cout<<"Element found at index "<<elementIndex;
        }
        else
        {
            std::cout<<"Element not found\n";
        }

    }

    return SDK_SUCCESS;
}

int BinarySearch::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationInput = (cl_uint *) malloc(length * sizeof(cl_int));
        CHECK_ALLOCATION(verificationInput,
                         "Failed to allocate host memory. (verificationInput)");

        int status = mapBuffer( inputBuffer, input, length * sizeof(cl_int),
                                CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to map device buffer.(inputBuffer in sampleArgs->verify)");

        memcpy(verificationInput, input, length*sizeof(cl_int));

        status = unmapBuffer( inputBuffer, input );
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed to unmap device buffer.(inputBuffer in sampleArgs->verify)");

        /* reference implementation
         * it overwrites the input array with the output
         */
        int refTimer = sampleTimer->createTimer();
        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        cl_int verified = binarySearchCPUReference();
        sampleTimer->stopTimer(refTimer);
        referenceKernelTime =sampleTimer->readTimer(refTimer);

        // compare the results and see if they match
        if(verified == SDK_SUCCESS)
        {
            std::cout << "\nPassed!" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void BinarySearch::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Setup Time (sec)", "Avg. Kernel Time (sec)", "Elements/sec"};
        std::string stats[4];

        sampleTimer->totalTime = ( totalKernelTime / iterations );

        stats[0] = toString(length   , std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString( sampleTimer->totalTime, std::dec);
        stats[3] = toString(length/sampleTimer->totalTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int BinarySearch::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)

    FREE(devices);

    FREE(verificationInput);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    BinarySearch clBinarySearch;

    if(clBinarySearch.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBinarySearch.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBinarySearch.sampleArgs->isDumpBinaryEnabled())
    {
        return clBinarySearch.genBinaryImage();
    }
    else
    {
        if(clBinarySearch.setup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBinarySearch.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBinarySearch.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBinarySearch.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clBinarySearch.printStats();
    }

    return SDK_SUCCESS;
}
