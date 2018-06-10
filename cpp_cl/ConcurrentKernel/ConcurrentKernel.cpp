/**********************************************************************
Copyright ©2012 Advanced Micro Devices, Inc. All rights reserved.

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

#include "ConcurrentKernel.hpp"

int ConcurrentKernel::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initilization failed");
    Option* buffer_size = new Option;
    CHECK_ALLOCATION(buffer_size, "Memory allocation error.\n");

    buffer_size->_sVersion = "x";
    buffer_size->_lVersion = "size";
    buffer_size->_description = "Buffer size in Bytes per Kernel";
    buffer_size->_type = CA_ARG_INT;
    buffer_size->_value = &bufferSize;

    sampleArgs->AddOption(buffer_size);
    delete buffer_size;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    Option* num_kernels = new Option;
    CHECK_ALLOCATION(num_kernels, "Memory allocation error.\n");

    num_kernels->_sVersion = "k";
    num_kernels->_lVersion = "kernels";
    num_kernels->_description = "Number of Kernels for concurrent execution";
    num_kernels->_type = CA_ARG_INT;
    num_kernels->_value = &numKernels;

    sampleArgs->AddOption(num_kernels);
    delete num_kernels;

    Option* num_workgroups = new Option;
    CHECK_ALLOCATION(num_workgroups, "Memory allocation error.\n");

    num_workgroups->_sVersion = "w";
    num_workgroups->_lVersion = "workgroups";
    num_workgroups->_description = "Number of WorkGroups per Kernel execution";
    num_workgroups->_type = CA_ARG_INT;
    num_workgroups->_value = &numWGsPerKernel;

    sampleArgs->AddOption(num_workgroups);
    delete num_workgroups;

    Option* local_size = new Option;
    CHECK_ALLOCATION(local_size, "Memory allocation error.\n");

    local_size->_sVersion = "l";
    local_size->_lVersion = "localsize";
    local_size->_description =
        "Number of Work items per Work Group(should be 2 ^ N)";
    local_size->_type = CA_ARG_INT;
    local_size->_value = &localSize;

    sampleArgs->AddOption(local_size);
    delete local_size;
    
    Option* num_queues = new Option;
    CHECK_ALLOCATION(num_queues, "Memory allocation error.\n");

    num_queues->_sVersion = "u";
    num_queues->_lVersion = "queues";
    num_queues->_description =
        "Number of Work items per Work Group(should be 2 ^ N)";
    num_queues->_type = CA_ARG_INT;
    num_queues->_value = &numQueues;

    sampleArgs->AddOption(num_queues);
    delete num_queues;

    return SDK_SUCCESS;
}

int ConcurrentKernel::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("ConcurrentKernel_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int ConcurrentKernel::setupCL(void)
{
    int status = 0;
    cl_device_type dType;

    if(sampleArgs-> deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //sampleArgs-> deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs-> isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    // Get platform
    cl_platform_id platform = NULL;
    status = getPlatform(platform, sampleArgs->platformId,
                         sampleArgs->isPlatformEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    status = displayDevices(platform, dType);
    CHECK_ERROR(status, SDK_SUCCESS, "displayDevices() failed");

    // If we could find our platform, use it. Otherwise use just available platform.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = cl::Context(dType, cps, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "Context::Context() failed.");

    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CHECK_ERROR((devices.size() > 0), true, "No device available");

    if(validateDeviceId(sampleArgs-> deviceId, (int)devices.size()))
    {
        error("validateDeviceId() failed");
        return SDK_FAILURE;
    }

    // Set device info of given sampleArgs-> deviceId
    status = setDeviceInfo(devices[sampleArgs-> deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Set parameters
    if(deviceInfo.maxComputeUnits > 1)
    {
        numComputeUnits = deviceInfo.maxComputeUnits;
    }
    if(numKernels < 2)
    {
        numKernels = 2;
    }
    if(bufferSize == 0)
    {
        bufferSize = BYTES_PER_MB;
    }

    if(!sampleArgs->quiet)
    {
        std::cout << "\n Sample parameters :" << std::endl;
        std::cout << "\t Local Size : " << localSize << std::endl;
        std::cout << "\t Num WGs Per Kernel : " << numWGsPerKernel << std::endl;
        std::cout << "\t Available Compute Units : " << numComputeUnits << std::endl;
        std::cout << "\t Num Kernels : " << numKernels << std::endl;
        std::cout << "\t Num Queues  : " << numQueues << std::endl;
        std::cout << "\t Buffer size per Kernel(Bytes) : " << bufferSize << std::endl <<
                  std::endl;
    }

    commandQueue = new cl::CommandQueue[numQueues];
    for(int i = 0; i < numQueues; ++i)
    {
        commandQueue[i] = cl::CommandQueue(context, devices[sampleArgs-> deviceId],
                                           NULL, &status);
        CHECK_OPENCL_ERROR(status, "cl::CommandQueue() failed.");
    }

    SDKFile kernelFile;
    std::string kernelPath = getPath();
    if(sampleArgs-> isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs-> loadBinary.c_str());
        if(kernelFile.readBinaryFromFile(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }
        cl::Program::Binaries programBinary(1,std::make_pair(
                                                (const void*)kernelFile.source().data(),
                                                kernelFile.source().size()));

        program = cl::Program(context, devices, programBinary, NULL, &status);
        CHECK_OPENCL_ERROR(status, "Program::Program(Binary) failed.");
    }
    else
    {
        kernelPath.append("ConcurrentKernel_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        cl::Program::Sources programSource(1,
                                           std::make_pair(kernelFile.source().data(),
                                                   kernelFile.source().size()));

        program = cl::Program(context, programSource, &status);
        CHECK_OPENCL_ERROR(status, "Program::Program(Source) failed.");
    }

    // Get additional options
    std::string flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load sampleArgs->flags file: " << flagsPath <<
                      std::endl;
            return SDK_FAILURE;
        }
        flagsFile.replaceNewlineWithSpaces();
        const char * flags = flagsFile.source().c_str();
        flagsStr.append(flags);
    }

    if(flagsStr.size() != 0)
    {
        std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
    }

    status = program.build(devices, flagsStr.c_str());
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>
                              (devices[sampleArgs-> deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
    }
    CHECK_OPENCL_ERROR(status, "Program::build() failed.");

    // Initialize the kernel objects
    kernel = new cl::Kernel[numKernels];
    for(int i = 0; i < numKernels; ++i)
    {
        std::stringstream ss;
        ss << ((i % NUM_CL_KERNELS) +1);
        std::string kernelName = "K" + ss.str();
        kernel[i] = cl::Kernel(program, kernelName.c_str(), &status);
        CHECK_OPENCL_ERROR(status, "cl::Kernel(consumeTimeAndCopy) failed.");
    }

    // Check localSize and should be power of 2.
    if((localSize > deviceInfo.maxWorkGroupSize) || (localSize < 2))
    {
        localSize = (cl_uint)deviceInfo.maxWorkGroupSize;
    }
    if(!isPowerOf2(localSize))
    {
        localSize = roundToPowerOf2(localSize);
    }
    localThreads = localSize;
    globalThreads = localSize * numWGsPerKernel;

    // Create device buffers
    inputBuffer = new cl::Buffer[numKernels];
    outputBuffer = new cl::Buffer[numKernels];
    for(int k = 0; k < numKernels; ++k)
    {
        inputBuffer[k] = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSize, NULL,
                                    &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(inputBuffer1) failed.");

        outputBuffer[k] = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL,
                                     &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(outputBuffer1) failed.");
    }

    // Initialize device buffers
    cl_uchar *ptr0  = (cl_uchar*)commandQueue[0].enqueueMapBuffer(inputBuffer[0],
                      CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                      0, bufferSize, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inputBuffer) failed.");
    for(cl_uint i = 0; i < bufferSize; ++i)
    {
        ptr0[i] = rand() % 256;
    }
    for(int k = 1; k < numKernels; ++k)
    {
        cl_uchar *ptr  = (cl_uchar*)commandQueue[0].enqueueMapBuffer(inputBuffer[k],
                         CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                         0, bufferSize, NULL, NULL, &status);
        CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inputBuffer) failed.");

        memcpy(ptr, ptr0, bufferSize);

        status = commandQueue[0].enqueueUnmapMemObject(inputBuffer[k], ptr);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");
    }
    status = commandQueue[0].enqueueUnmapMemObject(inputBuffer[0], ptr0);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");

    return SDK_SUCCESS;
}

int ConcurrentKernel::setup()
{
    if (iterations == 0)
    {
        iterations = 1;
    }
    cpuTimer = sampleTimer->createTimer();

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    sampleTimer-> stopTimer(cpuTimer);
    setupTime = sampleTimer-> readTimer(cpuTimer) * 1000;

    return SDK_SUCCESS;
}

int ConcurrentKernel::runSequentialKernels(void)
{
    int status = SDK_SUCCESS;

    // Set arguments for all kernels
    for(int i = 0; i < numKernels; ++i)
    {
        status |= kernel[i].setArg(0, inputBuffer[i]);
        status |= kernel[i].setArg(1, outputBuffer[i]);
        status |= kernel[i].setArg(2, bufferSize);
        status |= kernel[i].setArg(3, i);
    }
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed runSequentialKernels::kernel[i].setArg()");

    // Clear all queues
    for(int i = 0; i < numQueues; ++i)
    {
        status |= commandQueue[i].finish();
    }
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed runSequentialKernels::commandQueue[i].finish()");

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    
    // Run Kernels for all iterations
    for(int i = 0; i < iterations; ++i)
    {
        // Run kernels
        for(int k = 0; k < numKernels; ++k)
        {
            status |= commandQueue[0].enqueueNDRangeKernel(kernel[k], cl::NullRange,
                      globalThreads, localThreads, NULL, NULL);
            // Flush the queue
            status |= commandQueue[0].flush();
            // Wait for all kernels to finish.
            status |= commandQueue[0].finish();
        }
    }
    sampleTimer->stopTimer(cpuTimer);
    seqTime = sampleTimer-> readTimer(cpuTimer) * 1000;
    seqTime = seqTime / iterations;
    CHECK_ERROR(status, SDK_SUCCESS, "Failed Sequential Kernels run");

    return SDK_SUCCESS;
}

int ConcurrentKernel::runConcurrentKernels(void)
{
    int status = SDK_SUCCESS;
    cl_ulong startTime = 0;
    cl_ulong endTime = 0;

    // Set arguments for all kernels
    for(int i = 0; i < numKernels; ++i)
    {
        status |= kernel[i].setArg(0, inputBuffer[i]);
        status |= kernel[i].setArg(1, outputBuffer[i]);
        status |= kernel[i].setArg(2, bufferSize);
        status |= kernel[i].setArg(3, i);
    }
    CHECK_ERROR(status, SDK_SUCCESS, "Failed kernel[i].setArg()");

    // Clear all queues
    for(int i = 0; i < numQueues; ++i)
    {
        status |= commandQueue[i].finish();
    }
    CHECK_ERROR(status, SDK_SUCCESS, "Failed commandQueue[i].finish()");

    // Create events
    cl::Event *events = new cl::Event[iterations * numKernels];
    eventList.reserve(1);

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    
    // Run Kernels
    // 1. Execute all kernels for 1st iteration(Not depend on wait event)
    for(int k = 0; k < numKernels; ++k)
    {
        status |= commandQueue[k % numQueues].enqueueNDRangeKernel(kernel[k],
                  cl::NullRange, globalThreads, localThreads,
                  NULL, &(events[k]));
    }
    // 2. Execute all kernels for (N-1) iterations.
    // kernels wait till previous iteration event to finish.
    for(int i = 1; i < iterations; ++i)
    {
        for(int k = 0; k < numKernels; ++k)
        {
            eventList.clear();
            eventList.push_back(events[((i-1)*numKernels)+k]);
            status |= commandQueue[k % numQueues].enqueueNDRangeKernel(kernel[k],
                      cl::NullRange, globalThreads, localThreads,
                      &eventList, &(events[(i*numKernels)+k]));
        }
    }
    // Flush all queues
    for(int i = 0; i < numQueues; ++i)
    {
        status |= commandQueue[i].flush();
    }
    // Wait till all kernels to finish
    for(int i = 0; i < numQueues; ++i)
    {
        status |= commandQueue[i].finish();
    }
    sampleTimer->stopTimer(cpuTimer);
    conTime = sampleTimer-> readTimer(cpuTimer) * 1000;
    conTime = conTime / iterations;
    delete [] events;
    CHECK_ERROR(status, SDK_SUCCESS, "Failed Conquerrent Kernel runs");

    return SDK_SUCCESS;
}

int ConcurrentKernel::runCLKernels(void)
{
    std::cout << "\n\n Running Sequential Kernel Version  ...";
    CHECK_ERROR(runSequentialKernels(), SDK_SUCCESS, "Failed runSequentialKernels");
    if(sampleArgs->verify)
    {
        std::cout << "\n Sequential Kernel verification : ";
    }
    if(verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    std::cout << "\n\n Running Concurrent Kernel Version ... ";
    CHECK_ERROR(runConcurrentKernels(), SDK_SUCCESS, "Failed runConcurrentKernels");
    if(sampleArgs->verify)
    {
        std::cout << "\n Concurrent Kernel verification : ";
    }
    if(verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int ConcurrentKernel::run()
{
    // Warm up
    std::cout << "\n Warm up ..." << std::endl;
    if (runSequentialKernels() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    std::cout << "\n Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Arguments are set and execution call is enqueued on command buffer
    if (runCLKernels() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int ConcurrentKernel::cpuReferenceImpl(int numBuffer)
{
    int status;
    cl_uchar *ptr  = (cl_uchar*)commandQueue[0].enqueueMapBuffer(
                         inputBuffer[numBuffer],  CL_TRUE, CL_MAP_READ,
                         0, bufferSize, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inputBuffer) failed.");

    for(size_t i = 0; i < bufferSize; i ++)
    {
        verificationOutput[i] = (ptr[i] * numBuffer) % 256;
    }

    status = commandQueue[0].enqueueUnmapMemObject(inputBuffer[numBuffer], ptr);
    CHECK_OPENCL_ERROR(status, "Failed to unmap device buffer.(inputBuffer)");

    return SDK_SUCCESS;
}

int ConcurrentKernel::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationOutput = (cl_uchar *) malloc(bufferSize);
        int status;

        for(int k = 0; k < numKernels; ++k)
        {
            cl_uchar *ptr  = (cl_uchar*)commandQueue[0].enqueueMapBuffer(outputBuffer[k],
                             CL_TRUE, CL_MAP_READ,
                             0, bufferSize, NULL, NULL, &status);
            CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(outputBuffer) failed.");

            // CPU reference implementation
            CHECK_ERROR(cpuReferenceImpl(k), SDK_SUCCESS, "Failed cpuReferenceImpl");

            //sampleArgs->verify
            for(size_t i = 0; i < bufferSize; ++i)
            {
                if(verificationOutput[i] != ptr[i])
                {
                    std::cout << "Failed! on outputBuffer[" << k << "]\n" << std::endl;
                    commandQueue[0].enqueueUnmapMemObject(outputBuffer[k], ptr);
                    return SDK_FAILURE;
                }
            }
            status = commandQueue[0].enqueueUnmapMemObject(outputBuffer[k], ptr);
            CHECK_ERROR(status, SDK_SUCCESS,
                        "Failed to unmap device buffer.(outputBuffer)");
        }
        std::cout << "Passed!\n" << std::endl;
    }

    return SDK_SUCCESS;
}

void ConcurrentKernel::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Size(Bytes)",
            "Setup Time(ms)",
            "Avg. Seq time(ms)",
            "Avg. Con time(ms)"
        };
        std::string stats[4];

        stats[0] = toString(bufferSize * numKernels, std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString(seqTime, std::dec);
        stats[3] = toString(conTime, std::dec);
        printStatistics(strArray, stats, 4);
    }
}
int ConcurrentKernel::cleanup()
{
    devices.clear();
    eventList.clear();

    FREE(verificationOutput);
    delete [] kernel;
    delete [] commandQueue;
    delete [] inputBuffer;
    delete [] outputBuffer;

    return SDK_SUCCESS;
}

int ConcurrentKernel::setDeviceInfo(cl::Device& device)
{
    cl_int status = CL_SUCCESS;

    //Get device type
    status = device.getInfo(CL_DEVICE_TYPE, &this->deviceInfo.dType);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_TYPE) failed");

    //Get vender ID
    status = device.getInfo(CL_DEVICE_VENDOR_ID,&this->deviceInfo.venderId);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_VENDOR_ID) failed");

    //Get max compute units
    status = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                            &this->deviceInfo.maxComputeUnits);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_COMPUTE_UNITS) failed");

    //Get max work item dimensions
    status = device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            &this->deviceInfo.maxWorkItemDims);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) failed");

    // Maximum work group size
    status = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            &this->deviceInfo.maxWorkGroupSize);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed");

    // Preferred vector sizes of all data types
    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                            &this->deviceInfo.preferredCharVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                            &this->deviceInfo.preferredShortVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                            &this->deviceInfo.preferredIntVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                            &this->deviceInfo.preferredLongVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                            &this->deviceInfo.preferredFloatVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                            &this->deviceInfo.preferredDoubleVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                            &this->deviceInfo.preferredHalfVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) failed");

    // Clock frequency
    status = device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            &this->deviceInfo.maxClockFrequency);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_CLOCK_FREQUENCY) failed");

    // Address bits
    status = device.getInfo(CL_DEVICE_ADDRESS_BITS,&this->deviceInfo.addressBits);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_ADDRESS_BITS) failed");

    // Maximum memory alloc size
    status = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            &this->deviceInfo.maxMemAllocSize);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_MEM_ALLOC_SIZE) failed");

    return SDK_SUCCESS;
}

int main(int argc, char * argv[])
{
    ConcurrentKernel cke;
    if(cke.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(cke.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(cke.sampleArgs->isDumpBinaryEnabled())
    {
        return cke.genBinaryImage();
    }

    if(cke.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(cke.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    cke.printStats();

    if(cke.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}
